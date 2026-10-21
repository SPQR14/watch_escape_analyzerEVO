"""Unit and headless PyQt6 integration tests for AudioRingBuffer, DSPStreamWorker, and MainWin."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from wea.core.models import BeatTelemetry, WatchConfig
from wea.core.ring_buffer import AudioRingBuffer
from wea.core.synthetic import SyntheticEscapeGenerator
from wea.ui.dsp_worker import DSPStreamWorker
from wea.ui.main_window import MainWin

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_core_modules_free_of_pyqt6_imports() -> None:
    """Verify strict separation of concerns: src/wea/core/ must contain zero PyQt6 imports."""
    core_dir = Path(__file__).resolve().parents[1] / "src" / "wea" / "core"
    for py_file in core_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("PyQt6"), (
                        f"Forbidden PyQt6 import in {py_file}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("PyQt6"), f"Forbidden PyQt6 import in {py_file}"


def test_audio_ring_buffer_wrap_around_and_overflow_ordering() -> None:
    """Verify AudioRingBuffer wrap-around boundaries, exact sample ordering, and counters."""
    rb = AudioRingBuffer(capacity_samples=100)
    assert rb.capacity_samples == 100
    assert rb.available_samples == 0
    assert rb.total_samples_written == 0
    assert len(rb.read_latest(20)) == 0

    # 1. Write 60 samples (no wrap yet)
    chunk_a = np.arange(0, 60, dtype=np.float64)
    rb.write(chunk_a)
    assert rb.available_samples == 60
    assert rb.total_samples_written == 60
    np.testing.assert_array_equal(rb.read_latest(60), chunk_a)
    np.testing.assert_array_equal(rb.read_latest(25), np.arange(35, 60, dtype=np.float64))

    # 2. Write 70 samples (wraps around boundary 100 -> 30, total written = 130)
    chunk_b = np.arange(60, 130, dtype=np.float64)
    rb.write(chunk_b)
    assert rb.available_samples == 100
    assert rb.total_samples_written == 130
    np.testing.assert_array_equal(rb.read_latest(100), np.arange(30, 130, dtype=np.float64))
    np.testing.assert_array_equal(rb.read_latest(45), np.arange(85, 130, dtype=np.float64))

    # 3. Write another 85 samples (wraps across boundary again, total written = 215)
    chunk_c = np.arange(130, 215, dtype=np.float64)
    rb.write(chunk_c)
    assert rb.available_samples == 100
    assert rb.total_samples_written == 215
    np.testing.assert_array_equal(rb.read_latest(100), np.arange(115, 215, dtype=np.float64))

    # 4. Write a chunk larger than capacity_samples (150 samples > 100)
    chunk_d = np.arange(215, 365, dtype=np.float64)
    rb.write(chunk_d)
    assert rb.available_samples == 100
    assert rb.total_samples_written == 365
    np.testing.assert_array_equal(rb.read_latest(100), np.arange(265, 365, dtype=np.float64))
    np.testing.assert_array_equal(rb.read_latest(30), np.arange(335, 365, dtype=np.float64))

    # 5. Write a small chunk after overflow to confirm pointer continuity
    chunk_e = np.arange(365, 380, dtype=np.float64)
    rb.write(chunk_e)
    assert rb.available_samples == 100
    assert rb.total_samples_written == 380
    np.testing.assert_array_equal(rb.read_latest(100), np.arange(280, 380, dtype=np.float64))

    # 6. Clear/reset buffer
    rb.clear()
    assert rb.available_samples == 0
    assert rb.total_samples_written == 0


def test_dsp_stream_worker_headless_synthetic_chunks(qtbot: object) -> None:
    """Feed synthetic chunks for 21600 BPH / 53.0 deg into DSPStreamWorker in a QThread."""
    app = QApplication.instance() or QApplication([])
    assert app is not None

    config = WatchConfig(bph=21600, lift_angle=53.0, sample_rate=48000)
    true_rate_spd = 5.0
    true_be_ms = 0.40
    true_amp_deg = 275.0

    gen = SyntheticEscapeGenerator(config=config, seed=42)
    full_signal = gen.generate(
        duration_s=4.5,
        true_rate_spd=true_rate_spd,
        true_beat_error_ms=true_be_ms,
        true_amplitude_deg=true_amp_deg,
        snr_db=40.0,
    )

    thread = QThread()
    worker = DSPStreamWorker(
        config=config,
        buffer_duration_s=6.0,
        window_duration_s=4.0,
        chunk_size=4096,
        synthetic_mode=False,
        workers=2,
    )
    worker.moveToThread(thread)

    received_telemetry: list[BeatTelemetry] = []
    received_waveforms: list[tuple[np.ndarray, np.ndarray]] = []
    received_drifts: list[tuple[float, float, float]] = []
    received_errors: list[str] = []

    worker.telemetry_ready.connect(lambda t: received_telemetry.append(t))
    worker.waveform_ready.connect(lambda x, y: received_waveforms.append((x, y)))
    worker.drift_updated.connect(lambda fs, ppm, bias: received_drifts.append((fs, ppm, bias)))
    worker.error_occurred.connect(lambda msg: received_errors.append(msg))

    thread.start()
    try:
        worker.set_calibration_ppm(12.5)
        chunk_size = 4096
        base_ns = 1_000_000_000_000
        chunk_dt_ns = int(round((chunk_size / float(config.sample_rate)) * 1e9))

        for idx, start in enumerate(range(0, len(full_signal), chunk_size)):
            chunk = full_signal[start : start + chunk_size]
            t_ns = base_ns + idx * chunk_dt_ns
            worker.push_audio_chunk(chunk, timestamp_ns=t_ns)

        app.processEvents()

        assert not received_errors
        assert len(received_drifts) > 0
        eff_fs, drift_ppm, bias_spd = received_drifts[-1]
        assert drift_ppm == pytest.approx(12.5, abs=0.5)
        assert eff_fs == pytest.approx(48000.0 * (1.0 + 12.5e-6), rel=1e-5)
        assert bias_spd == pytest.approx(86400.0 * 12.5e-6, abs=0.05)

        assert len(received_waveforms) > 0
        t_axis, wave_arr = received_waveforms[-1]
        assert len(t_axis) == len(wave_arr) > 0

        assert len(received_telemetry) > 0
        latest = received_telemetry[-1]
        assert isinstance(latest, BeatTelemetry)
        assert len(latest.beat_times_sec) >= 15
        assert latest.confidence > 0.85
        assert latest.snr_db > 25.0
        assert len(latest.beats) >= 15
        assert len(latest.micro_envelope) == len(latest.envelope)
        # With +12.5 ppm effective_fs (~+1.08 s/d shift relative to nominal 48000 Hz):
        expected_compensated_rate = true_rate_spd + 86400.0 * 12.5e-6
        assert abs(latest.rate_spd - expected_compensated_rate) < 0.35
        assert abs(latest.beat_error_ms - true_be_ms) < 0.08
        assert abs(latest.amplitude_deg - true_amp_deg) < 5.0
    finally:
        worker.shutdown()
        thread.quit()
        assert thread.wait(3000) is True
        assert not thread.isRunning()


def test_main_window_dsp_worker_integration_and_clean_shutdown() -> None:
    """Verify MainWin wires UI controls, telemetry_ready, and drift_updated to DSPStreamWorker."""
    app = QApplication.instance() or QApplication([])
    assert app is not None

    win = MainWin()
    try:
        assert win.dsp_thread.isRunning()
        assert win.current_watch_config().sample_rate == 48000

        # Changing caliber preset updates BPH, Lift Angle, and DSPStreamWorker config in real time
        win.cmb_caliber.setCurrentText("Seiko NHXX/4RXX series")
        assert win.dsp_worker.config.bph == 21600
        assert win.dsp_worker.config.lift_angle == 53.0

        # Changing sample rate updates DSPStreamWorker config and drift estimator nominal_fs
        win.cmb_srate.setCurrentIndex(1)  # 96 kHz
        assert win.dsp_worker.config.sample_rate == 96000
        assert win.dsp_worker.drift_estimator.nominal_fs == 96000.0
        win.cmb_srate.setCurrentIndex(0)  # back to default 48 kHz
        assert win.dsp_worker.config.sample_rate == 48000
        assert win.dsp_worker.drift_estimator.nominal_fs == 48000.0

        # Feed synthetic audio into win.dsp_worker and verify UI updates via signals
        gen = SyntheticEscapeGenerator(config=win.dsp_worker.config, seed=99)
        synth_sig = gen.generate(
            duration_s=3.5,
            true_rate_spd=4.2,
            true_beat_error_ms=0.30,
            true_amplitude_deg=280.0,
            snr_db=40.0,
        )
        for start in range(0, len(synth_sig), 8192):
            win.dsp_worker.push_audio_chunk(synth_sig[start : start + 8192])

        app.processEvents()

        assert win.card_rate.value_label.text() != "---.-"
        assert win.card_beat_error.value_label.text() != "---.-"
        assert win.card_amplitude.value_label.text() != "---.-"

        # Verify SNR gauge is populated (> 0 dB and > 0%)
        assert win.prg_snr.value() > 50
        assert "0.0 dB (0%)" not in win.lbl_snr_value.text()
        assert "dB (" in win.lbl_snr_value.text()

        # Verify Paper Tape trace curves
        x_even, y_even = win.tape_curve_even.getData()
        assert x_even is not None and len(x_even) > 0
        assert y_even is not None and len(y_even) > 0

        # Verify beat-triggered bipolar oscilloscope wave + envelope curves and P1/P2/P3 markers
        x_osc_w, y_osc_w = win.osc_wave_curve.getData()
        x_osc_e, y_osc_e = win.osc_envelope_curve.getData()
        assert x_osc_w is not None and x_osc_e is not None
        np.testing.assert_allclose(x_osc_w, x_osc_e)
        assert float(np.max(np.abs(y_osc_w))) == pytest.approx(1.0, abs=1e-6)
        assert float(np.min(y_osc_w)) < -0.2
        assert float(np.max(y_osc_e)) > 0.2

        p1_ms = float(win.line_p1.value())
        p2_ms = float(win.line_p2.value())
        p3_ms = float(win.line_p3.value())
        assert 0.2 <= p1_ms <= p2_ms <= p3_ms

        # Verify Real-Time Audio Waveform curve is populated and normalized to [-1.0, 1.0]
        x_wave, y_wave = win.audio_waveform_curve.getData()
        assert x_wave is not None and len(x_wave) > 0
        assert y_wave is not None and len(y_wave) > 0
        assert float(np.max(np.abs(y_wave))) == pytest.approx(1.0, abs=1e-6)

        assert "ppm" in win.lbl_drift_status.text()
        assert "Hz" in win.lbl_drift_status.text()
    finally:
        win.close()
        assert not win.dsp_thread.isRunning()

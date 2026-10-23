"""Unit tests for hardware device enumeration, PortAudio capture, and DSP shock blanking + AGC."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PyQt6.QtWidgets import QApplication

from wea.core.dsp_engine import DSPEngine
from wea.core.models import BeatTelemetry, WatchConfig
from wea.core.synthetic import SyntheticEscapeGenerator
from wea.ui.dsp_worker import DSPStreamWorker
from wea.ui.main_window import MainWin
from wea.utils.audio_io import (
    SYNTHETIC_DEVICE_NAME,
    AudioDeviceInfo,
    list_input_devices,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_list_input_devices_with_mocked_sounddevice_and_headless_fallback() -> None:
    """Verify list_input_devices() returns Synthetic Generator first plus mocked USB devices."""
    mocked_portaudio_devices = [
        {
            "name": "Built-in HDMI Output",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        },
        {
            "name": "USB PnP Sound Device",
            "max_input_channels": 1,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
        },
        {
            "name": "USB PnP Sound Device",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
        {
            "name": "Piezo Contact Timegrapher Mic",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 96000.0,
        },
    ]

    with patch("sounddevice.query_devices", return_value=mocked_portaudio_devices):
        devices = list_input_devices()

    assert len(devices) == 3
    assert isinstance(devices[0], AudioDeviceInfo)
    assert devices[0].device_id is None
    assert devices[0].name == SYNTHETIC_DEVICE_NAME
    assert devices[0].is_synthetic is True
    assert devices[0].default_samplerate == 48000.0

    assert devices[1].device_id == 1
    assert devices[1].name == "USB PnP Sound Device"
    assert devices[1].max_input_channels == 1
    assert devices[1].default_samplerate == 48000.0
    assert devices[1].is_synthetic is False

    assert devices[2].device_id == 3
    assert devices[2].name == "Piezo Contact Timegrapher Mic"
    assert devices[2].max_input_channels == 2
    assert devices[2].default_samplerate == 96000.0
    assert devices[2].is_synthetic is False

    # Verify headless CI fallback when PortAudio raises OSError
    with patch("sounddevice.query_devices", side_effect=OSError("PortAudio library not found")):
        fallback_devices = list_input_devices()

    assert len(fallback_devices) == 1
    assert fallback_devices[0].device_id is None
    assert fallback_devices[0].name == SYNTHETIC_DEVICE_NAME
    assert fallback_devices[0].is_synthetic is True


def test_hardware_audio_capture_portaudio_callbacks_into_dsp_worker() -> None:
    """Verify HardwareAudioCapture PortAudio callbacks feed float64 chunks into DSPStreamWorker."""
    app = QApplication.instance() or QApplication([])
    assert app is not None

    config = WatchConfig(bph=21600, lift_angle=52.0, sample_rate=48000)
    gen = SyntheticEscapeGenerator(config=config, seed=123)
    synth_audio = gen.generate(
        duration_s=4.5,
        true_rate_spd=3.5,
        true_beat_error_ms=0.30,
        true_amplitude_deg=275.0,
        snr_db=38.0,
    )

    captured_stream_kwargs: dict[str, object] = {}

    class FakeInputStream:
        def __init__(self, **kwargs: object) -> None:
            captured_stream_kwargs.update(kwargs)
            self.active = False

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.active = False

    worker = DSPStreamWorker(
        config=config,
        device_id=1,
        buffer_duration_s=6.0,
        window_duration_s=4.0,
        chunk_size=4096,
        synthetic_mode=False,
        workers=2,
    )

    received_telemetry: list[BeatTelemetry] = []
    received_errors: list[str] = []
    worker.telemetry_ready.connect(lambda t: received_telemetry.append(t))
    worker.error_occurred.connect(lambda msg: received_errors.append(msg))

    try:
        with patch("sounddevice.InputStream", side_effect=FakeInputStream):
            worker.start_stream()
            assert worker.isRunning() is True
            assert worker._hardware_capture is not None
            assert worker._hardware_capture.is_active is True
            assert captured_stream_kwargs["device"] == 1
            assert captured_stream_kwargs["channels"] == 1
            assert captured_stream_kwargs["samplerate"] == 48000.0
            assert captured_stream_kwargs["blocksize"] == 4096
            assert captured_stream_kwargs["dtype"] == "float32"

            pa_callback = captured_stream_kwargs["callback"]
            assert callable(pa_callback)

            # Simulate PortAudio C-callbacks delivering 2D float32 mono column buffers
            chunk_size = 4096
            for idx, start in enumerate(range(0, len(synth_audio), chunk_size)):
                mono_f32 = synth_audio[start : start + chunk_size].astype(np.float32).reshape(-1, 1)
                status = SimpleNamespace(input_overflow=(idx == 2))
                pa_callback(mono_f32, len(mono_f32), {}, status)

            app.processEvents()

            assert not received_errors
            assert worker._hardware_capture.overflow_count == 1
            # The overflow on chunk #2 means samples were dropped before it, so the audio
            # buffered up to then is discarded instead of being glued across the gap.
            post_gap_samples = len(synth_audio) - 2 * chunk_size
            assert worker.discontinuity_count == 1
            assert worker.ring_buffer.total_samples_written == post_gap_samples
            assert worker.ring_buffer.available_samples == post_gap_samples
            assert len(received_telemetry) >= 1
            latest = received_telemetry[-1]
            assert abs(latest.rate_spd - 3.5) <= 0.25
            assert abs(latest.beat_error_ms - 0.30) <= 0.05

            worker.stop_stream()
            assert worker.isRunning() is False
            assert worker._hardware_capture is None

        # Verify graceful error handling when opening an unsupported sample rate fails
        with patch(
            "sounddevice.InputStream",
            side_effect=OSError("Invalid sample rate 96000 Hz for USB PnP Sound Device"),
        ):
            worker.set_input_device(1)
            worker.start_stream()
            assert worker.isRunning() is False
            assert len(received_errors) == 1
            assert "96000" in received_errors[0] or "Failed to open" in received_errors[0]
    finally:
        worker.shutdown()


def test_main_window_populates_and_switches_input_devices() -> None:
    """Verify MainWin populates cmb_device via list_input_devices() and forwards selection."""
    app = QApplication.instance() or QApplication([])
    assert app is not None

    mocked_devices = [
        {
            "name": "USB PnP Sound Device",
            "max_input_channels": 1,
            "max_output_channels": 0,
            "default_samplerate": 48000.0,
        },
    ]

    with patch("sounddevice.query_devices", return_value=mocked_devices):
        win = MainWin()
        try:
            assert win.cmb_device.count() == 2
            assert win.cmb_device.itemText(0) == SYNTHETIC_DEVICE_NAME
            assert win.cmb_device.itemData(0) is None
            assert "USB PnP Sound Device" in win.cmb_device.itemText(1)
            assert win.cmb_device.itemData(1) == 0

            win.cmb_device.setCurrentIndex(1)
            assert win.selected_device_id() == 0
            assert win.dsp_worker.device_id == 0
            assert win.dsp_worker.synthetic_mode is False

            win.cmb_device.setCurrentIndex(0)
            assert win.selected_device_id() is None
            assert win.dsp_worker.device_id is None
            assert win.dsp_worker.synthetic_mode is True
        finally:
            win.close()


def test_dsp_engine_shock_blanking_and_agc_on_50ms_mechanical_bump() -> None:
    """Verify DSPEngine blanks a huge 50 ms mechanical shock and preserves Rate and Beat Error."""
    fs = 48000
    config = WatchConfig(bph=21600, lift_angle=52.0, sample_rate=fs)
    true_rate_spd = 4.5
    true_be_ms = 0.35
    true_amp_deg = 280.0

    gen = SyntheticEscapeGenerator(config=config, seed=2026)
    clean_signal = gen.generate(
        duration_s=5.0,
        true_rate_spd=true_rate_spd,
        true_beat_error_ms=true_be_ms,
        true_amplitude_deg=true_amp_deg,
        snr_db=36.0,
    )

    engine = DSPEngine(config=config, workers=2)
    try:
        clean_telemetry = engine.process_window(clean_signal)
        assert abs(clean_telemetry.rate_spd - true_rate_spd) <= 0.15
        assert abs(clean_telemetry.beat_error_ms - true_be_ms) <= 0.03

        # Scale signal down (simulating a quiet titanium case) and inject a huge 50 ms
        # mechanical bump transient right in the middle (t = 2.48s to 2.53s, overlapping a beat).
        quiet_corrupted = clean_signal * 0.025
        shock_start = int(2.480 * fs)
        shock_len = int(0.050 * fs)  # 50 ms = 2400 samples
        shock_end = shock_start + shock_len

        t_shock = np.arange(shock_len, dtype=np.float64) / float(fs)
        rng = np.random.default_rng(777)
        mechanical_bump = (
            1.5 * np.sin(2.0 * np.pi * 4200.0 * t_shock)
            + 1.2 * np.sin(2.0 * np.pi * 7800.0 * t_shock)
            + 0.8 * rng.normal(0.0, 1.0, size=shock_len)
        )
        quiet_corrupted[shock_start:shock_end] += mechanical_bump

        # Verify blank_transients directly detects the 50 ms shock
        filtered = engine.filter_signal(quiet_corrupted, fs=float(fs))
        conditioned, blank_mask, blank_fraction = engine.blank_transients(
            filtered,
            fs=float(fs),
            raw_signal=quiet_corrupted,
        )
        assert blank_fraction > 0.008
        assert np.all(conditioned[shock_start:shock_end] == 0.0)
        assert np.all(blank_mask[shock_start:shock_end])

        # Verify full DSPEngine pipeline with Shock Blanking + AGC recovers Rate and Beat Error
        shock_telemetry = engine.process_window(quiet_corrupted)
        assert abs(shock_telemetry.rate_spd - true_rate_spd) <= 0.25
        assert abs(shock_telemetry.beat_error_ms - true_be_ms) <= 0.05
        assert 0.50 <= shock_telemetry.confidence < clean_telemetry.confidence

        # Verify AGC noise gate does not amplify pure silence or stationary low-level noise
        silence_noise = rng.normal(0.0, 1e-6, size=fs * 2)
        gated = engine.apply_agc(silence_noise, fs=float(fs))
        np.testing.assert_allclose(gated, silence_noise)
    finally:
        engine.shutdown()


def test_piezo_contact_mic_pipewire_128_quantum_no_false_blanking_or_glitch_spam(
    caplog: object,
) -> None:
    """Verify real-world CM108 piezo impulses at PipeWire quantum=128 are never falsely blanked."""
    app = QApplication.instance() or QApplication([])
    assert app is not None

    fs = 48000
    config = WatchConfig(bph=21600, lift_angle=52.0, sample_rate=fs)
    true_rate_spd = -2.4
    true_be_ms = 0.25
    true_amp_deg = 278.0

    gen = SyntheticEscapeGenerator(config=config, seed=404)
    piezo_signal = gen.generate(
        duration_s=6.0,
        true_rate_spd=true_rate_spd,
        true_beat_error_ms=true_be_ms,
        true_amplitude_deg=true_amp_deg,
        snr_db=34.0,
    )
    # Add low-frequency desk rumble (60 Hz + 180 Hz) that the 3000-14000 Hz bandpass must reject
    t_vec = np.arange(len(piezo_signal), dtype=np.float64) / float(fs)
    piezo_signal = (
        piezo_signal
        + 0.25 * np.sin(2.0 * np.pi * 60.0 * t_vec)
        + 0.15 * np.sin(2.0 * np.pi * 180.0 * t_vec)
    )

    engine = DSPEngine(config=config, workers=2)
    try:
        filtered = engine.filter_signal(piezo_signal, fs=float(fs))
        _, _, blank_fraction = engine.blank_transients(
            filtered,
            fs=float(fs),
            raw_signal=piezo_signal,
        )
        # Sharp ~2.5 ms watch ticks (~1.55% duty cycle) MUST NEVER be mistaken for macro-shocks!
        assert blank_fraction == 0.0
    finally:
        engine.shutdown()

    captured_stream_kwargs: dict[str, object] = {}

    class FakeInputStream:
        def __init__(self, **kwargs: object) -> None:
            captured_stream_kwargs.update(kwargs)
            self.active = False

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.active = False

    worker = DSPStreamWorker(
        config=config,
        device_id=1,
        buffer_duration_s=8.0,
        window_duration_s=3.0,
        chunk_size=128,
        synthetic_mode=False,
        workers=2,
    )
    received_telemetry: list[BeatTelemetry] = []
    worker.telemetry_ready.connect(lambda t: received_telemetry.append(t))

    try:
        with patch("sounddevice.InputStream", side_effect=FakeInputStream):
            worker.start_stream()
            pa_callback = captured_stream_kwargs["callback"]
            assert callable(pa_callback)

            # Stream 6.0s of audio in 128-frame PipeWire quantum chunks (2.67 ms per callback)
            # with realistic +-80 us USB scheduling jitter on timestamp_ns.
            quantum = 128
            base_ns = 5_000_000_000_000
            nominal_dt_ns = int(round((quantum / float(fs)) * 1e9))
            rng = np.random.default_rng(128)

            with patch.object(
                worker.drift_estimator,
                "update",
                wraps=worker.drift_estimator.update,
            ):
                for idx, start in enumerate(range(0, len(piezo_signal), quantum)):
                    chunk_f32 = (
                        piezo_signal[start : start + quantum].astype(np.float32).reshape(-1, 1)
                    )
                    jitter_ns = int(rng.integers(-80_000, 80_000))
                    ts_ns = base_ns + (idx + 1) * nominal_dt_ns + jitter_ns
                    with patch("time.perf_counter_ns", return_value=ts_ns):
                        pa_callback(chunk_f32, len(chunk_f32), {}, None)

            app.processEvents()

            assert len(received_telemetry) >= 1
            latest = received_telemetry[-1]
            assert abs(latest.rate_spd - true_rate_spd) <= 0.25
            assert abs(latest.beat_error_ms - true_be_ms) <= 0.05
            assert abs(latest.amplitude_deg - true_amp_deg) <= 5.0
            assert latest.confidence >= 0.85
            assert abs(worker.drift_estimator.drift_ppm) < 5.0

            if hasattr(caplog, "text"):
                assert "Rejected ADC clock glitch" not in caplog.text
    finally:
        worker.shutdown()

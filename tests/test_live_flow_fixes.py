"""Regression tests for the live capture -> DSP flow (worker, clock drift, capture, logging)."""

from __future__ import annotations

import logging
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from wea.core.clock_drift import SampleRateDriftEstimator
from wea.core.models import BeatTelemetry, WatchConfig
from wea.ui.dsp_worker import DSPStreamWorker
from wea.utils.audio_io import HardwareAudioCapture
from wea.utils.logger_setup import LoggerSetup

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FS = 48000


class FakeInputStream:
    """Stand-in for ``sounddevice.InputStream`` that records its kwargs."""

    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        FakeInputStream.last_kwargs = dict(kwargs)
        self.active = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def close(self) -> None:
        self.active = False


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _feed(
    callback, signal, chunk=1024, jitter_ns=0.0, gap_at=None, gap_chunks=3, seed=0, on_gap=None
):
    """Feed ``signal`` to a PortAudio-style callback; optionally drop audio and flag overflow."""
    rng = np.random.default_rng(seed)
    t_ns = 10**12
    pos = idx = 0
    while pos + chunk <= len(signal):
        status = None
        if gap_at is not None and idx == gap_at:
            pos += gap_chunks * chunk  # samples lost by the driver
            status = SimpleNamespace(input_overflow=True)
            if on_gap is not None:
                on_gap()
            if pos + chunk > len(signal):
                break
        t_ns += int(chunk / FS * 1e9) + int(rng.normal(0.0, jitter_ns))
        with patch("time.perf_counter_ns", return_value=t_ns):
            callback(signal[pos : pos + chunk].astype(np.float32).reshape(-1, 1), chunk, {}, status)
        pos += chunk
        idx += 1


# ----------------------------------------------------------------------------- worker


def test_live_hardware_flow_recovers_physical_watch_with_wrong_bph(physical_signal) -> None:
    """End-to-end regression of the reported bug (+7000 s/d, 8 ms, 180 deg on a +4/0.2/279 watch).

    Wrong configured BPH, 4 ms callback jitter and a dropped block flagged as overflow.
    """
    app = _app()
    signal = physical_signal(bph=28800, seconds=10.0, snr_db=20.0, seed=5)
    worker = DSPStreamWorker(
        config=WatchConfig(bph=21600, lift_angle=52.0, sample_rate=FS),
        device_id=1,
        synthetic_mode=False,
    )
    telemetry: list[BeatTelemetry] = []
    worker.telemetry_ready.connect(telemetry.append)
    try:
        with patch("sounddevice.InputStream", side_effect=FakeInputStream):
            worker.start_stream()
            _feed(FakeInputStream.last_kwargs["callback"], signal, jitter_ns=4e6, gap_at=150)
            app.processEvents()
        assert worker.discontinuity_count == 1
        assert len(telemetry) >= 5
        for t in telemetry[-5:]:
            assert t.bph == 28800 and t.bph_mismatch is True
            assert t.rate_spd == pytest.approx(4.0, abs=2.5)  # 3 s windows at SNR 20 dB
            assert t.beat_error_ms == pytest.approx(0.2, abs=0.06)
            assert t.amplitude_deg == pytest.approx(279.0, abs=5.0)
    finally:
        worker.shutdown()


def test_overflow_gap_is_not_glued_into_analysis_window(physical_signal) -> None:
    """Every window analyzed after a 1.5-beat gap must be free of it (no glued audio)."""
    app = _app()
    signal = physical_signal(bph=28800, seconds=8.0, seed=2)
    worker = DSPStreamWorker(
        config=WatchConfig(bph=28800, lift_angle=52.0, sample_rate=FS),
        device_id=1,
        synthetic_mode=False,
    )
    telemetry: list[BeatTelemetry] = []
    worker.telemetry_ready.connect(telemetry.append)
    gap_marks: list[int] = []
    try:
        with patch("sounddevice.InputStream", side_effect=FakeInputStream):
            worker.start_stream()
            # 9 * 1024 samples = 192 ms = 1.54 beats lost at 28800 BPH
            _feed(
                FakeInputStream.last_kwargs["callback"],
                signal,
                gap_at=180,
                gap_chunks=9,
                on_gap=lambda: gap_marks.append(len(telemetry)),
            )
            app.processEvents()
        post_gap = telemetry[gap_marks[0] :]
        assert len(post_gap) >= 5
        for t in post_gap:
            assert t.beat_error_ms == pytest.approx(0.2, abs=0.06)
        # Rate needs the full window: the first ~1 s warm-up windows are inherently noisy
        full_windows = [t for t in post_gap if len(t.filtered_signal) >= worker._window_samples]
        assert full_windows
        for t in full_windows:
            assert t.rate_spd == pytest.approx(4.0, abs=2.5)
    finally:
        worker.shutdown()


def test_hardware_audio_is_analyzed_once_per_hop(physical_signal) -> None:
    """Each hop must trigger one analysis (the scheduled eval and timer used to both run it)."""
    app = _app()
    worker = DSPStreamWorker(
        config=WatchConfig(bph=28800, lift_angle=52.0, sample_rate=FS),
        device_id=1,
        synthetic_mode=False,
    )
    try:
        with (
            patch("sounddevice.InputStream", side_effect=FakeInputStream),
            patch.object(
                worker.dsp_engine, "process_window", wraps=worker.dsp_engine.process_window
            ) as spy,
        ):
            worker.start_stream()
            signal = physical_signal(bph=28800, seconds=6.0)
            _feed(FakeInputStream.last_kwargs["callback"], signal)
            app.processEvents()
            # A telemetry timer tick right after the eval finds no new hop and must not rerun
            worker._on_telemetry_timer_tick()
            hops_after_warmup = (len(signal) - worker._min_analysis_samples) / (0.25 * FS)
            assert 1 <= spy.call_count <= int(hops_after_warmup) + 2
    finally:
        worker.shutdown()


def test_update_watch_config_sample_rate_change_does_not_crash() -> None:
    """Switching 48 -> 96 kHz referenced a non-existent ``lowpass_cutoff_hz`` attribute."""
    worker = DSPStreamWorker(config=WatchConfig(bph=28800, lift_angle=52.0, sample_rate=FS))
    errors: list[str] = []
    worker.error_occurred.connect(errors.append)
    try:
        worker.update_watch_config(WatchConfig(bph=28800, lift_angle=52.0, sample_rate=96000))
        assert not errors
        assert worker.dsp_engine.config.sample_rate == 96000
        expected = worker.dsp_engine._design_bandpass_sos(
            96000.0, worker.dsp_engine.highpass_cutoff_hz, worker.dsp_engine.highcut_hz
        )
        np.testing.assert_allclose(worker.dsp_engine._sos, expected)
    finally:
        worker.shutdown()


def test_background_analysis_emits_from_analysis_thread_and_skips_when_busy(
    physical_signal,
) -> None:
    """With analysis_in_background the worker's own thread is never blocked by the DSP."""
    app = _app()
    worker = DSPStreamWorker(
        config=WatchConfig(bph=28800, lift_angle=52.0, sample_rate=FS),
        synthetic_mode=False,
        analysis_in_background=True,
    )
    release = threading.Event()
    dsp_threads: list[str] = []
    received: list[BeatTelemetry] = []
    worker.telemetry_ready.connect(received.append)
    real_process = worker.dsp_engine.process_window

    def slow_process(*args, **kwargs):
        dsp_threads.append(threading.current_thread().name)
        release.wait(5.0)
        return real_process(*args, **kwargs)

    try:
        worker._running = True
        worker.ring_buffer.write(physical_signal(bph=28800, seconds=4.0))
        worker._samples_since_last_analysis = 4 * FS
        with patch.object(worker.dsp_engine, "process_window", side_effect=slow_process) as spy:
            t_start = time.perf_counter()
            worker._on_telemetry_timer_tick()
            assert time.perf_counter() - t_start < 0.5  # returned without waiting for the DSP
            worker._samples_since_last_analysis = 4 * FS
            worker._on_telemetry_timer_tick()  # busy -> skipped
            release.set()
            worker._analysis_future.result(timeout=5.0)
            assert spy.call_count == 1
        app.processEvents()  # cross-thread signal is queued, exactly as for the real UI
        assert dsp_threads and dsp_threads[0].startswith("WEA-Analysis")
        assert len(received) == 1 and received[0].bph == 28800
    finally:
        release.set()
        worker.shutdown()


def test_background_analysis_drops_results_after_stop(physical_signal) -> None:
    """A window analyzed while the stream stops must not reach the UI afterwards."""
    _app()
    worker = DSPStreamWorker(
        config=WatchConfig(bph=28800, lift_angle=52.0, sample_rate=FS),
        synthetic_mode=False,
        analysis_in_background=True,
    )
    release = threading.Event()
    received: list[BeatTelemetry] = []
    worker.telemetry_ready.connect(received.append)
    real_process = worker.dsp_engine.process_window

    def slow_process(*args, **kwargs):
        release.wait(5.0)
        return real_process(*args, **kwargs)

    try:
        worker._running = True
        worker.ring_buffer.write(physical_signal(bph=28800, seconds=4.0))
        worker._samples_since_last_analysis = 4 * FS
        with patch.object(worker.dsp_engine, "process_window", side_effect=slow_process):
            worker._on_telemetry_timer_tick()
            worker.stop_stream()
            release.set()
            worker._analysis_future.result(timeout=5.0)
        QApplication.processEvents()
        assert received == []
    finally:
        release.set()
        worker.shutdown()


def test_ntp_timer_only_runs_with_dynamic_drift_tracking() -> None:
    """NTP disciplining is pointless (and costs a thread + network) without host-clock tracking."""
    app = _app()
    for dynamic, expected in ((False, False), (True, True)):
        worker = DSPStreamWorker(
            config=WatchConfig(bph=28800, lift_angle=52.0, sample_rate=FS),
            device_id=1,
            synthetic_mode=False,
            adc_dynamic_drift=dynamic,
        )
        try:
            with patch("sounddevice.InputStream", side_effect=FakeInputStream):
                worker.start_stream()
                app.processEvents()
                assert worker._ntp_timer.isActive() is expected
                assert worker.drift_estimator.dynamic_tracking is dynamic
        finally:
            worker.shutdown()


# ----------------------------------------------------------------------- clock drift


def test_drift_estimator_without_dynamic_tracking_ignores_callback_jitter() -> None:
    """Jittery host timestamps must not move effective_fs when dynamic tracking is off."""
    est = SampleRateDriftEstimator(nominal_fs=48000.0, calibration_ppm=10.0, dynamic_tracking=False)
    rng = np.random.default_rng(0)
    base = 10**12
    for k in range(400):  # 34 s of 4096-sample chunks with 8 ms Windows-like jitter
        est.update(4096, timestamp_ns=base + int(k * 4096 / 48000 * 1e9 + rng.normal(0, 8e6)))
    assert est.is_converged is False
    assert est.effective_fs == pytest.approx(48000.0 * (1.0 + 10e-6), rel=1e-12)


def test_cpu_skew_is_not_applied_before_dynamic_estimate_exists() -> None:
    """The NTP CPU skew corrects a host-clock measurement; it says nothing about the ADC."""
    est = SampleRateDriftEstimator(nominal_fs=48000.0, calibration_ppm=5.0, cpu_skew_ppm=40.0)
    assert est.is_converged is False
    assert est.effective_fs == pytest.approx(48000.0 * (1.0 + 5e-6), rel=1e-12)


# ------------------------------------------------------------------------- capture


def test_hardware_capture_defaults_to_high_latency() -> None:
    """Large host buffers prevent overflows; latency is irrelevant for a timegrapher."""
    with patch("sounddevice.InputStream", side_effect=FakeInputStream):
        cap = HardwareAudioCapture(device_id=1, callback=lambda *_: None)
        cap.start()
        assert FakeInputStream.last_kwargs["latency"] == "high"
        cap.stop()


def test_overflow_logging_is_rate_limited(caplog) -> None:
    """The real-time audio callback must not log on every overflow."""
    cap = HardwareAudioCapture(device_id=1, callback=lambda *_: None)
    block = np.zeros((256, 1), dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger="wea.utils.audio_io"):
        for _ in range(120):
            cap._portaudio_callback(block, 256, {}, SimpleNamespace(input_overflow=True))
    hits = [r for r in caplog.records if "input overflow" in r.getMessage()]
    assert cap.overflow_count == 120
    assert len(hits) == 1 + 120 // HardwareAudioCapture.OVERFLOW_LOG_EVERY


# ------------------------------------------------------------------------- logging


@pytest.mark.parametrize(
    ("env_value", "default", "expected"),
    [(None, "INFO", logging.INFO), ("debug", "INFO", logging.DEBUG), ("bogus", "WARNING", 30)],
)
def test_log_level_resolution(monkeypatch, env_value, default, expected) -> None:
    """Hot-path DEBUG logging is opt-in through WEA_LOG_LEVEL."""
    if env_value is None:
        monkeypatch.delenv("WEA_LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("WEA_LOG_LEVEL", env_value)
    assert LoggerSetup._resolve_level("WEA_LOG_LEVEL", default) == expected

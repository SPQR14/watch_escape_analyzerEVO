"""Deterministic unit tests for SampleRateDriftEstimator, multi-threaded DSPEngine, and UI curve optimizations."""

from __future__ import annotations

import os

import numpy as np
import pytest

from wea.core.clock_drift import SampleRateDriftEstimator
from wea.core.dsp_engine import DSPEngine
from wea.core.models import BeatTelemetry, WatchConfig
from wea.core.synthetic import SyntheticEscapeGenerator

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_sample_rate_drift_estimator_dynamic_ols_and_guardrail() -> None:
    """Verify SampleRateDriftEstimator recovers +50 ppm drift within < 0.5 ppm and rejects >500 ppm glitches."""
    nominal_fs = 48000.0
    true_drift_ppm = 50.0
    true_fs = nominal_fs * (1.0 + true_drift_ppm * 1e-6)
    expected_rate_bias_spd = 86400.0 * (true_drift_ppm * 1e-6)  # +4.32 s/d

    estimator = SampleRateDriftEstimator(nominal_fs=nominal_fs, window_sec=30.0)
    rng = np.random.default_rng(2026)

    chunk_duration_sec = 0.5
    chunk_samples = true_fs * chunk_duration_sec
    base_ns = 1_000_000_000_000

    # Before 5.0 seconds and 10 chunks, dynamic OLS has not activated yet
    for step in range(5):
        t_ns = base_ns + int(round(step * chunk_duration_sec * 1e9))
        estimator.update(chunk_samples=chunk_samples, timestamp_ns=t_ns)
    assert estimator.drift_ppm == pytest.approx(0.0, abs=1e-9)

    # Feed 25 chunks (12.5 seconds total >= 5.0s and >= 10 chunks) with slight nanosecond jitter
    for step in range(5, 25):
        jitter_ns = int(rng.normal(0.0, 200.0))
        t_ns = base_ns + int(round(step * chunk_duration_sec * 1e9)) + jitter_ns
        estimator.update(chunk_samples=chunk_samples, timestamp_ns=t_ns)

    recovered_ppm_err = abs(estimator.drift_ppm - true_drift_ppm)
    fs_ppm_err = abs((estimator.effective_fs - true_fs) / nominal_fs) * 1e6

    assert recovered_ppm_err < 0.5, f"Expected drift ~+50.0 ppm, got {estimator.drift_ppm:+.4f} ppm"
    assert fs_ppm_err < 0.5, (
        f"Expected effective_fs ~{true_fs:.4f} Hz, got {estimator.effective_fs:.4f} Hz"
    )
    assert abs(estimator.rate_bias_spd - expected_rate_bias_spd) < 0.04, (
        f"Expected rate_bias_spd ~+4.32 s/d, got {estimator.rate_bias_spd:+.4f} s/d"
    )

    # Inject a massive +2500 ppm glitch chunk (> ±500 ppm guardrail) and verify it is rejected
    glitch_t_ns = base_ns + int(round(25 * chunk_duration_sec * 1e9))
    glitch_samples = nominal_fs * (1.0 + 2500.0 * 1e-6) * chunk_duration_sec
    estimator.update(chunk_samples=glitch_samples, timestamp_ns=glitch_t_ns)

    assert abs(estimator.drift_ppm - true_drift_ppm) < 0.5


def test_sample_rate_drift_estimator_quartz_calibration() -> None:
    """Verify quartz reference calibration recovers +50 ppm offset and +4.32 s/d rate bias."""
    nominal_fs = 96000.0
    true_drift_ppm = 50.0
    true_fs = nominal_fs * (1.0 + true_drift_ppm * 1e-6)
    expected_rate_bias_spd = 86400.0 * (true_drift_ppm * 1e-6)  # +4.32 s/d

    rng = np.random.default_rng(101)
    # Simulate 30 ticks of a 1 Hz Lavet stepping motor (3,600 BPH, target_tick_sec = 1.0s)
    measured_ticks = true_fs * 1.0 + rng.normal(0.0, 0.005, size=30)

    estimator = SampleRateDriftEstimator(nominal_fs=nominal_fs)
    cal_ppm = estimator.calibrate_from_quartz_intervals(
        measured_samples_per_tick=measured_ticks,
        nominal_fs=nominal_fs,
        target_tick_sec=1.0,
    )

    assert abs(cal_ppm - true_drift_ppm) < 0.5
    assert abs(((estimator.effective_fs - true_fs) / nominal_fs) * 1e6) < 0.5
    assert abs(estimator.rate_bias_spd - expected_rate_bias_spd) < 0.04


@pytest.mark.parametrize(
    ("preset_name", "bph", "lift_angle", "true_rate_spd", "true_be_ms", "true_amp_deg"),
    [
        ("Seiko NH35", 21600, 53.0, +5.0, 0.4, 275.0),
        ("Sellita SW200 / Miyota 9015", 28800, 51.0, +3.5, 0.1, 290.0),
    ],
)
def test_dsp_engine_multithreaded_horological_presets(
    preset_name: str,
    bph: int,
    lift_angle: float,
    true_rate_spd: float,
    true_be_ms: float,
    true_amp_deg: float,
) -> None:
    """Verify multi-threaded DSPEngine (workers=2) recovers Rate (±0.15 s/d), BE (±0.05 ms), and Amplitude (±3.0°)."""
    sample_rate = 96000
    config = WatchConfig(bph=bph, lift_angle=lift_angle, sample_rate=sample_rate)
    generator = SyntheticEscapeGenerator(config=config, seed=42)

    signal = generator.generate(
        duration_s=6.0,
        true_rate_spd=true_rate_spd,
        true_beat_error_ms=true_be_ms,
        true_amplitude_deg=true_amp_deg,
        snr_db=40.0,
    )

    with DSPEngine(config=config, workers=2) as engine:
        telemetry = engine.process_window(signal)

    assert isinstance(telemetry, BeatTelemetry)
    assert len(telemetry.beat_times_sec) >= 30
    assert len(telemetry.envelope) == len(signal)
    assert telemetry.confidence > 0.85

    rate_err = abs(telemetry.rate_spd - true_rate_spd)
    be_err = abs(telemetry.beat_error_ms - true_be_ms)
    amp_err = abs(telemetry.amplitude_deg - true_amp_deg)

    assert rate_err <= 0.15, (
        f"[{preset_name}] Expected rate {true_rate_spd:+.2f} s/d (±0.15), "
        f"got {telemetry.rate_spd:+.4f} s/d (err={rate_err:.4f})"
    )
    assert be_err <= 0.05, (
        f"[{preset_name}] Expected beat error {true_be_ms:.2f} ms (±0.05), "
        f"got {telemetry.beat_error_ms:.4f} ms (err={be_err:.4f})"
    )
    assert amp_err <= 3.0, (
        f"[{preset_name}] Expected amplitude {true_amp_deg:.1f}° (±3.0°), "
        f"got {telemetry.amplitude_deg:.2f}° (err={amp_err:.2f}°)"
    )


def test_dsp_engine_streaming_filter_and_clock_drift_integration() -> None:
    """Verify stateful streaming SOS filter and drift estimator integration in DSPEngine."""
    config = WatchConfig(bph=21600, lift_angle=53.0, sample_rate=48000)
    estimator = SampleRateDriftEstimator(nominal_fs=48000.0, calibration_ppm=0.0)
    engine = DSPEngine(config=config, workers=2, drift_estimator=estimator)
    try:
        gen = SyntheticEscapeGenerator(config=config, seed=77)
        sig = gen.generate(
            duration_s=5.0,
            true_rate_spd=5.0,
            true_beat_error_ms=0.4,
            true_amplitude_deg=275.0,
            snr_db=40.0,
        )
        # Process two consecutive chunks in streaming mode to verify stateful zi tracking
        half = len(sig) // 2
        filt_part1 = engine.filter_signal(sig[:half], streaming=True)
        filt_part2 = engine.filter_signal(sig[half:], streaming=True)
        assert len(filt_part1) == half
        assert len(filt_part2) == len(sig) - half

        engine.reset_filter_state()
        telemetry = engine.process_window(sig, streaming=False)
        assert abs(telemetry.rate_spd - 5.0) <= 0.15
        assert abs(telemetry.beat_error_ms - 0.4) <= 0.05
        assert abs(telemetry.amplitude_deg - 275.0) <= 3.0
    finally:
        engine.shutdown()


def test_ui_curve_optimizations_and_offscreen_gpu_guard() -> None:
    """Verify PyQtGraph curves enable clipToView, autoDownsample, and peak downsampling."""
    from PyQt6.QtWidgets import QApplication

    from wea.ui.main_window import MainWin, configure_pyqtgraph_gpu

    app = QApplication.instance() or QApplication([])
    assert app is not None
    assert configure_pyqtgraph_gpu() is False

    win = MainWin()
    try:
        for curve in (win.osc_wave_curve, win.osc_envelope_curve, win.audio_waveform_curve):
            assert curve.opts.get("clipToView") is True
            assert curve.opts.get("autoDownsample") is True
            assert curve.opts.get("downsampleMethod") == "peak"
    finally:
        win.close()

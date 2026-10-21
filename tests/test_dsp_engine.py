"""Deterministic unit tests for SampleRateDriftEstimator, multi-threaded DSPEngine, and UI curve optimizations."""

from __future__ import annotations

import os
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wea.core.clock_drift import AtomicNTPDiscipline, SampleRateDriftEstimator
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


def test_sample_rate_drift_estimator_decimated_ols_and_ema_three_layer_fusion() -> None:
    """Verify decimated OLS (recalc_interval_sec=0.5), EMA smoothing, and 3-layer effective_fs."""
    nominal_fs = 48000.0
    true_adc_drift_ppm = 50.0
    cal_ppm = 12.0
    cpu_skew_ppm = 4.0
    true_dynamic_fs = nominal_fs * (1.0 + true_adc_drift_ppm * 1e-6)

    estimator = SampleRateDriftEstimator(
        nominal_fs=nominal_fs,
        window_sec=30.0,
        min_duration_sec=5.0,
        min_chunks=10,
        recalc_interval_sec=0.5,
        ema_alpha=0.15,
        calibration_ppm=cal_ppm,
        cpu_skew_ppm=cpu_skew_ppm,
    )

    rng = np.random.default_rng(2026)
    chunk_dt_sec = 0.05  # 20 chunks/sec (high rate)
    chunk_samples = true_dynamic_fs * chunk_dt_sec
    base_ns = 2_000_000_000_000

    # Feed 300 chunks (15.0 seconds total; 10.0 seconds after min_duration_sec=5.0s)
    for step in range(301):
        jitter_ns = int(rng.normal(0.0, 500.0)) if step > 0 else 0
        t_ns = base_ns + int(round(step * chunk_dt_sec * 1e9)) + jitter_ns
        estimator.update(chunk_samples=chunk_samples, timestamp_ns=t_ns)

    assert estimator.is_converged is True
    # Over 10 seconds of active OLS with recalc_interval_sec=0.5, OLS runs ~21 times (not 201 times)
    assert 18 <= estimator._ols_eval_count <= 23

    expected_eff_fs = true_dynamic_fs * (1.0 + (cal_ppm - cpu_skew_ppm) * 1e-6)
    eff_fs_err_ppm = abs((estimator.effective_fs - expected_eff_fs) / nominal_fs) * 1e6
    assert eff_fs_err_ppm < 0.5


def _pack_ntp_response(t2_unix: float, t3_unix: float) -> bytes:
    """Build a synthetic 48-byte NTPv3 server response with T2 and T3 timestamps."""
    epoch_offset = AtomicNTPDiscipline.NTP_EPOCH_OFFSET
    two_pow_32 = AtomicNTPDiscipline.TWO_POW_32

    t2_ntp = t2_unix + epoch_offset
    t2_sec = int(t2_ntp)
    t2_frac = int(round((t2_ntp - t2_sec) * two_pow_32)) & 0xFFFFFFFF

    t3_ntp = t3_unix + epoch_offset
    t3_sec = int(t3_ntp)
    t3_frac = int(round((t3_ntp - t3_sec) * two_pow_32)) & 0xFFFFFFFF

    header_32 = b"\x1c" + 31 * b"\x00"
    timestamps_16 = struct.pack("!4I", t2_sec, t2_frac, t3_sec, t3_frac)
    return header_32 + timestamps_16


def test_atomic_ntp_discipline_minimum_delay_filter_and_cpu_skew_offline() -> None:
    """Verify AtomicNTPDiscipline minimum-delay hull filter and cpu_skew_ppm entirely offline."""
    discipline = AtomicNTPDiscipline(host="time.cloudflare.com", port=123, timeout_sec=0.4)

    # Burst 1 (anchor at t_atomic = 1_700_000_000.0, t_local = 100.0):
    # Packet 0: RTT = 50 ms, Packet 1: RTT = 8 ms (minimum delay!), Packet 2: RTT = 35 ms, Packet 3: RTT = 60 ms
    t_atomic_0 = 1_700_000_000.0
    burst1_packets = [
        (_pack_ntp_response(t_atomic_0 + 0.020, t_atomic_0 + 0.021), ("162.159.200.1", 123)),
        (_pack_ntp_response(t_atomic_0 + 0.104, t_atomic_0 + 0.104), ("162.159.200.1", 123)),
        (_pack_ntp_response(t_atomic_0 + 0.215, t_atomic_0 + 0.216), ("162.159.200.1", 123)),
        (_pack_ntp_response(t_atomic_0 + 0.325, t_atomic_0 + 0.326), ("162.159.200.1", 123)),
    ]
    # Corresponding (T1, T4) monotonic timestamps in seconds converted to ns:
    # Packet 0: T1=100.000, T4=100.051 -> delta = 0.050s
    # Packet 1: T1=100.100, T4=100.108 -> delta = 0.008s (minimum!) -> local_mid=100.104, atomic_mid=t_atomic_0+0.104
    # Packet 2: T1=100.200, T4=100.236 -> delta = 0.035s
    # Packet 3: T1=100.300, T4=100.361 -> delta = 0.060s
    burst1_t_ns = [
        int(100.000 * 1e9),
        int(100.051 * 1e9),
        int(100.100 * 1e9),
        int(100.108 * 1e9),
        int(100.200 * 1e9),
        int(100.236 * 1e9),
        int(100.300 * 1e9),
        int(100.361 * 1e9),
    ]

    mock_sock = MagicMock()
    mock_sock.__enter__.return_value = mock_sock
    mock_sock.recvfrom.side_effect = burst1_packets

    with (
        patch("socket.socket", return_value=mock_sock),
        patch("time.perf_counter_ns", side_effect=burst1_t_ns),
    ):
        skew0 = discipline.update_cpu_skew(burst_size=4)

    assert skew0 == 0.0
    assert discipline._anchor is not None
    assert discipline._anchor[0] == pytest.approx(100.104, abs=1e-6)
    assert discipline._anchor[1] == pytest.approx(t_atomic_0 + 0.104, abs=1e-6)

    # Burst 2: 20.0 atomic seconds later, with host CPU clock running +25.0 ppm fast
    true_skew_ppm = 25.0
    dt_atomic = 20.0
    dt_local = dt_atomic * (1.0 + true_skew_ppm * 1e-6)
    t_local_mid_2 = 100.104 + dt_local
    t_atomic_mid_2 = (t_atomic_0 + 0.104) + dt_atomic

    burst2_packets = [
        (_pack_ntp_response(t_atomic_mid_2, t_atomic_mid_2), ("162.159.200.1", 123)),
    ]
    burst2_t_ns = [
        int(round((t_local_mid_2 - 0.004) * 1e9)),
        int(round((t_local_mid_2 + 0.004) * 1e9)),
    ]
    mock_sock.recvfrom.side_effect = burst2_packets

    with (
        patch("socket.socket", return_value=mock_sock),
        patch("time.perf_counter_ns", side_effect=burst2_t_ns),
    ):
        skew1 = discipline.update_cpu_skew(burst_size=1)

    # First EMA step (0.3 * 25.0 + 0.7 * 0.0 = 7.5 ppm)
    assert skew1 == pytest.approx(0.3 * true_skew_ppm, abs=0.05)

    # Offline / OSError resilience: returns None from sample_burst and preserves cpu_skew_ppm
    mock_sock.recvfrom.side_effect = OSError("Network unreachable")
    with patch("socket.socket", return_value=mock_sock):
        assert discipline.sample_burst(burst_size=2) is None
        assert discipline.update_cpu_skew(burst_size=2) == pytest.approx(skew1, abs=1e-9)


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

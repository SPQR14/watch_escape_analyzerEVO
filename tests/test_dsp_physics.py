"""Physics regression tests: amplitude, beat error convention, BPH detection and DSP refactors.

Tolerances come from the uncertainty budget of a short analysis window, measured on the
independent physical generator in ``conftest.py`` (6 s, SNR 25 dB, 6 seeds, 18000-36000 BPH,
48/96 kHz): worst rate error 1.15 s/d, beat error 0.024 ms, amplitude 2.3 deg.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wea.core.dsp_engine import DSPEngine
from wea.core.dsp_pipeline import EscapeDSPPipeline
from wea.core.models import WatchConfig
from wea.core.synthetic import SyntheticEscapeGenerator

RATE_TOL_SPD = 1.5
BEAT_ERROR_TOL_MS = 0.05
AMPLITUDE_TOL_DEG = 5.0


# --------------------------------------------------------------------------- amplitude


def test_amplitude_formula_matches_reference_example() -> None:
    """28800 BPH, LA 52 deg, t_lift 7.40 ms must give ~280 deg (A = LA / (2 sin(pi f t)))."""
    amp = DSPEngine.compute_amplitude_from_lift_time(t_lift=7.40e-3, lift_angle=52.0, bph=28800)
    assert amp == pytest.approx(280.0, abs=0.5)


@pytest.mark.parametrize("bph", [18000, 21600, 28800, 36000])
@pytest.mark.parametrize("amplitude_deg", [160.0, 220.0, 279.0, 320.0])
def test_lift_time_and_amplitude_are_inverse(bph: int, amplitude_deg: float) -> None:
    """lift_time_from_amplitude and compute_amplitude_from_lift_time round-trip exactly."""
    t_lift = DSPEngine.lift_time_from_amplitude(amplitude_deg, 52.0, bph)
    expected = math.asin(52.0 / (2.0 * amplitude_deg)) / (math.pi * bph / 7200.0)
    assert t_lift == pytest.approx(expected, rel=1e-12)
    amp = DSPEngine.compute_amplitude_from_lift_time(t_lift, 52.0, bph)
    assert amp == pytest.approx(amplitude_deg, abs=1e-6)


def test_synthetic_generator_uses_physical_lift_time() -> None:
    """The project generator must place P3 one full physical lift time after P1."""
    gen = SyntheticEscapeGenerator(WatchConfig(bph=28800, lift_angle=52.0), seed=0)
    assert gen.compute_p1_p3_delay(280.0) == pytest.approx(7.40e-3, abs=0.01e-3)
    assert gen.compute_p1_p3_delay(279.0) == pytest.approx(
        DSPEngine.lift_time_from_amplitude(279.0, 52.0, 28800), rel=1e-9
    )


def test_lift_search_window_covers_low_amplitude_at_slow_bph() -> None:
    """At 18000 BPH and 150 deg the lift time (~22 ms) must fit inside the beat window."""
    t_lift = DSPEngine.lift_time_from_amplitude(150.0, 52.0, 18000)
    t_min, t_max = DSPEngine._lift_time_bounds(18000, 52.0)
    assert t_min < t_lift < t_max
    assert DSPEngine._beat_half_window_sec(18000, 52.0) > t_lift


# --------------------------------------------------------------------- beat error convention


def test_beat_error_reports_half_interval_difference() -> None:
    """Intervals alternating P + d / P - d must be reported as a beat error of d."""
    assert SyntheticEscapeGenerator.beat_error_half_offset(0.6) == pytest.approx(0.6e-3)
    config = WatchConfig(bph=21600, lift_angle=52.0, sample_rate=48000)
    signal = SyntheticEscapeGenerator(config, seed=3).generate(
        duration_s=5.0, true_rate_spd=0.0, true_beat_error_ms=0.6, true_amplitude_deg=270.0
    )
    with DSPEngine(config=config) as engine:
        telemetry = engine.process_window(signal)
    assert telemetry.beat_error_ms == pytest.approx(0.6, abs=BEAT_ERROR_TOL_MS)


# ------------------------------------------------------------- independent physical signal


@pytest.mark.parametrize("bph", [18000, 21600, 28800, 36000])
@pytest.mark.parametrize("fs", [48000, 96000])
def test_engine_recovers_physical_watch_parameters(physical_signal, bph: int, fs: int) -> None:
    """Regression for the live bug: a +4 s/d, 0.2 ms, 279 deg watch must read as such."""
    signal = physical_signal(bph=bph, fs=fs, rate_spd=4.0, beat_error_ms=0.2, amplitude_deg=279.0)
    with DSPEngine(bph=bph, lift_angle=52.0, sample_rate=fs) as engine:
        telemetry = engine.process_window(signal)
    assert telemetry.bph == bph
    assert telemetry.bph_mismatch is False
    assert telemetry.rate_spd == pytest.approx(4.0, abs=RATE_TOL_SPD)
    assert telemetry.beat_error_ms == pytest.approx(0.2, abs=BEAT_ERROR_TOL_MS)
    assert telemetry.amplitude_deg == pytest.approx(279.0, abs=AMPLITUDE_TOL_DEG)


@pytest.mark.parametrize("sound_levels", [(1.0, 0.4, 0.8), (1.0, 0.5, 1.0)])
def test_engine_handles_unlock_dominant_piezo_sounds(physical_signal, sound_levels) -> None:
    """P1 louder than (or as loud as) P3, common with contact piezos, must not bias amplitude."""
    signal = physical_signal(bph=21600, sound_levels=sound_levels)
    with DSPEngine(bph=21600, lift_angle=52.0, sample_rate=48000) as engine:
        telemetry = engine.process_window(signal)
    assert telemetry.amplitude_deg == pytest.approx(279.0, abs=AMPLITUDE_TOL_DEG)
    assert telemetry.beat_error_ms == pytest.approx(0.2, abs=BEAT_ERROR_TOL_MS)


@pytest.mark.parametrize("amplitude_deg", [180.0, 240.0, 310.0])
def test_engine_tracks_amplitude_range(physical_signal, amplitude_deg: float) -> None:
    """Amplitude must be recovered across the service range, not only near 280 deg."""
    signal = physical_signal(bph=28800, amplitude_deg=amplitude_deg)
    with DSPEngine(bph=28800, lift_angle=52.0, sample_rate=48000) as engine:
        telemetry = engine.process_window(signal)
    assert telemetry.amplitude_deg == pytest.approx(amplitude_deg, abs=AMPLITUDE_TOL_DEG)


def test_legacy_pipeline_amplitude_and_beat_error_on_physical_signal(physical_signal) -> None:
    """EscapeDSPPipeline shares the corrected amplitude formula and beat error convention."""
    config = WatchConfig(bph=28800, lift_angle=52.0, sample_rate=48000)
    telemetry, beats = EscapeDSPPipeline(config).analyze(physical_signal(bph=28800))
    assert len(beats) >= 40
    assert telemetry.amplitude_deg == pytest.approx(279.0, abs=AMPLITUDE_TOL_DEG)
    assert telemetry.beat_error_ms == pytest.approx(0.2, abs=BEAT_ERROR_TOL_MS)


# ------------------------------------------------------------------------- BPH detection


@pytest.mark.parametrize("true_bph", list(DSPEngine.STANDARD_BPH))
def test_estimate_bph_detects_standard_rates(physical_signal, true_bph: int) -> None:
    """Envelope autocorrelation finds the real BPH, including with a 3 ms beat error."""
    signal = physical_signal(bph=true_bph, beat_error_ms=3.0, seconds=4.0)
    with DSPEngine(bph=true_bph, lift_angle=52.0, sample_rate=48000) as engine:
        filtered = engine.filter_signal(signal)
        macro_env, _ = engine.compute_envelopes(filtered)
    detected, score, _ = DSPEngine.estimate_bph(macro_env, 48000.0)
    assert detected == true_bph
    assert score >= DSPEngine.BPH_DETECT_MIN_SCORE


def test_estimate_bph_returns_zero_for_noise() -> None:
    """Pure noise has no beat periodicity, so no BPH must be claimed."""
    noise = np.random.default_rng(0).normal(0.0, 0.01, 3 * 48000)
    detected, _, _ = DSPEngine.estimate_bph(np.abs(noise), 48000.0)
    assert detected == 0


@pytest.mark.parametrize(
    ("true_bph", "configured_bph"),
    [(28800, 21600), (28800, 25200), (21600, 28800), (36000, 18000), (19800, 21600)],
)
def test_wrong_configured_bph_is_detected_and_corrected(
    physical_signal, true_bph: int, configured_bph: int
) -> None:
    """A wrong BPH used to yield e.g. +10800 s/d or 186 deg; now it is detected and flagged."""
    signal = physical_signal(bph=true_bph, beat_error_ms=0.2)
    with DSPEngine(bph=configured_bph, lift_angle=52.0, sample_rate=48000) as engine:
        telemetry = engine.process_window(signal)
    assert telemetry.bph_mismatch is True
    assert telemetry.detected_bph == true_bph
    assert telemetry.bph == true_bph
    assert telemetry.rate_spd == pytest.approx(4.0, abs=RATE_TOL_SPD)
    assert telemetry.amplitude_deg == pytest.approx(279.0, abs=AMPLITUDE_TOL_DEG)


def test_auto_bph_disabled_keeps_configuration_but_flags_mismatch(physical_signal) -> None:
    """With auto_bph=False the configured BPH is used, but the mismatch is still reported."""
    signal = physical_signal(bph=28800)
    with DSPEngine(bph=21600, lift_angle=52.0, sample_rate=48000, auto_bph=False) as engine:
        telemetry = engine.process_window(signal)
    assert telemetry.bph == 21600
    assert telemetry.detected_bph == 28800
    assert telemetry.bph_mismatch is True


def test_bph_mismatch_warning_is_logged_once(physical_signal, caplog) -> None:
    """The mismatch warning must not flood the log on every 250 ms window."""
    signal = physical_signal(bph=28800, seconds=4.0)
    with DSPEngine(bph=21600, lift_angle=52.0, sample_rate=48000) as engine:
        with caplog.at_level("WARNING", logger="wea.core.dsp_engine"):
            for _ in range(4):
                engine.process_window(signal)
    hits = [r for r in caplog.records if "does not match the signal" in r.getMessage()]
    assert len(hits) == 1


# ----------------------------------------------------------------- performance refactors


@pytest.mark.parametrize(("n", "win"), [(144000, 288), (288000, 577), (1000, 6), (10, 20)])
def test_moving_average_matches_numpy_convolve(n: int, win: int) -> None:
    """The FFT box filter is a drop-in replacement for np.convolve(..., 'same')."""
    x = np.abs(np.random.default_rng(n).normal(size=n))
    expected = np.convolve(x, np.ones(win) / win, mode="same")
    np.testing.assert_allclose(DSPEngine._moving_average_same(x, win), expected, atol=1e-12)


def _reference_peak_scan(env, adaptive_thresh, nominal, refractory, pre, post, comb_tol):
    """Original per-sample peak scan, kept as the oracle for the vectorized version."""
    peaks: list[int] = []
    i, limit, n = pre, len(env) - post, len(env)
    while i < limit:
        if peaks:
            expected_idx = peaks[-1] + int(round(nominal))
            c0, c1 = max(i, expected_idx - comb_tol), min(limit, expected_idx + comb_tol)
            if c1 > c0:
                cm = c0 + int(np.argmax(env[c0:c1]))
                if env[cm] > adaptive_thresh:
                    peaks.append(cm)
                    i = cm + refractory
                    continue
        if env[i] > adaptive_thresh:
            lm = i + int(np.argmax(env[i : min(n, i + refractory)]))
            if pre <= lm < limit:
                peaks.append(lm)
            i = lm + refractory
        else:
            i += 1
    return peaks


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_peak_scan_matches_reference(physical_signal, seed: int) -> None:
    """detect_macro_beat_peaks gives exactly the peaks of the original sample-by-sample loop."""
    fs, bph = 48000.0, 21600
    with DSPEngine(bph=bph, lift_angle=52.0, sample_rate=48000) as engine:
        signal = physical_signal(bph=bph, snr_db=6.0 + 6.0 * seed, seed=seed)
        macro_env, _ = engine.compute_envelopes(engine.filter_signal(signal))
        edge = 900
        got = engine.detect_macro_beat_peaks(macro_env, bph=bph, fs_eff=fs, edge_samples=edge)

    nominal = 3600.0 / bph * fs
    med = float(np.median(macro_env))
    mad = float(np.median(np.abs(macro_env - med))) + 1e-12
    clean_peak = DSPEngine._estimate_clean_beat_peak(macro_env, fs)
    thresh = max(med + 5.5 * 1.4826 * mad, med + 0.22 * (clean_peak - med))
    expected = _reference_peak_scan(
        macro_env,
        thresh,
        nominal,
        max(1, int(round(DSPEngine.REFRACTORY_FACTOR * nominal))),
        edge,
        edge,
        int(round(0.15 * nominal)),
    )
    assert got == expected
    assert len(got) > 10


def test_filter_reuses_design_for_clock_drift_and_redesigns_for_new_rate() -> None:
    """Ppm-level drift must not trigger a Butterworth redesign on every window."""
    with DSPEngine(bph=21600, lift_angle=52.0, sample_rate=48000) as engine:
        assert engine._sos_for_fs(48000.0 * (1.0 + 300e-6)) is engine._sos
        sos_96k = engine._sos_for_fs(96000.0)
        assert sos_96k is not engine._sos
        assert engine._sos_for_fs(96000.0) is sos_96k  # cached

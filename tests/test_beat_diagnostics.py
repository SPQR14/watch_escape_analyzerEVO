"""Tests for the beat-timing diagnostics (line fit, residual classification, synthesizer)."""

from __future__ import annotations

import numpy as np
import pytest

from wea.core.beat_diagnostics import (
    beat_indices_from_times,
    classify_residuals,
    estimate_carrier_hz,
    fit_line_with_parity,
    residual_histogram,
    robust_line_fit,
    text_histogram,
)
from wea.core.synthetic import synthesize_escapement

T_21600 = 3600.0 / 21600.0


def _exact_times(rate_spd: float, be_ms: float, n: int = 180) -> np.ndarray:
    period = T_21600 / (1.0 + rate_spd / 86400.0)
    k = np.arange(n)
    return 0.05 + k * period + 0.5 * be_ms * 1e-3 * np.where(k % 2 == 0, 1.0, -1.0)


def test_beat_indices_tolerate_gaps() -> None:
    t = np.array([0.0, 1.0, 2.0, 4.0, 5.02, 8.0]) * T_21600
    np.testing.assert_array_equal(beat_indices_from_times(t, T_21600), [0, 1, 2, 4, 5, 8])


@pytest.mark.parametrize(("rate", "be"), [(5.0, 0.3), (-20.0, 0.0), (88.0, 1.2)])
def test_line_fit_recovers_rate_and_beat_error(rate: float, be: float) -> None:
    fit = fit_line_with_parity(_exact_times(rate, be), T_21600)
    assert fit.rate_spd == pytest.approx(rate, abs=1e-6)
    assert fit.beat_error_ms == pytest.approx(be, abs=1e-9)
    assert np.max(np.abs(fit.residuals_us)) < 1e-3


def test_robust_fit_ignores_far_cluster_but_reports_it() -> None:
    rng = np.random.default_rng(0)
    t = _exact_times(5.0, 0.3) + rng.normal(0.0, 30e-6, 180)
    jumped = rng.random(180) < 0.15
    t[jumped] += 1.4e-3  # sub-event confusion
    fit, inliers = robust_line_fit(t, T_21600)
    assert fit.rate_spd == pytest.approx(5.0, abs=0.5)
    assert fit.beat_error_ms == pytest.approx(0.3, abs=0.02)
    assert np.all(~inliers[jumped])
    assert np.median(fit.residuals_us[jumped]) == pytest.approx(1400.0, abs=60.0)


def test_classify_bell() -> None:
    rng = np.random.default_rng(1)
    for _ in range(10):
        assert classify_residuals(rng.normal(0.0, 60.0, 180), carrier_hz=6000.0).kind == "bell"


def test_classify_carrier_cycle_slips() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 15.0, 180) + 166.7 * rng.choice([-1, 0, 1], 180, p=[0.2, 0.6, 0.2])
    cls = classify_residuals(r, carrier_hz=6000.0)
    assert cls.kind == "carrier_steps"
    assert len(cls.cluster_centers_us) == 3
    assert np.diff(cls.cluster_centers_us) == pytest.approx([166.7, 166.7], abs=25.0)


def test_classify_subevent_jumps() -> None:
    rng = np.random.default_rng(3)
    r = np.r_[rng.normal(0.0, 60.0, 150), rng.normal(1400.0, 60.0, 30)]
    cls = classify_residuals(r, carrier_hz=6000.0)
    assert cls.kind == "subevent_steps"
    assert cls.cluster_fractions == pytest.approx([150 / 180, 30 / 180], abs=1e-9)


def test_wide_bell_is_not_called_carrier_steps() -> None:
    rng = np.random.default_rng(4)
    cls = classify_residuals(rng.normal(0.0, 250.0, 180), carrier_hz=6000.0)
    assert cls.kind == "bell"
    assert "too wide" in cls.summary


def test_estimate_carrier_and_histograms() -> None:
    fs = 48000
    t = np.arange(fs) / fs
    assert estimate_carrier_hz(np.sin(2 * np.pi * 6000.0 * t), fs) == pytest.approx(6000, abs=10)
    edges, counts = residual_histogram(np.random.default_rng(5).normal(0, 40, 500))
    assert counts.sum() == 500
    assert len(counts) <= 60
    text = text_histogram(edges, counts)
    assert "us |" in text and "#" in text


def test_synthesize_escapement_ground_truth() -> None:
    """Onsets follow rate / beat error exactly; missing beats are removed."""
    x, onsets = synthesize_escapement(
        rate_spd=-20.0, beat_error_ms=0.3, seconds=10.0, jitter_us=0.0, snr_db=30.0
    )
    assert len(x) == 480000
    fit = fit_line_with_parity(onsets, T_21600)
    assert fit.rate_spd == pytest.approx(-20.0, abs=1e-6)
    assert fit.beat_error_ms == pytest.approx(0.3, abs=1e-9)
    _, sparse = synthesize_escapement(seconds=10.0, missing_fraction=0.2, seed=1)
    assert 0.6 * len(onsets) < len(sparse) < 0.95 * len(onsets)

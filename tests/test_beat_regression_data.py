"""Regression tests on saved beat timestamps from ``scripts/diagnose_beats.py``.

``tests/data/mido_like`` is a SYNTHETIC stand-in (weak unlock, 2 ms case ring) that
reproduces the +-30 s/d paper-tape symptom; its ``report.json`` holds the generator
parameters so the audio can be rebuilt deterministically.

Real recordings: run ``python scripts/diagnose_beats.py --record 30 --device N
--out tests/data/real_<name>`` and commit the ``beats_*.npy`` + ``report.json``; the
``real`` tests below pick them up automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wea.core.beat_diagnostics import classify_residuals, robust_line_fit
from wea.core.beat_timing import detect_onsets
from wea.core.synthetic import synthesize_escapement

DATA = Path(__file__).parent / "data"
MIDO_LIKE = DATA / "mido_like"
REAL_DIRS = sorted(p for p in DATA.glob("real*") if (p / "report.json").exists())


def _report(path: Path) -> dict:
    return json.loads((path / "report.json").read_text())


def test_mido_like_old_detector_shows_subevent_jumps() -> None:
    report = _report(MIDO_LIKE)
    period = 3600.0 / report["bph"]
    fit, _ = robust_line_fit(np.load(MIDO_LIKE / "beats_engine.npy"), period)
    assert classify_residuals(fit.residuals_us, carrier_hz=6000.0).kind == "subevent_steps"


def test_mido_like_onset_detector_is_a_narrow_bell() -> None:
    report = _report(MIDO_LIKE)
    period = 3600.0 / report["bph"]
    fit, inliers = robust_line_fit(np.load(MIDO_LIKE / "beats_onset.npy"), period)
    cls = classify_residuals(fit.residuals_us, carrier_hz=6000.0)
    assert cls.kind == "bell"
    assert cls.robust_sigma_us < 80.0
    assert inliers.mean() > 0.95
    truth = report["synthetic_truth"]
    assert fit.rate_spd == pytest.approx(truth["rate_spd"], abs=0.5)
    assert fit.beat_error_ms == pytest.approx(truth["beat_error_ms"], abs=0.05)


def test_mido_like_onsets_are_reproducible() -> None:
    """Any change of the onset detector output on this case must be deliberate."""
    report = _report(MIDO_LIKE)
    truth = dict(report["synthetic_truth"])
    truth["sound_levels"] = tuple(truth["sound_levels"])
    audio, _ = synthesize_escapement(
        bph=report["bph"],
        lift_angle=report["lift_angle"],
        seconds=report["seconds"],
        fs=report["fs"],
        seed=1,
        **truth,
    )
    beats = detect_onsets(audio, report["fs"], report["bph"], report["lift_angle"])
    stored = np.load(MIDO_LIKE / "beats_onset.npy")
    np.testing.assert_allclose(
        np.array([b.sample for b in beats]) / report["fs"], stored, atol=1e-6
    )


@pytest.mark.skipif(not REAL_DIRS, reason="no real recording in tests/data/real*")
@pytest.mark.parametrize("path", REAL_DIRS, ids=lambda p: p.name)
def test_real_recording_onsets_have_no_steps(path: Path) -> None:
    report = _report(path)
    period = 3600.0 / report["bph"]
    fit, inliers = robust_line_fit(np.load(path / "beats_onset.npy"), period)
    assert classify_residuals(fit.residuals_us).kind == "bell"
    assert inliers.mean() > 0.9

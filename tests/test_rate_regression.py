"""Tests for rate / beat-error regression (B) and its use by the live worker and UI.

Acceptance (spec): synthetic 21600 BPH, 6 kHz carrier, noise -> rate within +-0.5 s/d,
beat error within +-0.05 ms, standard deviation of the 15 s windowed rate < 1 s/d; also at
-20 s/d and with missing beats.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from wea.core.models import BeatTelemetry, WatchConfig
from wea.core.rate_regression import SlidingRateEstimator, beat_indices, fit_rate_be
from wea.core.synthetic import synthesize_escapement
from wea.ui.dsp_worker import DSPStreamWorker

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Keep one QApplication alive for the whole module (a temporary one is destroyed at once)
APP = QApplication.instance() or QApplication([])

FS = 48000
T = 3600.0 / 21600.0


def _times(rate: float, be_ms: float, n: int = 120, jitter_s: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    period = T / (1.0 + rate / 86400.0)
    k = np.arange(n)
    t = 0.05 + k * period + 0.5 * be_ms * 1e-3 * np.where(k % 2 == 0, 1.0, -1.0)
    return t + rng.normal(0.0, jitter_s, n)


# ------------------------------------------------------------------------ model


@pytest.mark.parametrize(("rate", "be"), [(5.0, 0.3), (-20.0, 0.3), (88.0, 0.0), (-68.0, 1.5)])
def test_exact_rate_and_beat_error(rate: float, be: float) -> None:
    fit = fit_rate_be(_times(rate, be), T)
    assert fit is not None
    assert fit.rate_spd == pytest.approx(rate, abs=1e-6)
    assert fit.beat_error_ms == pytest.approx(be, abs=1e-6)
    assert fit.n_used == fit.n_total == 120


def test_sign_convention_positive_gains() -> None:
    """A shorter period than nominal (watch gains) gives a positive rate."""
    t = 0.1 + np.arange(50) * T * (1.0 - 10e-6)
    assert fit_rate_be(t, T).rate_spd == pytest.approx(86400 * 10e-6 / (1 - 10e-6), rel=1e-6)


def test_indices_tolerate_missing_beats() -> None:
    t = _times(5.0, 0.3)
    keep = np.ones(len(t), dtype=bool)
    keep[[3, 4, 17, 50, 51, 52, 90]] = False
    np.testing.assert_array_equal(beat_indices(t[keep], T), np.flatnonzero(keep))
    fit = fit_rate_be(t[keep], T)
    assert fit.rate_spd == pytest.approx(5.0, abs=1e-6)
    assert fit.beat_error_ms == pytest.approx(0.3, abs=1e-6)


def test_outliers_beyond_5_mad_are_rejected() -> None:
    rng = np.random.default_rng(1)
    t = _times(5.0, 0.3, n=180, jitter_s=40e-6, seed=1)
    bad = rng.random(180) < 0.12
    t[bad] += 1.4e-3  # sub-event confusion
    fit = fit_rate_be(t, T)
    assert fit.n_used <= 180 - bad.sum() + 2
    assert fit.rate_spd == pytest.approx(5.0, abs=0.5)
    assert fit.beat_error_ms == pytest.approx(0.3, abs=0.02)
    assert fit.residual_sigma_us == pytest.approx(40.0, rel=0.3)


def test_segments_get_their_own_intercept_and_parity() -> None:
    """Across an audio gap of unknown length the phase (and tick/tock parity) restart."""
    a = _times(5.0, 0.3, n=80)
    b = _times(5.0, 0.3, n=80)[1:] + 20.0123  # odd start: parity flipped
    t = np.r_[a, b]
    seg = np.r_[np.zeros(80, int), np.ones(79, int)]
    idx = np.r_[np.arange(80), np.arange(79)]
    fit = fit_rate_be(t, T, indices=idx, segments=seg)
    assert fit.rate_spd == pytest.approx(5.0, abs=1e-6)
    assert fit.beat_error_ms == pytest.approx(0.3, abs=1e-6)


def test_rate_sigma_matches_scatter() -> None:
    fits = [fit_rate_be(_times(3.0, 0.2, 90, 40e-6, seed=s), T) for s in range(60)]
    rates = np.array([f.rate_spd for f in fits])
    predicted = float(np.median([f.rate_sigma_spd for f in fits]))
    assert np.std(rates) == pytest.approx(predicted, rel=0.35)


def test_too_few_beats_returns_none() -> None:
    assert fit_rate_be(_times(0.0, 0.0, n=5), T) is None


def test_sliding_window_and_retroactive_calibration() -> None:
    t = _times(5.0, 0.3, n=400)
    est = SlidingRateEstimator(T, window_s=15.0)
    beats = [
        type("B", (), {"index": i, "sample": ti * FS, "segment": 0, "peak": 1.0, "noise": 0.01})
        for i, ti in enumerate(t)
    ]
    est.add(beats)
    fit = est.fit(FS)
    assert fit.span_s <= 15.0 and len(est) <= int(15.0 / T) + 1
    assert fit.rate_spd == pytest.approx(5.0, abs=1e-6)
    # A card sampling 10 ppm fast: same samples now mean shorter times -> +0.864 s/d
    fit_cal = est.fit(FS * (1.0 + 10e-6))
    assert fit_cal.rate_spd - fit.rate_spd == pytest.approx(0.864, abs=0.01)


# ---------------------------------------------------------- live worker (end to end)


def _run_worker(signal: np.ndarray, chunk: int = 4096) -> list[BeatTelemetry]:
    app = APP
    worker = DSPStreamWorker(
        config=WatchConfig(bph=21600, lift_angle=51.0, sample_rate=FS),
        device_id=1,
        synthetic_mode=False,
    )
    out: list[BeatTelemetry] = []
    worker.telemetry_ready.connect(out.append)
    kwargs: dict = {}

    class FakeStream:
        def __init__(self, **kw):
            kwargs.update(kw)
            self.active = True

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    try:
        with patch("sounddevice.InputStream", side_effect=FakeStream):
            worker.start_stream()
            cb = kwargs["callback"]
            for pos in range(0, len(signal) - chunk + 1, chunk):
                cb(signal[pos : pos + chunk].astype(np.float32).reshape(-1, 1), chunk, {}, None)
                app.processEvents()
    finally:
        worker.shutdown()
    return out


CASES = {
    "+5 s/d": dict(rate_spd=5.0, beat_error_ms=0.3, amplitude_deg=290.0, snr_db=20.0, seed=21),
    "-20 s/d": dict(rate_spd=-20.0, beat_error_ms=0.3, amplitude_deg=280.0, snr_db=20.0, seed=22),
    "missing beats": dict(
        rate_spd=5.0, beat_error_ms=0.3, amplitude_deg=290.0, snr_db=20.0,
        missing_fraction=0.15, seed=23,
    ),
    "mido-like weak unlock": dict(
        rate_spd=5.0, beat_error_ms=0.3, amplitude_deg=299.0, snr_db=15.0,
        sound_levels=(0.2, 0.5, 1.0), decay_s=0.002, seed=24,
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_live_rate_and_beat_error_meet_acceptance(case: str) -> None:
    params = CASES[case]
    signal, _ = synthesize_escapement(seconds=40.0, carrier_hz=6000.0, lift_angle=51.0, **params)
    telemetry = _run_worker(signal)
    full = [t for t in telemetry if t.rate_valid and t.timing.window_fit.span_s >= 14.0]
    assert len(full) >= 40
    rates = np.array([t.rate_spd for t in full])
    bes = np.array([t.beat_error_ms for t in full])
    assert np.mean(rates) == pytest.approx(params["rate_spd"], abs=0.5)
    assert np.std(rates) < 1.0
    assert np.all(np.abs(rates - params["rate_spd"]) < 1.0)
    assert np.mean(bes) == pytest.approx(params["beat_error_ms"], abs=0.05)


def test_live_rate_is_invalid_until_three_seconds_of_beats() -> None:
    signal, _ = synthesize_escapement(seconds=6.0, seed=25)
    telemetry = _run_worker(signal)
    assert telemetry and telemetry[0].rate_valid is False
    assert any(t.rate_valid for t in telemetry)
    first_valid = next(t for t in telemetry if t.rate_valid)
    assert first_valid.timing.window_fit.span_s >= DSPStreamWorker.MIN_RATE_SPAN_S


# ---------------------------------------------------------------------------- UI


def test_ui_rate_indicator_and_mean_use_the_regression() -> None:
    from wea.ui.main_window import MainWin

    signal, _ = synthesize_escapement(seconds=20.0, rate_spd=5.0, beat_error_ms=0.3, seed=26)
    telemetry = _run_worker(signal)
    win = MainWin()
    try:
        win.spn_stab.setValue(0)
        for t in telemetry:
            win._on_telemetry_ready(t)
        last = telemetry[-1]
        assert win.card_rate.value_label.text() == f"{last.rate_spd:+.1f}"
        assert win.card_beat_error.value_label.text() == f"{last.beat_error_ms:.2f}"
        session = win.session_rate_fit()
        assert session is not None and session.rate_spd == pytest.approx(5.0, abs=0.5)
        assert win.lbl_stat_mean_val.text() == f"{session.rate_spd:+.2f} s/d"
        invalid = BeatTelemetry(
            rate_spd=0.0,
            beat_error_ms=0.0,
            amplitude_deg=280.0,
            beat_times_sec=np.empty(0),
            envelope=np.empty(0),
            confidence=0.0,
            timing=last.timing,
            rate_valid=False,
        )
        win._on_telemetry_ready(invalid)
        assert win.card_rate.value_label.text() == "---.-"
    finally:
        win.close()


def test_ring_buffer_read_since_reports_gaps() -> None:
    from wea.core.ring_buffer import AudioRingBuffer

    rb = AudioRingBuffer(capacity_samples=100)
    rb.write(np.arange(60, dtype=np.float64))
    start, data = rb.read_since(0)
    assert start == 0 and len(data) == 60
    rb.write(np.arange(60, 150, dtype=np.float64))
    start, data = rb.read_since(60)
    assert start == 60
    np.testing.assert_array_equal(data, np.arange(60, 150))
    # Position 10 was overwritten: the oldest available sample (50) is returned instead
    start, data = rb.read_since(10)
    assert start == 50 and data[0] == 50.0
    rb.clear()
    start, data = rb.read_since(150)  # beyond the data after a clear
    assert start == 0 and len(data) == 0

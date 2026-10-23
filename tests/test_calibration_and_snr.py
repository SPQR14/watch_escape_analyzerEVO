"""Tests for sound-card clock calibration and the unified SNR (D)."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from wea.core.clock_calibration import (
    CalibrationResult,
    measure_ppm_from_ticks,
    measure_ppm_from_tone,
)
from wea.core.dsp_engine import DSPEngine
from wea.core.models import BeatTelemetry, WatchConfig
from wea.core.synthetic import synthesize_escapement
from wea.i18n.translator import Translator
from wea.ui.dsp_worker import DSPStreamWorker

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP = QApplication.instance() or QApplication([])
FS = 48000.0


def _resample_to_card(signal: np.ndarray, card_ppm: float, fs: float = FS) -> np.ndarray:
    """What a card sampling ``card_ppm`` fast records from audio synthesized at ``fs``."""
    fs_true = fs * (1.0 + card_ppm * 1e-6)
    n = int(len(signal) / fs * fs_true)
    return np.interp(np.arange(n) / fs_true, np.arange(len(signal)) / fs, signal)


def _tone(card_ppm: float, seconds: float = 60.0, hz: float = 6.0, seed: int = 0) -> np.ndarray:
    fs_true = FS * (1.0 + card_ppm * 1e-6)
    t = np.arange(int(seconds * fs_true)) / fs_true
    rng = np.random.default_rng(seed)
    return 0.5 * np.sin(2 * np.pi * hz * t + 0.4) + rng.normal(0.0, 0.05, len(t))


def _quartz_ticks(card_ppm: float, seconds: float = 60.0, gain_spd: float = 0.0) -> np.ndarray:
    fs_true = FS * (1.0 + card_ppm * 1e-6)
    period = 1.0 / (1.0 + gain_spd / 86400.0)
    x = np.random.default_rng(1).normal(0.0, 0.002, int(seconds * fs_true))
    tb = np.arange(400) / fs_true
    burst = np.sin(2 * np.pi * 4000 * tb) * np.exp(-tb / 1e-3) * (1 - np.exp(-tb / 1e-4))
    for k in range(int(seconds / period) - 1):
        pos = (0.3 + k * period) * fs_true
        i0, w = int(pos), pos - int(pos)
        x[i0 : i0 + 400] += (1 - w) * burst
        x[i0 + 1 : i0 + 401] += w * burst
    return x


# ----------------------------------------------------------------------- methods


@pytest.mark.parametrize("card_ppm", [-37.0, 0.0, 23.5])
def test_tone_reference(card_ppm: float) -> None:
    res = measure_ppm_from_tone(_tone(card_ppm), FS, tone_hz=6.0)
    assert res.method == "tone"
    assert res.ppm == pytest.approx(card_ppm, abs=max(0.5, 4 * res.sigma_ppm))
    assert 0.0 < res.sigma_ppm < 0.5
    assert res.rate_bias_spd == pytest.approx(res.ppm * 0.0864)


@pytest.mark.parametrize("card_ppm", [-37.0, 23.5])
def test_quartz_tick_reference(card_ppm: float) -> None:
    res = measure_ppm_from_ticks(_quartz_ticks(card_ppm), FS, period_s=1.0)
    assert res.ppm == pytest.approx(card_ppm, abs=0.05)
    assert res.n_points >= 55


def test_quartz_reference_rate_is_accounted_for() -> None:
    """A quartz gaining 0.5 s/d on an exact card: 0 ppm once its rate is declared."""
    ticks = _quartz_ticks(0.0, gain_spd=0.5)
    declared = measure_ppm_from_ticks(ticks, FS, 1.0, reference_rate_spd=0.5)
    undeclared = measure_ppm_from_ticks(ticks, FS, 1.0)
    assert declared.ppm == pytest.approx(0.0, abs=0.05)
    assert undeclared.ppm == pytest.approx(-0.5 / 86400 * 1e6, abs=0.05)


def test_six_hz_pulse_reference() -> None:
    sig, _ = synthesize_escapement(
        rate_spd=0.0, beat_error_ms=0.0, jitter_us=0.0, seconds=60, seed=2
    )
    res = measure_ppm_from_ticks(_resample_to_card(sig, 23.5), FS, period_s=1.0 / 6.0)
    assert res.ppm == pytest.approx(23.5, abs=0.5)


def test_calibration_rejects_unusable_input() -> None:
    with pytest.raises(ValueError):
        measure_ppm_from_tone(np.zeros(1000), FS, 6.0)
    with pytest.raises(ValueError):
        measure_ppm_from_ticks(np.random.default_rng(0).normal(0, 1e-3, int(5 * FS)), FS, 1.0)


# ------------------------------------------------------------------ live worker


class _FakeStream:
    kwargs: dict = {}

    def __init__(self, **kw):
        _FakeStream.kwargs = kw
        self.active = True

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


def _feed(worker: DSPStreamWorker, signal: np.ndarray, chunk: int = 4096) -> None:
    cb = _FakeStream.kwargs["callback"]
    for pos in range(0, len(signal) - chunk + 1, chunk):
        cb(signal[pos : pos + chunk].astype(np.float32).reshape(-1, 1), chunk, {}, None)
        APP.processEvents()


def _worker() -> DSPStreamWorker:
    return DSPStreamWorker(
        config=WatchConfig(bph=21600, lift_angle=51.0, sample_rate=int(FS)),
        device_id=1,
        synthetic_mode=False,
    )


def test_worker_measures_tone_calibration_on_live_input() -> None:
    worker = _worker()
    results: list[object] = []
    worker.calibration_measured.connect(results.append)
    try:
        with patch("sounddevice.InputStream", side_effect=_FakeStream):
            worker.start_stream()
            worker.start_calibration("tone", seconds=30.0, reference=6.0)
            _feed(worker, _tone(23.5, seconds=33.0))
        assert len(results) == 1 and isinstance(results[0], CalibrationResult)
        assert results[0].ppm == pytest.approx(23.5, abs=1.0)
        assert worker.calibration_progress == 0.0
    finally:
        worker.shutdown()


def test_worker_calibration_restarts_after_gap_and_can_be_cancelled() -> None:
    worker = _worker()
    results: list[object] = []
    worker.calibration_measured.connect(results.append)
    try:
        with patch("sounddevice.InputStream", side_effect=_FakeStream):
            worker.start_stream()
            worker.start_calibration("tone", seconds=20.0, reference=6.0)
            _feed(worker, _tone(0.0, seconds=8.0))
            progress = worker.calibration_progress
            assert progress > 0.2
            worker._handle_stream_discontinuity()
            _feed(worker, _tone(0.0, seconds=2.0, seed=3))
            assert worker.calibration_progress < progress
            worker.cancel_calibration()
            _feed(worker, _tone(0.0, seconds=21.0, seed=4))
        assert results == []
    finally:
        worker.shutdown()


def test_calibration_correction_applies_to_rate() -> None:
    """A +23.5 ppm card makes a +5 s/d watch read ~+2.97; the correction restores +5."""
    sig, _ = synthesize_escapement(rate_spd=5.0, beat_error_ms=0.3, seconds=25.0, seed=7)
    card = _resample_to_card(sig, 23.5)
    rates = {}
    for ppm in (0.0, 23.5):
        worker = _worker()
        out: list[BeatTelemetry] = []
        worker.telemetry_ready.connect(out.append)
        try:
            with patch("sounddevice.InputStream", side_effect=_FakeStream):
                worker.start_stream()
                worker.set_calibration_ppm(ppm)
                _feed(worker, card)
        finally:
            worker.shutdown()
        rates[ppm] = out[-1].rate_spd
    assert rates[0.0] == pytest.approx(5.0 - 23.5 * 0.0864, abs=0.3)
    assert rates[23.5] == pytest.approx(5.0, abs=0.3)


def test_ui_calibration_controls() -> None:
    from wea.ui.main_window import MainWin

    win = MainWin()
    try:
        win.spn_cal_ppm.setValue(12.5)
        assert win.dsp_worker.drift_estimator.calibration_ppm == pytest.approx(12.5)
        assert win.dsp_worker.drift_estimator.effective_fs == pytest.approx(FS * (1 + 12.5e-6))
        assert win.start_clock_calibration() is False  # no live capture
        win._on_calibration_measured(CalibrationResult(-8.25, 0.2, "tone", 60.0, 360))
        assert win.spn_cal_ppm.value() == pytest.approx(-8.25)
        assert "-8.25" in win.lbl_cal_result.text()
        win._on_calibration_measured("no tone")
        assert win.lbl_cal_result.text() == win.translator.tr("cal_failed")
    finally:
        win.close()


# ---------------------------------------------------------------------------- SNR


def test_snr_label_and_bar_show_the_same_value() -> None:
    from wea.ui.main_window import MainWin

    win = MainWin()
    try:
        for snr in (31.04, 55.0, 0.0):
            win._show_snr(snr)
            shown = float(win.lbl_snr_value.text().split()[0])
            assert shown == pytest.approx(snr, abs=0.05)
            assert win.prg_snr.value() == min(round(snr), win.SNR_BAR_FULL_SCALE_DB)
            assert win.prg_snr.format() == f"{snr:.0f} dB"
    finally:
        win.close()


def test_engine_snr_uses_pre_beat_noise_definition() -> None:
    snrs = []
    for snr_db in (12.0, 24.0):
        sig, _ = synthesize_escapement(seconds=4.0, snr_db=snr_db, seed=8)
        with DSPEngine(bph=21600, lift_angle=51.0, sample_rate=48000) as engine:
            snrs.append(engine.process_window(sig).snr_db)
    assert snrs[1] - snrs[0] == pytest.approx(12.0, abs=3.0)


@pytest.mark.parametrize("locale", ["en", "es"])
def test_new_ui_strings_are_translated(locale: str) -> None:
    tr = Translator(locale)
    for key in (
        "cal_title", "cal_ppm", "cal_measure", "cal_ref_tone6", "cal_ref_pulses6",
        "cal_ref_quartz1", "cal_need_capture", "cal_measuring", "cal_failed",
        "btn_tape_rate", "rate_window",
    ):  # fmt: skip
        assert tr.tr(key) != key

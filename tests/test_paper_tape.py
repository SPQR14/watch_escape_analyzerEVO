"""Phase paper-tape tests (C): y = t - n T - t0, slopes, segments and the rate overlay."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from wea.core.models import TimingUpdate
from wea.ui.main_window import MainWin

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
APP = QApplication.instance() or QApplication([])
FS = 48000.0
T = 3600.0 / 21600.0


@pytest.fixture
def win():
    window = MainWin()
    yield window
    window.close()


def _timing(k, t, seg=None, period=T) -> TimingUpdate:
    k = np.asarray(k)
    return TimingUpdate(
        indices=k,
        samples=np.asarray(t) * FS,
        segments=np.zeros(len(k), int) if seg is None else np.asarray(seg),
        fs_eff=FS,
        nominal_period_s=period,
    )


@pytest.mark.parametrize("rate", [-20.0, 0.0, 6.0, 88.0])
def test_slope_angle_matches_phase_drift(rate: float) -> None:
    slope = np.tan(np.radians(MainWin.rate_slope_angle_deg(rate)))
    assert slope == pytest.approx(-rate / 86.4)


def test_phase_series_definition(win) -> None:
    k = np.arange(60)
    t = 2.0 + k * T * (1.0 - 10e-6)
    win._append_measurement_beats(_timing(k, t))
    tape = win.tape_phase_series()
    np.testing.assert_allclose(tape["x"], t - t[0])
    np.testing.assert_allclose(tape["y"], (t - k * T - t[0]) * 1000.0, atol=1e-9)
    assert tape["even"].sum() == 30 and tape["odd"].sum() == 30


def test_segments_continue_and_are_not_connected(win) -> None:
    k1 = np.arange(40)
    t1 = k1 * T
    k2 = np.arange(40)
    t2 = 20.0 + k2 * T + 0.0123  # after a gap: arbitrary phase
    win._append_measurement_beats(_timing(np.r_[k1, k2], np.r_[t1, t2], np.r_[[0] * 40, [1] * 40]))
    tape = win.tape_phase_series()
    # Second segment starts where the first ended instead of jumping by 12.3 ms
    assert tape["y"][40] == pytest.approx(tape["y"][39], abs=1e-6)
    win._refresh_paper_tape()
    x, _ = win.tape_curve_even.getData()
    assert len(x) == 40


def test_bph_change_restarts_measurement_beats(win) -> None:
    win._append_measurement_beats(_timing(np.arange(20), np.arange(20) * T))
    win._append_measurement_beats(_timing(np.arange(5), np.arange(5) * 0.125, period=0.125))
    assert len(win._meas_sample) == 5 and win._meas_period_s == 0.125


def test_rate_overlay_toggle_and_window_spinbox(win) -> None:
    item = win.paper_tape_plot.getPlotItem()
    assert not win.tape_rate_curve.isVisible()
    win.btn_tape_rate.setChecked(True)
    assert win.tape_rate_curve.isVisible() and item.getAxis("right").isVisible()
    win.btn_tape_rate.setChecked(False)
    assert not win.tape_rate_curve.isVisible()

    win.spn_rate_window.setValue(30)
    assert win.dsp_worker.rate_window_s == 30.0
    assert win.dsp_worker.rate_estimator.window_s == 30.0

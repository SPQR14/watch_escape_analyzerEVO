"""UI refresh regression tests: waveform coalescing, cheap curves, BPH warning, background DSP."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from wea.core.models import BeatTelemetry
from wea.i18n.translator import Translator
from wea.ui.main_window import MainWin

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def win():
    app = QApplication.instance() or QApplication([])
    assert app is not None
    window = MainWin()
    yield window
    window.close()


def _frame(n: int, value: float) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(n, dtype=np.float64), np.full(n, value, dtype=np.float64)


def test_main_window_runs_dsp_in_background(win) -> None:
    """The live worker must analyze off its event loop so waveforms keep flowing."""
    assert win.dsp_worker.analysis_in_background is True
    assert win.dsp_worker._analysis_executor is not None


def test_high_rate_curves_use_cheap_rendering(win) -> None:
    """Wide or antialiased pens are very slow in Qt's raster engine (notably on Windows)."""
    for curve in (win.osc_wave_curve, win.osc_envelope_curve, win.audio_waveform_curve):
        assert curve.opts["antialias"] is False
        pen = curve.opts["pen"]
        assert pen.widthF() == pytest.approx(1.0)
        assert pen.isCosmetic() is True
    # The paper tape keeps its dot symbols (phase tape: one dot per beat)
    assert win.tape_curve_even.opts["symbol"] == "o"
    assert win.tape_curve_even.opts["antialias"] is True


def test_waveform_frames_are_coalesced_to_latest(win) -> None:
    """A burst of frames draws the first immediately and only the newest one afterwards."""
    win._last_waveform_draw = 0.0
    win._on_waveform_ready(*_frame(10, 1.0))
    _, y = win.audio_waveform_curve.getData()
    assert y is not None and float(y[0]) == 1.0

    win._on_waveform_ready(*_frame(10, 2.0))
    win._on_waveform_ready(*_frame(10, 3.0))
    _, y = win.audio_waveform_curve.getData()
    assert float(y[0]) == 1.0  # throttled: still the first frame
    assert win._waveform_flush_timer.isActive()

    deadline = time.monotonic() + 1.0
    while win._pending_waveform is not None and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    _, y = win.audio_waveform_curve.getData()
    assert float(y[0]) == 3.0  # newest frame wins, frame 2 was skipped


def test_telemetry_does_not_overwrite_streamed_waveform(win) -> None:
    """Telemetry used to redraw the waveform from another window, making the trace jump."""
    win._on_waveform_ready(*_frame(10, 0.5))
    telemetry = BeatTelemetry(
        rate_spd=4.0,
        beat_error_ms=0.2,
        amplitude_deg=279.0,
        beat_times_sec=np.empty(0),
        envelope=np.ones(48000),
        confidence=0.9,
        filtered_signal=np.sin(np.linspace(0.0, 400.0, 48000)),
        effective_fs=48000.0,
        bph=28800,
    )
    win._on_telemetry_ready(telemetry)
    _, y = win.audio_waveform_curve.getData()
    assert len(y) == 10 and float(y[0]) == 0.5


@pytest.mark.parametrize("locale", ["en", "es"])
def test_bph_mismatch_warning_in_status_bar(win, locale: str) -> None:
    """A wrong BPH setting is surfaced to the user and cleared once it matches."""
    win.translator = Translator(locale)
    configured = win.current_watch_config().bph
    mismatch = BeatTelemetry(
        rate_spd=4.0,
        beat_error_ms=0.2,
        amplitude_deg=279.0,
        beat_times_sec=np.empty(0),
        envelope=np.empty(0),
        confidence=0.9,
        bph=28800,
        detected_bph=28800,
        bph_mismatch=True,
    )
    win._on_telemetry_ready(mismatch)
    message = win.statusBar().currentMessage()
    assert "28800" in message and str(configured) in message

    mismatch.bph_mismatch = False
    win._on_telemetry_ready(mismatch)
    assert win.statusBar().currentMessage() == ""

"""Headless UI tests for MainWin, Translator switching, and real-time telemetry updates."""

from __future__ import annotations

import os
import numpy as np
from PyQt6.QtWidgets import QApplication

from wea.core.models import BeatEvent, TelemetryResult
from wea.i18n.translator import Translator
from wea.ui.main_window import MainWin

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_layout_and_i18n_switching() -> None:
    """Verify MainWin constructs all 3 columns, top bar controls, and switches EN/ES."""
    app = QApplication.instance() or QApplication([])
    assert app is not None

    tr = Translator("en")
    win = MainWin(translator=tr)

    # Verify 3-column splitter exists
    assert win.splitter.count() == 3
    assert win.btn_start.text() == "START"
    assert win.btn_stop.text() == "STOP"

    # Switch to Spanish via language ComboBox
    win.cmb_lang.setCurrentIndex(1)
    assert win.translator.locale == "es"
    assert win.btn_start.text() == "INICIAR"
    assert win.btn_stop.text() == "DETENER"
    assert win.card_rate.title_label.text() == "MARCHA"

    # Switch caliber preset to Seiko NH35 / 4R36 and verify BPH & Lift Angle update
    win.cmb_caliber.setCurrentText("Seiko NH35 / 4R36")
    cfg = win.current_watch_config()
    assert cfg.bph == 21600
    assert cfg.lift_angle == 53.0

    # Simulate a telemetry update callback from the worker thread
    fake_wave = np.sin(np.linspace(0, 20 * np.pi, 500))
    fake_beat = BeatEvent(
        timestamp_p1=0.100,
        timestamp_p2=0.1005,
        timestamp_p3=0.1018,
        peak_energy=1.0,
        raw_waveform=fake_wave,
    )
    fake_telemetry = TelemetryResult(
        rate_spd=5.4,
        beat_error_ms=0.35,
        amplitude_deg=278.0,
        confidence=0.96,
    )
    win._on_telemetry_updated(fake_telemetry, [fake_beat], elapsed_s=4.5, snr_db=39.2)

    assert win.card_rate.value_label.text() == "+5.4"
    assert win.card_beat_error.value_label.text() == "0.35"
    assert win.card_amplitude.value_label.text() == "278°"
    win.close()

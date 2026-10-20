"""Headless UI tests for MainWin, Translator switching, and real-time telemetry updates."""

from __future__ import annotations

import os
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QListView

from wea.core.models import BeatEvent, TelemetryResult
from wea.i18n.translator import Translator
from wea.ui.main_window import MainWin

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_layout_and_i18n_switching() -> None:
    """Verify MainWin constructs transposed graph panels, macOS combo/spin controls, COSC lines, and EN/ES."""
    app = QApplication.instance() or QApplication([])
    assert app is not None

    tr = Translator("en")
    win = MainWin(translator=tr)

    # Verify main horizontal splitter (Left Panel + Right Transposed Graphs Splitter)
    assert win.splitter.count() == 2
    assert win.graphs_splitter.orientation() == Qt.Orientation.Vertical
    assert win.graphs_splitter.count() == 2
    assert win.bottom_graphs_splitter.orientation() == Qt.Orientation.Horizontal
    assert win.bottom_graphs_splitter.count() == 2

    # Verify QComboBox uses QListView popup and sufficient minimum width on macOS
    assert isinstance(win.cmb_caliber.view(), QListView)
    assert win.cmb_caliber.view().minimumWidth() >= win.cmb_caliber.minimumWidth()

    assert win.btn_start.text() == "START"
    assert win.btn_stop.text() == "STOP"
    assert win.btn_center_trace.text() == "Center Trace"
    assert win.btn_save_position.text() == "Save Position"
    assert "Global Statistics" in win.btn_global_stats.text()
    assert win.txt_test_name.text() == "Caliber-Test-01"

    # Verify COSC horizontal lines (-4 and +6 labels) and point symbols ('o')
    assert win.cosc_lower_line.value() == -4.0
    assert win.cosc_upper_line.value() == 6.0
    assert win.cosc_lower_line.angle == 0
    assert win.cosc_upper_line.angle == 0
    assert win.cosc_lower_line.label.format == "-4"
    assert win.cosc_upper_line.label.format == "+6"
    assert win.tape_curve_even.opts["symbol"] == "o"
    assert win.tape_curve_odd.opts["symbol"] == "o"

    # Switch to Spanish via language ComboBox
    win.cmb_lang.setCurrentIndex(1)
    assert win.translator.locale == "es"
    assert win.btn_start.text() == "INICIAR"
    assert win.btn_stop.text() == "DETENER"
    assert win.btn_center_trace.text() == "Centrar Trazo"
    assert win.btn_save_position.text() == "Guardar Posición"
    assert "Estadísticas Globales" in win.btn_global_stats.text()
    assert win.lbl_test_name.text() == "Nombre de la prueba"
    assert win.card_rate.title_label.text() == "MARCHA"

    # Switch caliber preset to Seiko NH35 / 4R36 and verify BPH & Lift Angle update
    win.cmb_caliber.setCurrentText("Seiko NH35 / 4R36")
    cfg = win.current_watch_config()
    assert cfg.bph == 21600
    assert cfg.lift_angle == 53.0

    # Simulate telemetry update callbacks and verify straight trace + oscilloscope + audio waveform
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
    win._on_telemetry_updated(fake_telemetry, [fake_beat], elapsed_s=4.5, snr_db=39.2, audio_chunk=fake_wave)
    win._on_telemetry_updated(fake_telemetry, [fake_beat], elapsed_s=5.0, snr_db=39.2, audio_chunk=fake_wave)

    assert win.card_rate.value_label.text() == "+5.4"
    assert win.card_beat_error.value_label.text() == "0.35"
    assert win.card_amplitude.value_label.text() == "278°"

    # Verify both paper tape trace and audio waveform curve contain data
    x_even, y_even = win.tape_curve_even.getData()
    assert len(x_even) == 2
    assert np.allclose(y_even, y_even[0])
    x_wave, y_wave = win.audio_waveform_curve.getData()
    assert len(x_wave) > 0
    assert len(y_wave) > 0

    # Verify "Centrar Trazo" auto-centers Y-axis around current trace values (~5.4 s/d)
    win.center_paper_tape_trace()
    y_range = win.paper_tape_plot.viewRange()[1]
    y_mid = 0.5 * (y_range[0] + y_range[1])
    assert abs(y_mid - 5.4) < 0.5

    # Verify "Guardar Posición" records DD and updates the 3-position Global Stats table
    win.cmb_position.setCurrentText("DD")
    win.save_current_position()
    assert "DD" in win._saved_positions
    assert win.tbl_global_stats.rowCount() == 3
    assert "(3P)" in win.lbl_global_mean_val.text()

    # Verify collapsible Global Statistics section toggle
    win.btn_global_stats.setChecked(False)
    assert not win.global_stats_container.isVisibleTo(win)
    win.btn_global_stats.setChecked(True)
    assert win.global_stats_container.isVisibleTo(win)

    win.close()

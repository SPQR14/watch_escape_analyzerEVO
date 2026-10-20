"""Headless UI tests for MainWin, Translator switching, and real-time telemetry updates."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import QApplication, QListView

from wea.core.models import BeatEvent, TelemetryResult
from wea.i18n.translator import Translator
from wea.ui.main_window import MainWin
from wea.utils.logger_setup import LoggerSetup
from wea.utils.watch_db import load_watch_db

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_layout_and_i18n_switching(tmp_path: Path) -> None:
    """Verify MainWin constructs transposed graph panels, watch_db presets, clean stats, exports, and EN/ES."""
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

    # Verify initial caliber is Custom, 52° lift angle, and 21600 BPH, and items match watch_db.csv
    db_calibers = load_watch_db()
    combo_items = [win.cmb_caliber.itemText(i) for i in range(win.cmb_caliber.count())]
    assert combo_items == list(db_calibers.keys())
    assert win.cmb_caliber.currentText() == "Custom"
    init_cfg = win.current_watch_config()
    assert init_cfg.bph == 21600
    assert init_cfg.lift_angle == 52.0

    # Verify telemetry statistics and positional table start completely clean
    assert win.lbl_stat_max_val.text() == ""
    assert win.lbl_stat_min_val.text() == ""
    assert win.lbl_stat_delta_val.text() == ""
    assert win.lbl_stat_mean_val.text() == ""
    assert win.lbl_stat_elapsed_val.text() == ""
    assert win.tbl_global_stats.rowCount() == 0
    assert win.lbl_global_mean_val.text() == ""
    assert win.lbl_global_delta_val.text() == ""

    assert win.btn_start.text() == "START"
    assert win.btn_stop.text() == "STOP"
    assert win.btn_center_trace.text() == "Center Trace"
    assert win.btn_save_position.text() == "Save Position"
    assert win.btn_save_csv.text() == "Export Results to CSV"
    assert win.btn_export_test_results.text() == "Export Test Results"
    assert "Buy me a coffee" in win.btn_buy_coffee.text()
    assert win.btn_export_traces.text() == "Export Traces"
    assert win.btn_print_traces.text() == "Print Traces"
    assert "RECORD AUDIO" in win.btn_record.text()
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
    assert win.btn_save_csv.text() == "Export Results to CSV"
    assert win.btn_export_test_results.text() == "Export Test Results"
    assert "Buy me a coffee" in win.btn_buy_coffee.text()
    assert win.btn_export_traces.text() == "Exportar Trazas"
    assert win.btn_print_traces.text() == "Imprimir Trazas"
    assert "GRABAR SONIDO" in win.btn_record.text()
    assert "Estadísticas Globales" in win.btn_global_stats.text()
    assert win.lbl_test_name.text() == "Nombre de la prueba"
    assert win.card_rate.title_label.text() == "MARCHA"

    # Switch caliber preset to Seiko NHXX/4RXX series and verify BPH & Lift Angle update from watch_db.csv
    win.cmb_caliber.setCurrentText("Seiko NHXX/4RXX series")
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

    # Verify "Guardar Posición" records DU and DD into Global Stats table (started empty)
    win.cmb_position.setCurrentText("DU")
    win.save_current_position()
    win.cmb_position.setCurrentText("DD")
    win.save_current_position()
    assert "DU" in win._saved_positions
    assert "DD" in win._saved_positions
    assert win.tbl_global_stats.rowCount() == 2
    assert "(2P)" in win.lbl_global_mean_val.text()

    # Verify CSV export functionality
    csv_out = win.save_results_csv(tmp_path / "telemetry_results.csv")
    assert csv_out is not None and csv_out.exists()
    with csv_out.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4  # 2 sample rows + 2 saved_position rows

    # Verify Export Test Results (only global positional statistics + positional delta)
    global_out = win.export_test_results(tmp_path / "global_test_results.csv")
    assert global_out is not None and global_out.exists()
    with global_out.open("r", encoding="utf-8") as fh:
        g_rows = list(csv.DictReader(fh))
    assert len(g_rows) == 3  # DU + DD + GLOBAL_SUMMARY
    assert g_rows[-1]["position"] == "GLOBAL_SUMMARY"
    assert g_rows[-1]["positional_delta_spd"] == "0.00"

    # Verify placeholder Buy me a coffee button click
    win.btn_buy_coffee.click()

    # Verify exporting and printing audio traces
    png_trace = win.export_traces(tmp_path / "traces.png")
    assert png_trace is not None and png_trace.exists() and png_trace.stat().st_size > 0
    csv_trace = win.export_traces(tmp_path / "traces.csv")
    assert csv_trace is not None and csv_trace.exists() and csv_trace.stat().st_size > 0

    pdf_printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    pdf_printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    pdf_path = tmp_path / "printed_traces.pdf"
    pdf_printer.setOutputFileName(str(pdf_path))
    assert win.print_traces(printer=pdf_printer) is True
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    # Verify audio recording matches exact test duration and saves as FLAC (default) and Ogg Vorbis (fallback)
    win.spn_duration.setValue(5)
    win.cmb_srate.setCurrentIndex(0)  # 48 kHz
    rec_samples, rec_fs = win._ensure_test_length_recording()
    assert rec_fs == 48000
    assert len(rec_samples) == 5 * 48000

    flac_path = win.record_audio(file_path=tmp_path / "capture_default", prefer_flac=True)
    assert flac_path is not None and flac_path.exists() and flac_path.stat().st_size > 0
    assert flac_path.suffix in (".flac", ".ogg")

    ogg_path = win.record_audio(file_path=tmp_path / "capture_vorbis.ogg", prefer_flac=False)
    assert ogg_path is not None and ogg_path.exists() and ogg_path.stat().st_size > 0
    assert ogg_path.suffix == ".ogg"

    # Verify unified log file exists and has logged critical events
    assert LoggerSetup.LOG_FILE.exists()
    log_text = LoggerSetup.LOG_FILE.read_text(encoding="utf-8")
    assert "wea.ui.main_window" in log_text
    assert "Saved positional telemetry" in log_text

    # Verify collapsible Global Statistics section toggle
    win.btn_global_stats.setChecked(False)
    assert not win.global_stats_container.isVisibleTo(win)
    win.btn_global_stats.setChecked(True)
    assert win.global_stats_container.isVisibleTo(win)

    win.close()

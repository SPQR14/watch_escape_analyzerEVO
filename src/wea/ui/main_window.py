"""Main window and real-time PyQtGraph visualization for Watch Escape Analyzer EVO."""

from __future__ import annotations

import time
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from wea.audio.ring_buffer import AudioRingBuffer
from wea.core.dsp_pipeline import EscapeDSPPipeline
from wea.core.models import BeatEvent, TelemetryResult, WatchConfig
from wea.core.synthetic import SyntheticEscapeGenerator
from wea.i18n.translator import Translator

# Configure PyQtGraph global options for crisp anti-aliased industrial plotting
pg.setConfigOptions(antialias=True, background="#0d1015", foreground="#9aa5b4")


CALIBER_PRESETS: dict[str, tuple[int, float]] = {
    "Custom": (28800, 52.0),
    "Seiko NH35 / 4R36": (21600, 53.0),
    "Miyota 9015": (28800, 51.0),
    "ETA 2824-2 / SW200": (28800, 50.0),
    "Rolex 3135": (28800, 52.0),
    "Omega 1861": (21600, 50.0),
    "Omega 2500 Co-Axial": (25200, 38.0),
    "Zenith El Primero 400": (36000, 52.0),
    "ETA 6497-1 (Unitas)": (18000, 44.0),
}

POSITION_VARIATION: dict[str, tuple[float, float, float]] = {
    "DU": (+4.5, 0.35, 282.0),
    "DD": (+3.8, 0.40, 278.0),
    "CU": (-1.5, 0.55, 248.0),
    "CD": (-2.2, 0.50, 245.0),
    "CL": (+1.2, 0.45, 255.0),
    "CR": (+0.6, 0.48, 252.0),
}


class SyntheticTelemetryWorker(QThread):
    """Background worker thread that synthesizes and analyzes escapement audio in real time.

    Connects `SyntheticEscapeGenerator`, `AudioRingBuffer`, and `EscapeDSPPipeline`
    without blocking the PyQt6 UI thread.
    """

    telemetry_updated = pyqtSignal(object, object, float, float)
    session_finished = pyqtSignal()

    def __init__(
        self,
        config: WatchConfig,
        position: str = "DU",
        stabilization_s: float = 2.0,
        duration_s: float = 30.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.position = position
        self.stabilization_s = float(stabilization_s)
        self.duration_s = float(duration_s)
        self._running = False

    def stop(self) -> None:
        """Request cooperative termination of the worker loop."""
        self._running = False

    def run(self) -> None:
        """Continuously synthesize, buffer, and process sliding acoustic windows."""
        self._running = True
        fs = self.config.sample_rate
        window_seconds = 3.0
        ring_buffer = AudioRingBuffer(capacity=int(window_seconds * fs))
        pipeline = EscapeDSPPipeline(config=self.config)

        base_rate, base_be, base_amp = POSITION_VARIATION.get(
            self.position, (+4.5, 0.35, 280.0)
        )
        rng = np.random.default_rng(int(time.time() * 1000) % (2**31 - 1))

        start_wall = time.monotonic()
        step_interval = 0.25
        step_idx = 0

        while self._running:
            elapsed = time.monotonic() - start_wall
            if elapsed > self.duration_s + self.stabilization_s:
                break

            inst_rate = base_rate + float(rng.normal(0.0, 0.08))
            inst_be = max(0.0, base_be + float(rng.normal(0.0, 0.015)))
            inst_amp = base_amp + float(rng.normal(0.0, 0.6))
            snr_db = 38.0 + float(rng.normal(0.0, 0.8))

            gen = SyntheticEscapeGenerator(config=self.config, seed=1000 + step_idx)
            synth_signal = gen.generate(
                duration_s=window_seconds,
                true_rate_spd=inst_rate,
                true_beat_error_ms=inst_be,
                true_amplitude_deg=inst_amp,
                snr_db=snr_db,
            )

            ring_buffer.write(synth_signal)
            window_data = ring_buffer.read_latest(int(window_seconds * fs))
            telemetry, beats = pipeline.analyze(window_data)

            if self._running:
                self.telemetry_updated.emit(telemetry, beats, elapsed, snr_db)

            step_idx += 1
            sleep_deadline = time.monotonic() + step_interval
            while self._running and time.monotonic() < sleep_deadline:
                self.msleep(25)

        self.session_finished.emit()


class HudCard(QFrame):
    """Industrial monospaced numeric HUD display card for a primary horological metric."""

    def __init__(
        self,
        title: str,
        unit: str,
        accent_color: str = "#00e5ff",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("HudCard")
        self._accent_color = accent_color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)

        header_layout = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setObjectName("HudCardTitle")
        self.unit_label = QLabel(unit)
        self.unit_label.setObjectName("HudCardUnit")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.unit_label)

        self.value_label = QLabel("---.-")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        mono_font = QFont("JetBrains Mono", 24, QFont.Weight.Bold)
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self.value_label.setFont(mono_font)
        self.value_label.setStyleSheet(f"color: {self._accent_color};")

        layout.addLayout(header_layout)
        layout.addWidget(self.value_label)

    def set_title(self, title: str) -> None:
        """Update localized title text."""
        self.title_label.setText(title.upper())

    def set_value(self, text: str) -> None:
        """Update numeric readout string."""
        self.value_label.setText(text)


class MainWin(QMainWindow):
    """Primary application window for Watch Escape Analyzer EVO."""

    COSC_MIN_SPD: float = -4.0
    COSC_MAX_SPD: float = +6.0

    def __init__(self, translator: Translator | None = None) -> None:
        super().__init__()
        self.translator = translator if translator is not None else Translator("en")
        self.worker: SyntheticTelemetryWorker | None = None

        # Telemetry history for statistics & horizontal paper tape trace
        self._rate_history: list[float] = []
        self._tape_times_even: list[float] = []
        self._tape_rate_even: list[float] = []
        self._tape_times_odd: list[float] = []
        self._tape_rate_odd: list[float] = []
        self._last_telemetry: TelemetryResult | None = None

        # Saved positional telemetry records (keyed by position: DU, DD, CU, CD, CL, CR)
        self._saved_positions: dict[str, TelemetryResult] = {
            "DU": TelemetryResult(rate_spd=+4.4, beat_error_ms=0.35, amplitude_deg=282.0, confidence=0.98),
            "CU": TelemetryResult(rate_spd=-1.4, beat_error_ms=0.52, amplitude_deg=249.0, confidence=0.96),
        }

        self.setMinimumSize(1360, 840)
        self._apply_industrial_theme()
        self._build_ui()
        self.retranslate_ui()
        self._refresh_global_stats_table()

    def _apply_industrial_theme(self) -> None:
        """Apply dark industrial test-bench stylesheet."""
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #0f1217;
                color: #d8dee9;
                font-family: "Inter", "Segoe UI", "DejaVu Sans", sans-serif;
                font-size: 12px;
            }
            QFrame#TopBar, QFrame#ControlPanel, QFrame#PlotContainer {
                background-color: #161b22;
                border: 1px solid #28303d;
                border-radius: 6px;
            }
            QFrame#HudCard {
                background-color: #11161d;
                border: 1px solid #2b3545;
                border-left: 4px solid #00e5ff;
                border-radius: 5px;
            }
            QFrame#SubPanel {
                background-color: #11161d;
                border: 1px solid #262f3d;
                border-radius: 5px;
            }
            QLabel#HudCardTitle {
                color: #8b96a8;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QLabel#HudCardUnit {
                color: #6b778d;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#SectionHeader {
                color: #9aa5b4;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }
            QLabel#TopBarLabel {
                color: #8b96a8;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#StatValue {
                color: #eceff4;
                font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
                font-size: 12px;
                font-weight: 700;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0d1117;
                border: 1px solid #303947;
                border-radius: 4px;
                padding: 4px 8px;
                color: #e5e9f0;
                min-height: 22px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #00e5ff;
            }
            QProgressBar {
                background-color: #0b0e13;
                border: 1px solid #2c3545;
                border-radius: 4px;
                text-align: center;
                color: #e5e9f0;
                font-weight: bold;
                min-height: 16px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00b0ff, stop:1 #00e676
                );
                border-radius: 3px;
            }
            QPushButton#ActionButton, QPushButton#CenterTraceButton {
                background-color: #1c2430;
                color: #00e5ff;
                border: 1px solid #2f3e52;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton#ActionButton:hover, QPushButton#CenterTraceButton:hover {
                background-color: #253142;
                border-color: #00e5ff;
            }
            QPushButton#CollapsibleButton {
                background-color: #151c25;
                color: #c8d1de;
                border: 1px solid #2b3545;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 11px;
                font-weight: 700;
                text-align: left;
            }
            QPushButton#CollapsibleButton:hover {
                border-color: #00e5ff;
                color: #ffffff;
            }
            QPushButton#StartButton {
                background-color: #00c853;
                color: #08120c;
                border: 1px solid #00e676;
                border-radius: 5px;
                padding: 9px 14px;
                font-size: 13px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QPushButton#StartButton:hover {
                background-color: #00e676;
            }
            QPushButton#StartButton:disabled {
                background-color: #1e3326;
                color: #556b5d;
                border-color: #294233;
            }
            QPushButton#StopButton {
                background-color: #d50000;
                color: #ffffff;
                border: 1px solid #ff1744;
                border-radius: 5px;
                padding: 9px 14px;
                font-size: 13px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QPushButton#StopButton:hover {
                background-color: #ff1744;
            }
            QPushButton#StopButton:disabled {
                background-color: #381a1e;
                color: #6e4f53;
                border-color: #4a2328;
            }
            QTableWidget {
                background-color: #0d1117;
                border: 1px solid #262f3d;
                gridline-color: #1f2631;
                color: #d8dee9;
                font-family: "JetBrains Mono", "DejaVu Sans Mono", monospace;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #161b22;
                color: #8b96a8;
                border: 1px solid #262f3d;
                padding: 3px 4px;
                font-size: 10px;
                font-weight: 700;
            }
            QSplitter::handle {
                background-color: #1f2631;
                width: 4px;
            }
            """
        )

    def _build_ui(self) -> None:
        """Construct the Top Bar and 3-column Splitter layout."""
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # 1. Top Configuration Bar
        self.top_bar = QFrame()
        self.top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.setSpacing(10)

        # Test Name (QLineEdit)
        self.lbl_test_name = QLabel()
        self.lbl_test_name.setObjectName("TopBarLabel")
        self.txt_test_name = QLineEdit()
        self.txt_test_name.setMinimumWidth(165)
        self.txt_test_name.setText("Caliber-Test-01")

        # Caliber Preset
        self.lbl_caliber = QLabel()
        self.lbl_caliber.setObjectName("TopBarLabel")
        self.cmb_caliber = QComboBox()
        self.cmb_caliber.addItems(list(CALIBER_PRESETS.keys()))
        self.cmb_caliber.setCurrentText("ETA 2824-2 / SW200")
        self.cmb_caliber.currentTextChanged.connect(self._on_caliber_changed)

        # BPH
        self.lbl_bph = QLabel()
        self.lbl_bph.setObjectName("TopBarLabel")
        self.cmb_bph = QComboBox()
        for bph in (18000, 21600, 25200, 28800, 36000):
            self.cmb_bph.addItem(str(bph), bph)
        self.cmb_bph.setCurrentText("28800")

        # Lift Angle
        self.lbl_lift = QLabel()
        self.lbl_lift.setObjectName("TopBarLabel")
        self.spn_lift = QDoubleSpinBox()
        self.spn_lift.setRange(30.0, 70.0)
        self.spn_lift.setSingleStep(0.5)
        self.spn_lift.setDecimals(1)
        self.spn_lift.setSuffix("°")
        self.spn_lift.setValue(52.0)

        # Position
        self.lbl_position = QLabel()
        self.lbl_position.setObjectName("TopBarLabel")
        self.cmb_position = QComboBox()
        self.cmb_position.addItems(["DU", "DD", "CU", "CD", "CL", "CR"])

        # IN Device
        self.lbl_device = QLabel()
        self.lbl_device.setObjectName("TopBarLabel")
        self.cmb_device = QComboBox()
        self._populate_audio_devices()

        # Sample Rate
        self.lbl_srate = QLabel()
        self.lbl_srate.setObjectName("TopBarLabel")
        self.cmb_srate = QComboBox()
        self.cmb_srate.addItem("48 kHz", 48000)
        self.cmb_srate.addItem("96 kHz", 96000)
        self.cmb_srate.setCurrentIndex(1)

        # Stabilization time
        self.lbl_stab = QLabel()
        self.lbl_stab.setObjectName("TopBarLabel")
        self.spn_stab = QSpinBox()
        self.spn_stab.setRange(0, 30)
        self.spn_stab.setValue(2)
        self.spn_stab.setSuffix(" s")

        # Test duration
        self.lbl_duration = QLabel()
        self.lbl_duration.setObjectName("TopBarLabel")
        self.spn_duration = QSpinBox()
        self.spn_duration.setRange(5, 600)
        self.spn_duration.setValue(30)
        self.spn_duration.setSuffix(" s")

        # Language selector (EN / ES)
        self.lbl_lang = QLabel()
        self.lbl_lang.setObjectName("TopBarLabel")
        self.cmb_lang = QComboBox()
        self.cmb_lang.addItem("EN", "en")
        self.cmb_lang.addItem("ES", "es")
        idx_lang = 0 if self.translator.locale == "en" else 1
        self.cmb_lang.setCurrentIndex(idx_lang)
        self.cmb_lang.currentIndexChanged.connect(self._on_language_changed)

        for lbl, widget in (
            (self.lbl_test_name, self.txt_test_name),
            (self.lbl_caliber, self.cmb_caliber),
            (self.lbl_bph, self.cmb_bph),
            (self.lbl_lift, self.spn_lift),
            (self.lbl_position, self.cmb_position),
            (self.lbl_device, self.cmb_device),
            (self.lbl_srate, self.cmb_srate),
            (self.lbl_stab, self.spn_stab),
            (self.lbl_duration, self.spn_duration),
            (self.lbl_lang, self.cmb_lang),
        ):
            pair = QVBoxLayout()
            pair.setSpacing(2)
            pair.addWidget(lbl)
            pair.addWidget(widget)
            top_layout.addLayout(pair)

        top_layout.addStretch()
        root_layout.addWidget(self.top_bar)

        # 2. Main Content Area (QHBoxLayout with 3-column QSplitter)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        # --- Left Column: Control Panel & HUD ---
        self.left_panel = QFrame()
        self.left_panel.setObjectName("ControlPanel")
        self.left_panel.setMinimumWidth(300)
        self.left_panel.setMaximumWidth(390)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        self.card_rate = HudCard("RATE", "s/d", accent_color="#00e5ff")
        self.card_beat_error = HudCard("BEAT ERROR", "ms", accent_color="#00e676")
        self.card_amplitude = HudCard("AMPLITUDE", "deg (°)", accent_color="#ffab00")

        left_layout.addWidget(self.card_rate)
        left_layout.addWidget(self.card_beat_error)
        left_layout.addWidget(self.card_amplitude)

        # Signal Quality / SNR Box
        self.snr_panel = QFrame()
        self.snr_panel.setObjectName("SubPanel")
        snr_layout = QVBoxLayout(self.snr_panel)
        snr_layout.setContentsMargins(10, 6, 10, 6)
        snr_layout.setSpacing(4)

        snr_header = QHBoxLayout()
        self.lbl_signal_quality = QLabel()
        self.lbl_signal_quality.setObjectName("SectionHeader")
        self.lbl_snr_value = QLabel("-- dB")
        self.lbl_snr_value.setObjectName("StatValue")
        snr_header.addWidget(self.lbl_signal_quality)
        snr_header.addStretch()
        snr_header.addWidget(self.lbl_snr_value)

        self.prg_snr = QProgressBar()
        self.prg_snr.setRange(0, 100)
        self.prg_snr.setValue(0)

        snr_layout.addLayout(snr_header)
        snr_layout.addWidget(self.prg_snr)
        left_layout.addWidget(self.snr_panel)

        # Statistical Panel (Max, Min, Delta, Mean, Elapsed Time)
        self.stats_panel = QFrame()
        self.stats_panel.setObjectName("SubPanel")
        stats_layout = QVBoxLayout(self.stats_panel)
        stats_layout.setContentsMargins(10, 6, 10, 6)
        stats_layout.setSpacing(4)

        self.lbl_stats_title = QLabel()
        self.lbl_stats_title.setObjectName("SectionHeader")
        stats_layout.addWidget(self.lbl_stats_title)

        stats_grid = QGridLayout()
        stats_grid.setHorizontalSpacing(8)
        stats_grid.setVerticalSpacing(3)

        self.lbl_stat_max_key = QLabel()
        self.lbl_stat_max_val = QLabel("--- s/d")
        self.lbl_stat_max_val.setObjectName("StatValue")
        self.lbl_stat_max_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.lbl_stat_min_key = QLabel()
        self.lbl_stat_min_val = QLabel("--- s/d")
        self.lbl_stat_min_val.setObjectName("StatValue")
        self.lbl_stat_min_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.lbl_stat_delta_key = QLabel()
        self.lbl_stat_delta_val = QLabel("--- s/d")
        self.lbl_stat_delta_val.setObjectName("StatValue")
        self.lbl_stat_delta_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.lbl_stat_mean_key = QLabel()
        self.lbl_stat_mean_val = QLabel("--- s/d")
        self.lbl_stat_mean_val.setObjectName("StatValue")
        self.lbl_stat_mean_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.lbl_stat_elapsed_key = QLabel()
        self.lbl_stat_elapsed_val = QLabel("00:00.0")
        self.lbl_stat_elapsed_val.setObjectName("StatValue")
        self.lbl_stat_elapsed_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        for row_idx, (k_lbl, v_lbl) in enumerate(
            (
                (self.lbl_stat_max_key, self.lbl_stat_max_val),
                (self.lbl_stat_min_key, self.lbl_stat_min_val),
                (self.lbl_stat_delta_key, self.lbl_stat_delta_val),
                (self.lbl_stat_mean_key, self.lbl_stat_mean_val),
                (self.lbl_stat_elapsed_key, self.lbl_stat_elapsed_val),
            )
        ):
            stats_grid.addWidget(k_lbl, row_idx, 0)
            stats_grid.addWidget(v_lbl, row_idx, 1)

        stats_layout.addLayout(stats_grid)
        left_layout.addWidget(self.stats_panel)

        # Positional Test Management: "Guardar Posición" & Collapsible "Estadísticas Globales (Delta Posicional)"
        self.btn_save_position = QPushButton()
        self.btn_save_position.setObjectName("ActionButton")
        self.btn_save_position.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_position.clicked.connect(self.save_current_position)
        left_layout.addWidget(self.btn_save_position)

        self.btn_global_stats = QPushButton()
        self.btn_global_stats.setObjectName("CollapsibleButton")
        self.btn_global_stats.setCheckable(True)
        self.btn_global_stats.setChecked(True)
        self.btn_global_stats.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_global_stats.toggled.connect(self._on_toggle_global_stats)
        left_layout.addWidget(self.btn_global_stats)

        self.global_stats_container = QFrame()
        self.global_stats_container.setObjectName("SubPanel")
        gstats_layout = QVBoxLayout(self.global_stats_container)
        gstats_layout.setContentsMargins(8, 6, 8, 6)
        gstats_layout.setSpacing(4)

        self.tbl_global_stats = QTableWidget(0, 4)
        self.tbl_global_stats.setMinimumHeight(95)
        self.tbl_global_stats.setMaximumHeight(130)
        self.tbl_global_stats.verticalHeader().setVisible(False)
        self.tbl_global_stats.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_global_stats.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self.tbl_global_stats.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        gstats_layout.addWidget(self.tbl_global_stats)

        summary_grid = QGridLayout()
        summary_grid.setHorizontalSpacing(8)
        summary_grid.setVerticalSpacing(2)
        self.lbl_global_mean_key = QLabel()
        self.lbl_global_mean_val = QLabel("--- s/d")
        self.lbl_global_mean_val.setObjectName("StatValue")
        self.lbl_global_mean_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.lbl_global_delta_key = QLabel()
        self.lbl_global_delta_val = QLabel("--- s/d")
        self.lbl_global_delta_val.setObjectName("StatValue")
        self.lbl_global_delta_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        summary_grid.addWidget(self.lbl_global_mean_key, 0, 0)
        summary_grid.addWidget(self.lbl_global_mean_val, 0, 1)
        summary_grid.addWidget(self.lbl_global_delta_key, 1, 0)
        summary_grid.addWidget(self.lbl_global_delta_val, 1, 1)
        gstats_layout.addLayout(summary_grid)

        left_layout.addWidget(self.global_stats_container)
        left_layout.addStretch()

        # Start / Stop Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.btn_start = QPushButton()
        self.btn_start.setObjectName("StartButton")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_capture)

        self.btn_stop = QPushButton()
        self.btn_stop.setObjectName("StopButton")
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_capture)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)

        # --- Center Column: Paper Tape Trace (Horizontal Time on X, Deviation/Rate on Y) ---
        self.center_container = QFrame()
        self.center_container.setObjectName("PlotContainer")
        center_layout = QVBoxLayout(self.center_container)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(6)

        self.paper_tape_plot = pg.PlotWidget()
        self.paper_tape_plot.showGrid(x=True, y=True, alpha=0.25)
        self.paper_tape_plot.setXRange(0.0, 30.0)
        self.paper_tape_plot.setYRange(-10.0, 12.0)

        # Zero-deviation horizontal reference line
        self.tape_zero_line = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen(color="#3b4659", width=1.2, style=Qt.PenStyle.DashLine),
        )
        self.paper_tape_plot.addItem(self.tape_zero_line)

        # Static horizontal translucent green COSC limit lines (-4 s/d and +6 s/d) on Y-axis
        cosc_pen = pg.mkPen(color=(0, 230, 118, 135), width=1.8, style=Qt.PenStyle.DashLine)
        self.cosc_lower_line = pg.InfiniteLine(
            pos=self.COSC_MIN_SPD,
            angle=0,
            pen=cosc_pen,
            label="COSC -4 s/d",
            labelOpts={"position": 0.08, "color": (0, 230, 118, 200)},
        )
        self.cosc_upper_line = pg.InfiniteLine(
            pos=self.COSC_MAX_SPD,
            angle=0,
            pen=cosc_pen,
            label="COSC +6 s/d",
            labelOpts={"position": 0.08, "color": (0, 230, 118, 200)},
        )
        self.paper_tape_plot.addItem(self.cosc_lower_line)
        self.paper_tape_plot.addItem(self.cosc_upper_line)

        # Two parallel traces (even/odd beats) using triangle symbols ('t' and 't1')
        self.tape_curve_even = self.paper_tape_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#00e5ff", width=1.5),
            symbol="t",
            symbolSize=7,
            symbolBrush="#00e5ff",
            symbolPen=None,
            name="Tick (Even)",
        )
        self.tape_curve_odd = self.paper_tape_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#ffab00", width=1.5),
            symbol="t1",
            symbolSize=7,
            symbolBrush="#ffab00",
            symbolPen=None,
            name="Tock (Odd)",
        )
        center_layout.addWidget(self.paper_tape_plot, stretch=1)

        # Footer bar right beneath Paper Tape Trace with the "Center Trace" button
        tape_footer_layout = QHBoxLayout()
        tape_footer_layout.setContentsMargins(2, 0, 2, 0)
        tape_footer_layout.addStretch()
        self.btn_center_trace = QPushButton()
        self.btn_center_trace.setObjectName("CenterTraceButton")
        self.btn_center_trace.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_center_trace.clicked.connect(self.center_paper_tape_trace)
        tape_footer_layout.addWidget(self.btn_center_trace)
        center_layout.addLayout(tape_footer_layout)

        # --- Right Column: Real-Time Oscilloscope (PlotWidget) ---
        self.right_container = QFrame()
        self.right_container.setObjectName("PlotContainer")
        right_layout = QVBoxLayout(self.right_container)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self.oscilloscope_plot = pg.PlotWidget()
        self.oscilloscope_plot.showGrid(x=True, y=True, alpha=0.25)
        self.oscilloscope_plot.addLegend(offset=(10, 10))

        self.osc_wave_curve = self.oscilloscope_plot.plot(
            [],
            [],
            pen=pg.mkPen(color="#00e5ff", width=1.6),
        )

        # Vertical markers for P1 (Unlock), P2 (Impulse), P3 (Drop)
        self.line_p1 = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(color="#00e676", width=1.8, style=Qt.PenStyle.DashLine),
            label="P1",
            labelOpts={"position": 0.9, "color": "#00e676"},
        )
        self.line_p2 = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(color="#ffab00", width=1.8, style=Qt.PenStyle.DashLine),
            label="P2",
            labelOpts={"position": 0.82, "color": "#ffab00"},
        )
        self.line_p3 = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen(color="#ff1744", width=1.8, style=Qt.PenStyle.DashLine),
            label="P3",
            labelOpts={"position": 0.9, "color": "#ff1744"},
        )
        self.oscilloscope_plot.addItem(self.line_p1)
        self.oscilloscope_plot.addItem(self.line_p2)
        self.oscilloscope_plot.addItem(self.line_p3)

        self.osc_impulse_markers = pg.ScatterPlotItem(size=10)
        self.oscilloscope_plot.addItem(self.osc_impulse_markers)

        right_layout.addWidget(self.oscilloscope_plot)

        # Add all 3 columns to the splitter
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.center_container)
        self.splitter.addWidget(self.right_container)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 1)
        self.splitter.setSizes([330, 515, 515])

        content_layout.addWidget(self.splitter)
        root_layout.addLayout(content_layout, stretch=1)

    def _populate_audio_devices(self) -> None:
        """Populate input audio device selector with synthetic generator and hardware inputs."""
        self.cmb_device.clear()
        self.cmb_device.addItem(self.translator.tr("synthetic_device"), "synthetic")
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if isinstance(dev, dict) and dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Device {idx}")
                    self.cmb_device.addItem(f"[{idx}] {name}", idx)
        except Exception:
            pass

    def _on_caliber_changed(self, caliber_name: str) -> None:
        """Update BPH and Lift Angle automatically when a caliber preset is selected."""
        if caliber_name in CALIBER_PRESETS and caliber_name != "Custom":
            bph, lift_angle = CALIBER_PRESETS[caliber_name]
            self.cmb_bph.setCurrentText(str(bph))
            self.spn_lift.setValue(lift_angle)

    def _on_language_changed(self, index: int) -> None:
        """Switch active language in Translator and refresh all UI texts."""
        lang_code = self.cmb_lang.itemData(index)
        if lang_code:
            self.translator.set_locale(str(lang_code))
            self.retranslate_ui()

    def _on_toggle_global_stats(self, checked: bool) -> None:
        """Show or hide the collapsible positional statistics section."""
        self.global_stats_container.setVisible(checked)
        arrow = "▼" if checked else "▶"
        self.btn_global_stats.setText(f"{arrow} {self.translator.tr('btn_global_stats')}")

    def center_paper_tape_trace(self) -> None:
        """Auto-adjust the Paper Tape Y-axis range to center on current trace values."""
        all_y = self._tape_rate_even + self._tape_rate_odd
        if not all_y:
            self.paper_tape_plot.setYRange(-10.0, 12.0)
            return

        y_min = float(min(all_y))
        y_max = float(max(all_y))
        y_center = 0.5 * (y_min + y_max)
        half_span = max(3.5, 0.75 * (y_max - y_min) + 2.0)
        self.paper_tape_plot.setYRange(y_center - half_span, y_center + half_span)

    def save_current_position(self) -> None:
        """Save current telemetry linked to the selected position (DU, DD, CU, CD, CL, CR)."""
        pos = self.cmb_position.currentText()
        if self._last_telemetry is not None:
            record = self._last_telemetry
        else:
            base_rate, base_be, base_amp = POSITION_VARIATION.get(pos, (+2.0, 0.4, 270.0))
            record = TelemetryResult(
                rate_spd=base_rate,
                beat_error_ms=base_be,
                amplitude_deg=base_amp,
                confidence=0.97,
            )
        self._saved_positions[pos] = record
        self._refresh_global_stats_table()

    def _refresh_global_stats_table(self) -> None:
        """Update the positional summary table and compute multi-position mean & max delta."""
        positions_order = [p for p in ("DU", "DD", "CU", "CD", "CL", "CR") if p in self._saved_positions]
        self.tbl_global_stats.setRowCount(len(positions_order))

        rates: list[float] = []
        for row_idx, pos in enumerate(positions_order):
            rec = self._saved_positions[pos]
            rates.append(rec.rate_spd)
            vals = (
                pos,
                f"{rec.rate_spd:+.1f}",
                f"{rec.beat_error_ms:.2f}",
                f"{rec.amplitude_deg:.0f}°",
            )
            for col_idx, text in enumerate(vals):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.tbl_global_stats.setItem(row_idx, col_idx, item)

        if rates:
            mean_rate = float(np.mean(rates))
            max_delta = float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0
            self.lbl_global_mean_val.setText(f"{mean_rate:+.2f} s/d ({len(rates)}P)")
            self.lbl_global_delta_val.setText(f"{max_delta:.2f} s/d")
        else:
            self.lbl_global_mean_val.setText("--- s/d")
            self.lbl_global_delta_val.setText("--- s/d")

    def retranslate_ui(self) -> None:
        """Apply localized strings from `self.translator` to all widgets."""
        tr = self.translator.tr
        self.setWindowTitle(tr("app_title"))

        self.lbl_test_name.setText(tr("test_name"))
        self.txt_test_name.setPlaceholderText(tr("test_name_placeholder"))
        self.lbl_caliber.setText(tr("caliber"))
        self.lbl_bph.setText(tr("bph"))
        self.lbl_lift.setText(tr("lift_angle"))
        self.lbl_position.setText(tr("position"))
        self.lbl_device.setText(tr("in_device"))
        self.lbl_srate.setText(tr("sample_rate"))
        self.lbl_stab.setText(tr("stab_time"))
        self.lbl_duration.setText(tr("test_duration"))
        self.lbl_lang.setText(tr("language"))

        if self.cmb_device.count() > 0:
            self.cmb_device.setItemText(0, tr("synthetic_device"))

        self.card_rate.set_title(tr("rate"))
        self.card_beat_error.set_title(tr("beat_error"))
        self.card_amplitude.set_title(tr("amplitude"))

        self.lbl_signal_quality.setText(tr("signal_quality").upper())
        self.lbl_stats_title.setText(tr("statistics").upper())

        self.lbl_stat_max_key.setText(f"{tr('stat_max')}:")
        self.lbl_stat_min_key.setText(f"{tr('stat_min')}:")
        self.lbl_stat_delta_key.setText(f"{tr('stat_delta')}:")
        self.lbl_stat_mean_key.setText(f"{tr('stat_mean')}:")
        self.lbl_stat_elapsed_key.setText(f"{tr('stat_elapsed')}:")

        self.btn_save_position.setText(tr("btn_save_position"))
        arrow = "▼" if self.btn_global_stats.isChecked() else "▶"
        self.btn_global_stats.setText(f"{arrow} {tr('btn_global_stats')}")
        self.tbl_global_stats.setHorizontalHeaderLabels(
            [tr("col_pos"), tr("col_rate"), tr("col_be"), tr("col_amp")]
        )
        self.lbl_global_mean_key.setText(f"{tr('global_mean')}:")
        self.lbl_global_delta_key.setText(f"{tr('global_max_delta')}:")

        self.btn_center_trace.setText(tr("btn_center_trace"))
        self.btn_start.setText(tr("btn_start"))
        self.btn_stop.setText(tr("btn_stop"))

        self.paper_tape_plot.setTitle(tr("paper_tape_title"), color="#c8d1de", size="11pt")
        self.paper_tape_plot.setLabel("bottom", tr("paper_tape_x"))
        self.paper_tape_plot.setLabel("left", tr("paper_tape_y"))

        self.oscilloscope_plot.setTitle(tr("oscilloscope_title"), color="#c8d1de", size="11pt")
        self.oscilloscope_plot.setLabel("bottom", tr("oscilloscope_x"))
        self.oscilloscope_plot.setLabel("left", tr("oscilloscope_y"))

    def current_watch_config(self) -> WatchConfig:
        """Build a `WatchConfig` instance from the active Top Bar controls."""
        bph = int(self.cmb_bph.currentData() or int(self.cmb_bph.currentText()))
        lift_angle = float(self.spn_lift.value())
        sample_rate = int(self.cmb_srate.currentData() or 96000)
        return WatchConfig(bph=bph, lift_angle=lift_angle, sample_rate=sample_rate)

    def start_capture(self) -> None:
        """Start the synthetic DSP telemetry worker thread and reset plots."""
        if self.worker is not None and self.worker.isRunning():
            return

        self._rate_history.clear()
        self._tape_times_even.clear()
        self._tape_rate_even.clear()
        self._tape_times_odd.clear()
        self._tape_rate_odd.clear()

        self.tape_curve_even.setData([], [])
        self.tape_curve_odd.setData([], [])
        self.osc_wave_curve.setData([], [])
        self.osc_impulse_markers.setData([], [])

        duration_s = float(self.spn_duration.value())
        stab_s = float(self.spn_stab.value())
        self.paper_tape_plot.setXRange(0.0, max(10.0, duration_s))

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_top_controls_enabled(False)

        config = self.current_watch_config()
        position = self.cmb_position.currentText()

        self.worker = SyntheticTelemetryWorker(
            config=config,
            position=position,
            stabilization_s=stab_s,
            duration_s=duration_s,
            parent=self,
        )
        self.worker.telemetry_updated.connect(self._on_telemetry_updated)
        self.worker.session_finished.connect(self._on_session_finished)
        self.worker.start()

    def stop_capture(self) -> None:
        """Stop the running background telemetry worker thread."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self._on_session_finished()

    def _on_session_finished(self) -> None:
        """Restore button and selector states once capture finishes."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_top_controls_enabled(True)

    def _set_top_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable configuration controls during active measurement."""
        for widget in (
            self.txt_test_name,
            self.cmb_caliber,
            self.cmb_bph,
            self.spn_lift,
            self.cmb_position,
            self.cmb_device,
            self.cmb_srate,
            self.spn_stab,
            self.spn_duration,
        ):
            widget.setEnabled(enabled)

    def _on_telemetry_updated(
        self,
        telemetry: TelemetryResult,
        beats: list[BeatEvent],
        elapsed_s: float,
        snr_db: float,
    ) -> None:
        """Update HUD cards, statistics, horizontal Witschi paper tape, and P1/P2/P3 oscilloscope."""
        self._last_telemetry = telemetry
        stab_s = float(self.spn_stab.value())

        # Format elapsed time
        mins = int(elapsed_s) // 60
        secs = elapsed_s % 60.0
        self.lbl_stat_elapsed_val.setText(f"{mins:02d}:{secs:04.1f}")

        # Update Signal Quality / SNR bar
        quality_pct = int(np.clip((snr_db / 50.0) * 100.0 * telemetry.confidence, 0.0, 100.0))
        self.prg_snr.setValue(quality_pct)
        self.lbl_snr_value.setText(f"{snr_db:.1f} dB ({int(telemetry.confidence * 100)}%)")

        # Update HUD Cards
        sign_str = "+" if telemetry.rate_spd >= 0 else ""
        self.card_rate.set_value(f"{sign_str}{telemetry.rate_spd:.1f}")
        self.card_beat_error.set_value(f"{telemetry.beat_error_ms:.2f}")
        self.card_amplitude.set_value(f"{telemetry.amplitude_deg:.0f}°")

        # Update Statistics & Horizontal Paper Tape after stabilization period
        if elapsed_s >= stab_s:
            t_meas = elapsed_s - stab_s
            self._rate_history.append(telemetry.rate_spd)

            r_max = max(self._rate_history)
            r_min = min(self._rate_history)
            r_delta = r_max - r_min
            r_mean = float(np.mean(self._rate_history))

            self.lbl_stat_max_val.setText(f"{r_max:+.2f} s/d")
            self.lbl_stat_min_val.setText(f"{r_min:+.2f} s/d")
            self.lbl_stat_delta_val.setText(f"{r_delta:.2f} s/d")
            self.lbl_stat_mean_val.setText(f"{r_mean:+.2f} s/d")

            # Horizontal Paper Tape: X = Elapsed Time (s), Y = Deviation/Rate (s/d)
            # Separate even/odd tracks visually by beat error asymmetry
            half_be_offset = 0.5 * telemetry.beat_error_ms
            self._tape_times_even.append(t_meas)
            self._tape_rate_even.append(telemetry.rate_spd - half_be_offset)
            self._tape_times_odd.append(t_meas)
            self._tape_rate_odd.append(telemetry.rate_spd + half_be_offset)

            self.tape_curve_even.setData(self._tape_times_even, self._tape_rate_even)
            self.tape_curve_odd.setData(self._tape_times_odd, self._tape_rate_odd)

        # Update Right Column Oscilloscope with the latest BeatEvent
        if beats:
            latest_beat = beats[-1]
            wave = latest_beat.raw_waveform
            fs = self.current_watch_config().sample_rate
            if len(wave) > 0:
                peak_abs = float(np.max(np.abs(wave))) or 1.0
                norm_wave = wave / peak_abs
                t_ms = (np.arange(len(wave), dtype=np.float64) / fs) * 1000.0

                dt_p2 = (latest_beat.timestamp_p2 - latest_beat.timestamp_p1) * 1000.0
                dt_p3 = (latest_beat.timestamp_p3 - latest_beat.timestamp_p1) * 1000.0

                p3_idx = int(np.argmax(np.abs(norm_wave)))
                t_p3_ms = float(t_ms[p3_idx])
                t_p1_ms = max(0.2, t_p3_ms - dt_p3)
                t_p2_ms = np.clip(t_p1_ms + dt_p2, t_p1_ms, t_p3_ms)

                self.osc_wave_curve.setData(t_ms, norm_wave)
                self.line_p1.setPos(t_p1_ms)
                self.line_p2.setPos(t_p2_ms)
                self.line_p3.setPos(t_p3_ms)

                p1_idx = int(np.clip(round((t_p1_ms / 1000.0) * fs), 0, len(norm_wave) - 1))
                p2_idx = int(np.clip(round((t_p2_ms / 1000.0) * fs), 0, len(norm_wave) - 1))

                spots = [
                    {
                        "pos": (t_p1_ms, float(norm_wave[p1_idx])),
                        "brush": pg.mkBrush("#00e676"),
                        "pen": pg.mkPen("#ffffff", width=1),
                    },
                    {
                        "pos": (t_p2_ms, float(norm_wave[p2_idx])),
                        "brush": pg.mkBrush("#ffab00"),
                        "pen": pg.mkPen("#ffffff", width=1),
                    },
                    {
                        "pos": (t_p3_ms, float(norm_wave[p3_idx])),
                        "brush": pg.mkBrush("#ff1744"),
                        "pen": pg.mkPen("#ffffff", width=1),
                    },
                ]
                self.osc_impulse_markers.setData(spots)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Ensure background worker thread is cleanly stopped before window closes."""
        self.stop_capture()
        super().closeEvent(event)

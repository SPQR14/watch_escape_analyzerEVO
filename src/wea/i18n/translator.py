"""Lightweight bilingual translator (EN/ES) for horological terminology and UI labels."""

from __future__ import annotations


class Translator:
    """Provides localized strings in English ('en') and Spanish ('es') for horological metrics."""

    DICTIONARIES: dict[str, dict[str, str]] = {
        "en": {
            "app_title": "Watch Escape Analyzer EVO — Horological DSP Test Bench",
            "rate": "Rate",
            "beat_error": "Beat Error",
            "amplitude": "Amplitude",
            "lift_angle": "Lift Angle",
            "bph": "BPH",
            "confidence": "Confidence",
            "unlock_impulse": "P1 Unlock",
            "impulse": "P2 Impulse",
            "drop_impulse": "P3 Drop",
            "test_name": "Test Name",
            "test_name_placeholder": "e.g. Movement #01 - Service",
            "caliber": "Caliber",
            "position": "Position",
            "in_device": "IN Device",
            "sample_rate": "Sample Rate",
            "stab_time": "Stabilization",
            "test_duration": "Test Duration",
            "language": "Lang",
            "signal_quality": "Signal Quality / SNR",
            "statistics": "Telemetry Statistics",
            "stat_max": "Max",
            "stat_min": "Min",
            "stat_delta": "Delta",
            "stat_mean": "Mean",
            "stat_elapsed": "Elapsed Time",
            "btn_start": "START",
            "btn_stop": "STOP",
            "btn_center_trace": "Center Trace",
            "btn_save_position": "Save Position",
            "btn_global_stats": "Global Statistics (Positional Delta)",
            "global_stats_summary": "Positional Summary (2–6 Pos)",
            "col_pos": "Pos",
            "col_rate": "Rate (s/d)",
            "col_be": "BE (ms)",
            "col_amp": "Amp (°)",
            "global_mean": "Global Mean",
            "global_max_delta": "Positional Delta",
            "paper_tape_title": "Paper Tape Trace (Horizontal Rate / Deviation)",
            "paper_tape_x": "Elapsed Time (s)",
            "paper_tape_y": "Deviation / Rate (s/d)",
            "oscilloscope_title": "Real-Time Escapement Oscilloscope (P1 / P2 / P3)",
            "oscilloscope_x": "Time Window (ms)",
            "oscilloscope_y": "Normalized Acoustic Amplitude",
            "synthetic_device": "Synthetic Ground-Truth Generator",
            "status_idle": "READY",
            "status_stabilizing": "STABILIZING...",
            "status_running": "MEASURING",
            "status_completed": "COMPLETED",
        },
        "es": {
            "app_title": "Watch Escape Analyzer EVO — Banco de Pruebas DSP Horológico",
            "rate": "Marcha",
            "beat_error": "Error de Batido",
            "amplitude": "Amplitud",
            "lift_angle": "Ángulo de Alzamiento",
            "bph": "A/h",
            "confidence": "Confianza",
            "unlock_impulse": "P1 Desbloqueo",
            "impulse": "P2 Impulso",
            "drop_impulse": "P3 Caída",
            "test_name": "Nombre de la prueba",
            "test_name_placeholder": "ej. Calibre #01 - Servicio",
            "caliber": "Calibre",
            "position": "Posición",
            "in_device": "Dispositivo IN",
            "sample_rate": "Frec. Muestreo",
            "stab_time": "Estabilización",
            "test_duration": "Duración Test",
            "language": "Idioma",
            "signal_quality": "Calidad de Señal / SNR",
            "statistics": "Estadísticas de Telemetría",
            "stat_max": "Máx",
            "stat_min": "Mín",
            "stat_delta": "Delta",
            "stat_mean": "Media",
            "stat_elapsed": "Tiempo Transcurrido",
            "btn_start": "INICIAR",
            "btn_stop": "DETENER",
            "btn_center_trace": "Centrar Trazo",
            "btn_save_position": "Guardar Posición",
            "btn_global_stats": "Estadísticas Globales (Delta Posicional)",
            "global_stats_summary": "Resumen Posicional (2–6 Pos)",
            "col_pos": "Pos",
            "col_rate": "Marcha (s/d)",
            "col_be": "EB (ms)",
            "col_amp": "Amp (°)",
            "global_mean": "Media Global",
            "global_max_delta": "Delta Posicional",
            "paper_tape_title": "Trazo de Cinta de Papel (Desviación / Marcha Horizontal)",
            "paper_tape_x": "Tiempo Transcurrido (s)",
            "paper_tape_y": "Desviación / Marcha (s/d)",
            "oscilloscope_title": "Osciloscopio de Escape en Tiempo Real (P1 / P2 / P3)",
            "oscilloscope_x": "Ventana Temporal (ms)",
            "oscilloscope_y": "Amplitud Acústica Normalizada",
            "synthetic_device": "Generador Sintético Ground-Truth",
            "status_idle": "LISTO",
            "status_stabilizing": "ESTABILIZANDO...",
            "status_running": "MIDIENDO",
            "status_completed": "COMPLETADO",
        },
    }

    def __init__(self, locale: str = "en") -> None:
        """Initialize the translator with the desired language code ('en' or 'es')."""
        self._locale = "en"
        self.set_locale(locale)

    @property
    def locale(self) -> str:
        """Return the active language code."""
        return self._locale

    def set_locale(self, locale: str) -> None:
        """Change the active language code ('en' or 'es')."""
        normalized = locale.strip().lower()
        if normalized not in self.DICTIONARIES:
            raise ValueError(f"Unsupported locale '{locale}'. Supported: {list(self.DICTIONARIES)}")
        self._locale = normalized

    def tr(self, key: str, default: str | None = None) -> str:
        """Translate a horological key into the current locale."""
        table = self.DICTIONARIES[self._locale]
        if key in table:
            return table[key]
        if default is not None:
            return default
        return key

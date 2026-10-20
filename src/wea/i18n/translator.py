"""Lightweight bilingual translator (EN/ES) for horological terminology."""

from __future__ import annotations


class Translator:
    """Provides localized strings in English ('en') and Spanish ('es') for horological metrics."""

    DICTIONARIES: dict[str, dict[str, str]] = {
        "en": {
            "rate": "Rate",
            "beat_error": "Beat Error",
            "amplitude": "Amplitude",
            "lift_angle": "Lift Angle",
            "bph": "BPH",
            "confidence": "Confidence",
            "unlock_impulse": "P1 Unlock",
            "impulse": "P2 Impulse",
            "drop_impulse": "P3 Drop",
        },
        "es": {
            "rate": "Marcha",
            "beat_error": "Error de Batido",
            "amplitude": "Amplitud",
            "lift_angle": "Ángulo de Alzamiento",
            "bph": "A/h",
            "confidence": "Confianza",
            "unlock_impulse": "P1 Desbloqueo",
            "impulse": "P2 Impulso",
            "drop_impulse": "P3 Caída",
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

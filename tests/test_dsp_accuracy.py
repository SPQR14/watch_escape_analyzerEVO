"""Accuracy validation suite for the Watch Escape Analyzer EVO DSP pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from wea.audio.ring_buffer import AudioRingBuffer
from wea.core.dsp_pipeline import EscapeDSPPipeline
from wea.core.models import WatchConfig
from wea.core.synthetic import SyntheticEscapeGenerator
from wea.i18n.translator import Translator


def test_dsp_pipeline_rate_and_beat_error_accuracy() -> None:
    """Validate that EscapeDSPPipeline recovers Rate (< 0.2 s/d) and Beat Error (< 0.1 ms)."""
    config = WatchConfig(bph=28800, lift_angle=52.0, sample_rate=96000)
    generator = SyntheticEscapeGenerator(config=config, seed=42)
    pipeline = EscapeDSPPipeline(config=config)

    true_rate_spd = 6.5
    true_beat_error_ms = 0.6
    true_amplitude_deg = 275.0

    signal = generator.generate(
        duration_s=5.0,
        true_rate_spd=true_rate_spd,
        true_beat_error_ms=true_beat_error_ms,
        true_amplitude_deg=true_amplitude_deg,
        snr_db=35.0,
    )

    telemetry, beats = pipeline.analyze(signal)

    assert len(beats) >= 30
    assert abs(telemetry.rate_spd - true_rate_spd) < 0.2, (
        f"Expected rate ~{true_rate_spd} s/d, got {telemetry.rate_spd:.4f} s/d"
    )
    assert abs(telemetry.beat_error_ms - true_beat_error_ms) < 0.1, (
        f"Expected beat error ~{true_beat_error_ms} ms, got {telemetry.beat_error_ms:.4f} ms"
    )
    assert abs(telemetry.amplitude_deg - true_amplitude_deg) < 10.0, (
        f"Expected amplitude ~{true_amplitude_deg} deg, got {telemetry.amplitude_deg:.2f} deg"
    )
    assert telemetry.confidence > 0.8


@pytest.mark.parametrize(
    ("true_rate", "true_be", "true_amp"),
    [
        (-4.2, 0.3, 250.0),
        (0.0, 0.0, 285.0),
        (12.0, 1.1, 230.0),
    ],
)
def test_dsp_pipeline_multiple_operating_points(
    true_rate: float, true_be: float, true_amp: float
) -> None:
    """Verify DSP accuracy across slow, nominal, and fast operating points."""
    config = WatchConfig(bph=28800, lift_angle=52.0, sample_rate=96000)
    generator = SyntheticEscapeGenerator(config=config, seed=123)
    pipeline = EscapeDSPPipeline(config=config)

    signal = generator.generate(
        duration_s=4.0,
        true_rate_spd=true_rate,
        true_beat_error_ms=true_be,
        true_amplitude_deg=true_amp,
        snr_db=40.0,
    )

    telemetry, _ = pipeline.analyze(signal)
    assert abs(telemetry.rate_spd - true_rate) < 0.2
    assert abs(telemetry.beat_error_ms - true_be) < 0.1
    assert abs(telemetry.amplitude_deg - true_amp) < 10.0


def test_audio_ring_buffer_thread_safe_operations() -> None:
    """Verify circular wrap-around and latest-window extraction in AudioRingBuffer."""
    rb = AudioRingBuffer(capacity=100, dtype=np.float64)
    assert rb.size == 0

    chunk1 = np.arange(60, dtype=np.float64)
    chunk2 = np.arange(60, 140, dtype=np.float64)
    rb.write(chunk1)
    assert rb.size == 60
    rb.write(chunk2)
    assert rb.size == 100

    latest_50 = rb.read_latest(50)
    np.testing.assert_array_equal(latest_50, np.arange(90, 140, dtype=np.float64))


def test_translator_bilingual_terms() -> None:
    """Verify English and Spanish horological translations."""
    tr = Translator("en")
    assert tr.tr("rate") == "Rate"
    assert tr.tr("beat_error") == "Beat Error"
    assert tr.tr("amplitude") == "Amplitude"
    assert tr.tr("lift_angle") == "Lift Angle"
    assert tr.tr("bph") == "BPH"

    tr.set_locale("es")
    assert tr.tr("rate") == "Marcha"
    assert tr.tr("beat_error") == "Error de Batido"
    assert tr.tr("amplitude") == "Amplitud"
    assert tr.tr("lift_angle") == "Ángulo de Alzamiento"
    assert tr.tr("bph") == "A/h"

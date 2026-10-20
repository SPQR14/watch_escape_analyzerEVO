"""Core horological models, synthetic signal generator, and DSP pipeline."""

from wea.core.dsp_pipeline import EscapeDSPPipeline
from wea.core.models import BeatEvent, TelemetryResult, WatchConfig
from wea.core.synthetic import SyntheticEscapeGenerator

__all__ = [
    "WatchConfig",
    "BeatEvent",
    "TelemetryResult",
    "SyntheticEscapeGenerator",
    "EscapeDSPPipeline",
]

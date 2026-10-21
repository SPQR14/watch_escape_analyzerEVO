"""Core horological models, ring buffer, synthetic signal generator, clock drift, and DSP engines."""

from wea.core.clock_drift import AtomicNTPDiscipline, SampleRateDriftEstimator
from wea.core.dsp_engine import DSPEngine
from wea.core.dsp_pipeline import EscapeDSPPipeline
from wea.core.models import BeatEvent, BeatTelemetry, TelemetryResult, WatchConfig
from wea.core.ring_buffer import AudioRingBuffer
from wea.core.synthetic import SyntheticEscapeGenerator

__all__ = [
    "AtomicNTPDiscipline",
    "AudioRingBuffer",
    "WatchConfig",
    "BeatEvent",
    "TelemetryResult",
    "BeatTelemetry",
    "SyntheticEscapeGenerator",
    "EscapeDSPPipeline",
    "SampleRateDriftEstimator",
    "DSPEngine",
]

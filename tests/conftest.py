"""Shared fixtures: an independent, physically-modelled escapement signal generator.

This generator deliberately does NOT reuse ``wea.core.synthetic`` so the DSP is validated
against the physics instead of against its own conventions (both once shared the same
factor-of-two errors in lift time and beat error, so the tests passed while real watches
read ~half their amplitude).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray


def physical_escapement_signal(
    bph: int = 28800,
    rate_spd: float = 4.0,
    beat_error_ms: float = 0.2,
    amplitude_deg: float = 279.0,
    lift_angle: float = 52.0,
    seconds: float = 6.0,
    fs: int = 48000,
    snr_db: float = 25.0,
    seed: int = 0,
    sound_levels: tuple[float, float, float] = (0.6, 0.35, 1.0),
) -> NDArray[np.float64]:
    """Return a piezo-like escapement recording with known parameters.

    Physics used:
        * Beat period ``P = 3600 / bph / (1 + rate / 86400)`` (fast watch -> shorter period).
        * Beat error convention: beat start times are offset by ``+-d / 2`` alternately, so
          intervals alternate ``P + d`` / ``P - d`` and a timegrapher reports ``d``.
        * Lift time ``t_lift = arcsin(LA / (2 A)) / (pi f_osc)`` with ``f_osc = bph / 7200``:
          unlock (P1) at 0, impulse (P2) at ``0.45 t_lift`` and drop (P3) at ``t_lift``.

    Each sound is a damped two-mode burst (case resonance), plus white noise and 50 Hz hum.
    """
    rng = np.random.default_rng(seed)
    period_s = 3600.0 / bph / (1.0 + rate_spd / 86400.0)
    f_osc = bph / 7200.0
    t_lift = math.asin(lift_angle / (2.0 * amplitude_deg)) / (math.pi * f_osc)

    n_total = int(seconds * fs)
    x = np.zeros(n_total + fs, dtype=np.float64)
    burst_n = int(0.006 * fs)
    t_burst = np.arange(burst_n, dtype=np.float64) / fs
    n_beats = int(seconds / period_s) - 1
    for i in range(n_beats):
        t0 = 0.05 + i * period_s + 0.5 * (beat_error_ms * 1e-3) * (-1) ** i
        for frac, level, f0 in zip((0.0, 0.45, 1.0), sound_levels, (5200.0, 6100.0, 4300.0)):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            burst = (
                level
                * (
                    np.sin(2.0 * np.pi * f0 * t_burst + phase)
                    + 0.5 * np.sin(2.0 * np.pi * 1.7 * f0 * t_burst)
                )
                * np.exp(-t_burst / 0.0004)
            )
            start = int(round((t0 + frac * t_lift) * fs))
            x[start : start + burst_n] += burst

    active = x[x != 0.0]
    sig_rms = float(np.sqrt(np.mean(active**2))) if active.size else 1.0
    t_all = np.arange(len(x), dtype=np.float64) / fs
    noisy = (
        x
        + rng.normal(0.0, sig_rms / (10.0 ** (snr_db / 20.0)), len(x))
        + 0.3 * sig_rms * np.sin(2.0 * np.pi * 50.0 * t_all)
    )
    return noisy[:n_total]


@pytest.fixture
def physical_signal() -> Callable[..., NDArray[np.float64]]:
    """Fixture returning :func:`physical_escapement_signal`."""
    return physical_escapement_signal

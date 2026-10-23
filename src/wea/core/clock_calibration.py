"""Sound-card clock calibration against a reference of known frequency.

A typical audio crystal is off by +-20-50 ppm, i.e. +-1.7-4.3 s/d of systematic rate bias
(1 ppm = 0.0864 s/d). The correction is expressed as ``calibration_ppm`` with the
convention used by ``SampleRateDriftEstimator``: positive when the card samples FAST, so
``effective_fs = nominal_fs * (1 + ppm * 1e-6)``.

Two references are supported:
    * ``measure_ppm_from_tone``: a sinusoid of exactly known frequency (e.g. a 6 Hz tone
      from a GPS-disciplined generator). Complex demodulation at the reference frequency,
      averaging over whole nominal cycles, then a linear fit of the unwrapped phase.
    * ``measure_ppm_from_ticks``: a pulse train of known period (a 6 Hz pulse generator or
      the 1 Hz steps of a quartz watch whose own rate is known). P1 onsets plus the beat
      regression give the period in nominal-clock seconds.

If the card runs fast (true rate ``fs_true > fs_nominal``), a true frequency ``f`` appears as
``f * fs_nominal / fs_true`` and a true period ``P`` as ``P * fs_true / fs_nominal``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wea.core.beat_timing import OnsetBeatTracker, OnsetConfig
from wea.core.rate_regression import fit_rate_be

SECONDS_PER_DAY = 86400.0


@dataclass(slots=True)
class CalibrationResult:
    """Measured sound-card clock error.

    Attributes:
        ppm: Card clock error in ppm (positive = samples fast). Use as ``calibration_ppm``.
        sigma_ppm: 1-sigma statistical uncertainty.
        method: ``"tone"`` or ``"ticks"``.
        duration_s: Length of the analyzed audio.
        n_points: Phase blocks (tone) or ticks (ticks) used.
    """

    ppm: float
    sigma_ppm: float
    method: str
    duration_s: float
    n_points: int

    @property
    def rate_bias_spd(self) -> float:
        """Rate error (s/d) this clock error would cause if left uncorrected."""
        return self.ppm * 1e-6 * SECONDS_PER_DAY


def measure_ppm_from_tone(
    signal: NDArray[np.float64],
    fs_nominal: float,
    tone_hz: float = 6.0,
    cycles_per_block: int = 1,
) -> CalibrationResult:
    """Card clock error from a sinusoidal reference of known frequency ``tone_hz``.

    Args:
        signal: Recording of the reference tone.
        fs_nominal: Nominal sample rate the card was opened at.
        tone_hz: Exact reference frequency.
        cycles_per_block: Nominal cycles averaged per phase sample (removes the 2f term).

    Raises:
        ValueError: If the recording holds fewer than 8 blocks or no tone.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - float(np.mean(x))
    block = int(round(cycles_per_block * fs_nominal / tone_hz))
    n_blocks = len(x) // block
    if n_blocks < 8:
        raise ValueError(f"Need at least 8 x {cycles_per_block} cycles of {tone_hz} Hz")
    t = np.arange(n_blocks * block, dtype=np.float64) / fs_nominal
    z = x[: n_blocks * block] * np.exp(-2j * np.pi * tone_hz * t)
    zb = z.reshape(n_blocks, block).mean(axis=1)
    if float(np.median(np.abs(zb))) <= 1e-12:
        raise ValueError("No tone found at the reference frequency")
    phase = np.unwrap(np.angle(zb))
    tb = (np.arange(n_blocks) + 0.5) * block / fs_nominal
    coef, cov = np.polyfit(tb, phase, 1, cov=True)
    f_apparent = tone_hz + coef[0] / (2.0 * np.pi)
    sigma_f = float(np.sqrt(max(cov[0, 0], 0.0))) / (2.0 * np.pi)
    ppm = (tone_hz / f_apparent - 1.0) * 1e6
    sigma_ppm = tone_hz / f_apparent**2 * sigma_f * 1e6
    return CalibrationResult(
        ppm=float(ppm),
        sigma_ppm=float(sigma_ppm),
        method="tone",
        duration_s=len(x) / fs_nominal,
        n_points=int(n_blocks),
    )


def measure_ppm_from_ticks(
    signal: NDArray[np.float64],
    fs_nominal: float,
    period_s: float = 1.0,
    reference_rate_spd: float = 0.0,
    config: OnsetConfig | None = None,
) -> CalibrationResult:
    """Card clock error from a pulse train of known period.

    Args:
        signal: Recording of the reference ticks.
        fs_nominal: Nominal sample rate the card was opened at.
        period_s: Nominal period of the reference (1.0 for a quartz watch, 1/6 for 6 Hz).
        reference_rate_spd: Known rate of the reference itself (a quartz watch measured to
            gain 0.3 s/d -> 0.3); its true period is ``period_s / (1 + rate / 86400)``.
        config: Onset detector parameters.

    Raises:
        ValueError: If fewer than 8 ticks are found.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    bph = int(round(3600.0 / period_s))
    tracker = OnsetBeatTracker(fs_nominal, bph, config=config)
    onsets = tracker.detect(x)
    fit = fit_rate_be(
        np.array([b.sample for b in onsets]) / fs_nominal,
        3600.0 / bph,
        indices=np.array([b.index for b in onsets], dtype=np.int64),
    )
    if fit is None:
        raise ValueError(f"Found only {len(onsets)} reference ticks (need >= 8)")
    true_period = period_s / (1.0 + reference_rate_spd / SECONDS_PER_DAY)
    # Measured period (nominal-clock seconds) = true period * fs_true / fs_nominal
    ppm = (fit.period_s / true_period - 1.0) * 1e6
    sigma_period = fit.rate_sigma_spd / SECONDS_PER_DAY * fit.period_s
    return CalibrationResult(
        ppm=float(ppm),
        sigma_ppm=float(sigma_period / true_period * 1e6),
        method="ticks",
        duration_s=len(x) / fs_nominal,
        n_points=int(fit.n_used),
    )

"""Rate and beat error by least-squares regression over many beats.

Model (per continuous audio segment ``s``)::

    t = a * n + b_s + c_s * (n mod 2)

* ``n``: integer beat index, ``round((t - t0) / T)`` - tolerates missing beats.
* ``a``: real beat period, shared by all segments.
* ``rate = (T - a) / a * 86400`` s/d (positive = the watch gains).
* ``c``: tick/tock offset; beat error = ``|c|`` (intervals alternate ``a + c`` / ``a - c``).

Outliers are rejected before the final fit: beats whose residual is more than 5 MAD away
from the median residual are dropped and the fit is repeated until the inlier set is stable.

Why regression instead of per-interval rates: one 48 kHz sample of jitter on a single
166.7 ms interval is already 21 us / 0.1667 s * 86400 = 10.8 s/d. A regression over N beats
spanning T seconds has ``sigma_rate ~ sigma_t * sqrt(12) / (T * sqrt(N)) * 86400``; with
40 us onset jitter, 15 s and 90 beats that is ~0.1 s/d.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SECONDS_PER_DAY = 86400.0


@dataclass(slots=True)
class RateFit:
    """Result of :func:`fit_rate_be`.

    Attributes:
        rate_spd: ``(T - a) / a * 86400``; positive = gaining.
        beat_error_ms: ``|c|`` in ms (count-weighted mean over segments).
        period_s: Fitted beat period ``a``.
        tick_tock_offset_s: Signed ``c`` of the largest segment.
        n_used: Beats kept after outlier rejection.
        n_total: Beats offered to the fit.
        span_s: Time covered by the inliers.
        residual_sigma_us: Robust sigma (1.4826 MAD) of the inlier residuals.
        rate_sigma_spd: 1-sigma statistical uncertainty of ``rate_spd``.
    """

    rate_spd: float
    beat_error_ms: float
    period_s: float
    tick_tock_offset_s: float
    n_used: int
    n_total: int
    span_s: float
    residual_sigma_us: float
    rate_sigma_spd: float


def beat_indices(times_s: NDArray[np.float64], nominal_period_s: float) -> NDArray[np.int64]:
    """``n = round((t - t0) / T)``: integer beat index tolerant to missing beats."""
    t = np.asarray(times_s, dtype=np.float64)
    if len(t) == 0:
        return np.empty(0, dtype=np.int64)
    return np.rint((t - t[0]) / nominal_period_s).astype(np.int64)


def _design(n: NDArray[np.int64], seg: NDArray[np.int64]) -> tuple[NDArray[np.float64], list]:
    """Columns: centered n, one intercept and one parity column per segment."""
    labels = list(np.unique(seg))
    cols = [(n - n.mean()).astype(np.float64)]
    for s in labels:
        cols.append((seg == s).astype(np.float64))
    for s in labels:
        parity = (n % 2).astype(np.float64) * (seg == s)
        if np.any(parity) and not np.all(parity[seg == s]):
            cols.append(parity)
        else:
            cols.append(np.zeros(len(n)))  # only one parity present: offset not estimable
    return np.column_stack(cols), labels


def fit_rate_be(
    times_s: NDArray[np.float64] | Iterable[float],
    nominal_period_s: float,
    indices: NDArray[np.int64] | None = None,
    segments: NDArray[np.int64] | None = None,
    mad_k: float = 5.0,
    min_beats: int = 8,
    max_iterations: int = 8,
    min_mad_s: float = 1e-6,
) -> RateFit | None:
    """Robust least squares of ``t = a n + b_s + c_s (n mod 2)``.

    Args:
        times_s: Beat instants in seconds.
        nominal_period_s: Nominal beat period ``T`` (3600 / BPH).
        indices: Integer beat indices; default ``round((t - t0) / T)`` per segment.
        segments: Segment id of each beat (audio discontinuities); default one segment.
        mad_k: Rejection threshold in MADs of the residuals.
        min_beats: Minimum inliers for a fit.
        max_iterations: Rejection/refit iterations.
        min_mad_s: Floor of the MAD so perfectly clean data is not over-rejected.

    Returns:
        :class:`RateFit`, or ``None`` with fewer than ``min_beats`` usable beats.
    """
    t = np.asarray(list(times_s) if not isinstance(times_s, np.ndarray) else times_s, float)
    if len(t) < min_beats:
        return None
    seg = np.zeros(len(t), dtype=np.int64) if segments is None else np.asarray(segments, np.int64)
    if indices is None:
        n = np.empty(len(t), dtype=np.int64)
        for s in np.unique(seg):
            m = seg == s
            n[m] = beat_indices(t[m], nominal_period_s)
    else:
        n = np.asarray(indices, dtype=np.int64)

    t_ref = float(t[0])
    y = t - t_ref
    design, labels = _design(n, seg)
    mask = np.ones(len(t), dtype=bool)
    coef = np.zeros(design.shape[1])
    for _ in range(max_iterations):
        if mask.sum() < min_beats:
            return None
        coef, *_ = np.linalg.lstsq(design[mask], y[mask], rcond=None)
        resid = y - design @ coef
        med = float(np.median(resid[mask]))
        mad = max(float(np.median(np.abs(resid[mask] - med))), min_mad_s)
        new_mask = np.abs(resid - med) <= mad_k * mad
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    if mask.sum() < min_beats:
        return None

    a = float(coef[0])
    if a <= 0.0:
        return None
    resid = (y - design @ coef)[mask]
    n_seg = len(labels)
    parity_coef = coef[1 + n_seg : 1 + 2 * n_seg]
    counts = np.array([np.sum(mask & (seg == s)) for s in labels], dtype=np.float64)
    be_ms = float(np.sum(np.abs(parity_coef) * counts) / max(counts.sum(), 1.0)) * 1000.0
    main = int(np.argmax(counts))

    dof = max(1, int(mask.sum()) - np.linalg.matrix_rank(design[mask]))
    s2 = float(np.sum(resid**2)) / dof
    try:
        cov = np.linalg.pinv(design[mask].T @ design[mask]) * s2
        sigma_a = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        sigma_a = float("nan")
    rate = (nominal_period_s - a) / a * SECONDS_PER_DAY
    rate_sigma = nominal_period_s / a**2 * sigma_a * SECONDS_PER_DAY
    t_in = t[mask]
    return RateFit(
        rate_spd=float(rate),
        beat_error_ms=be_ms,
        period_s=a,
        tick_tock_offset_s=float(parity_coef[main]),
        n_used=int(mask.sum()),
        n_total=int(len(t)),
        span_s=float(t_in.max() - t_in.min()),
        residual_sigma_us=1.4826 * float(np.median(np.abs(resid - np.median(resid)))) * 1e6,
        rate_sigma_spd=float(rate_sigma),
    )


class SlidingRateEstimator:
    """Keeps the beats of the last ``window_s`` seconds and fits rate / beat error.

    Beats are stored as stream sample positions so a changed clock calibration
    (``fs_eff``) applies to all of them at fit time.
    """

    def __init__(
        self,
        nominal_period_s: float,
        window_s: float = 15.0,
        mad_k: float = 5.0,
        min_beats: int = 8,
    ) -> None:
        self.nominal_period_s = float(nominal_period_s)
        self.window_s = float(window_s)
        self.mad_k = float(mad_k)
        self.min_beats = int(min_beats)
        self.reset()

    def reset(self) -> None:
        """Forget all beats."""
        self._index: list[int] = []
        self._sample: list[float] = []
        self._segment: list[int] = []
        self._peak: list[float] = []
        self._noise: list[float] = []

    def __len__(self) -> int:
        return len(self._sample)

    def add(self, beats: Iterable) -> None:
        """Append :class:`~wea.core.beat_timing.BeatOnset`-like objects."""
        for b in beats:
            self._index.append(int(b.index))
            self._sample.append(float(b.sample))
            self._segment.append(int(b.segment))
            self._peak.append(float(b.peak))
            self._noise.append(float(b.noise))

    def _prune(self, fs: float) -> None:
        if not self._sample:
            return
        cutoff = self._sample[-1] - self.window_s * fs
        k = int(np.searchsorted(np.asarray(self._sample), cutoff, side="left"))
        if k > 0:
            for buf in (self._index, self._sample, self._segment, self._peak, self._noise):
                del buf[:k]

    def fit(self, fs_eff: float) -> RateFit | None:
        """Fit the beats inside the window (times = sample / ``fs_eff``)."""
        self._prune(fs_eff)
        if len(self._sample) < self.min_beats:
            return None
        return fit_rate_be(
            np.asarray(self._sample) / fs_eff,
            self.nominal_period_s,
            indices=np.asarray(self._index),
            segments=np.asarray(self._segment),
            mad_k=self.mad_k,
            min_beats=self.min_beats,
        )

    def window_levels(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Envelope peaks and pre-beat noise of the beats in the window (for the SNR)."""
        return np.asarray(self._peak), np.asarray(self._noise)

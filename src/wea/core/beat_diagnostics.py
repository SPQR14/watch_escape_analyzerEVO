"""Beat-timing diagnostics: line fit, residual histogram and step (cycle-slip) classification.

Used by ``scripts/diagnose_beats.py`` to decide whether beat timestamp errors are a bell
(random timing noise) or discrete steps (the onset detector jumping between carrier cycles,
or between sub-events P1/P2/P3 of the escapement sound).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch


@dataclass(slots=True)
class LineFit:
    """Least-squares fit ``t = a * n + b + c * (n mod 2)`` of beat timestamps.

    Attributes:
        a_s: Fitted beat period in seconds.
        b_s: Intercept in seconds.
        c_s: Tick/tock offset in seconds (beat error = ``|c|``).
        rate_spd: ``(T_nominal - a) / a * 86400`` (positive = watch gains).
        beat_error_ms: ``|c|`` in milliseconds.
        residuals_us: Residuals of every input beat in microseconds.
        indices: Integer beat indices ``n`` used for the fit.
    """

    a_s: float
    b_s: float
    c_s: float
    rate_spd: float
    beat_error_ms: float
    residuals_us: NDArray[np.float64]
    indices: NDArray[np.int64]


@dataclass(slots=True)
class ResidualClassification:
    """Shape of the residual distribution.

    Attributes:
        kind: ``"bell"``, ``"carrier_steps"`` (clusters ~k / f_carrier apart),
            ``"subevent_steps"`` (clusters >= 1 ms apart: P1/P2/P3 confusion) or
            ``"steps"`` (several clusters at another spacing).
        robust_sigma_us: ``1.4826 * MAD`` of the residuals.
        cluster_centers_us: Centers of the detected clusters.
        cluster_fractions: Fraction of beats in each cluster.
        carrier_hz: Carrier frequency used to interpret the spacing (0 if unknown).
        summary: Human-readable one-line verdict.
    """

    kind: str
    robust_sigma_us: float
    cluster_centers_us: list[float] = field(default_factory=list)
    cluster_fractions: list[float] = field(default_factory=list)
    carrier_hz: float = 0.0
    summary: str = ""


def beat_indices_from_times(times_s: NDArray[np.float64], nominal_period_s: float) -> NDArray:
    """Return integer beat indices, ``n_k = n_(k-1) + round(dt / T)`` (tolerates gaps)."""
    t = np.asarray(times_s, dtype=np.float64)
    if len(t) == 0:
        return np.empty(0, dtype=np.int64)
    steps = np.maximum(1, np.rint(np.diff(t) / nominal_period_s).astype(np.int64))
    return np.concatenate([[0], np.cumsum(steps)]).astype(np.int64)


def fit_line_with_parity(
    times_s: NDArray[np.float64],
    nominal_period_s: float,
    indices: NDArray[np.int64] | None = None,
) -> LineFit:
    """Ordinary least squares of ``t = a n + b + c (n mod 2)`` over all beats (no rejection).

    The parity term keeps the beat error from showing up as two residual clusters.
    """
    t = np.asarray(times_s, dtype=np.float64)
    n = beat_indices_from_times(t, nominal_period_s) if indices is None else np.asarray(indices)
    if len(t) < 4:
        raise ValueError("At least 4 beats are required for a line fit")
    design = np.column_stack([n.astype(np.float64), np.ones(len(t)), (n % 2).astype(np.float64)])
    t_ref = t[0]
    coef, *_ = np.linalg.lstsq(design, t - t_ref, rcond=None)
    a, b, c = (float(v) for v in coef)
    residuals = (t - t_ref) - design @ coef
    return LineFit(
        a_s=a,
        b_s=b + t_ref,
        c_s=c,
        rate_spd=(nominal_period_s - a) / a * 86400.0,
        beat_error_ms=abs(c) * 1000.0,
        residuals_us=residuals * 1e6,
        indices=n.astype(np.int64),
    )


def estimate_carrier_hz(
    signal: NDArray[np.float64],
    fs: float,
    band_hz: tuple[float, float] = (2000.0, 10000.0),
) -> float:
    """Return the frequency of maximum power of ``signal`` inside ``band_hz`` (Welch PSD)."""
    x = np.asarray(signal, dtype=np.float64)
    if len(x) < 1024:
        return 0.0
    freqs, psd = welch(x, fs=fs, nperseg=min(8192, len(x)))
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][int(np.argmax(psd[mask]))])


def residual_histogram(
    residuals_us: NDArray[np.float64],
    bin_us: float | None = None,
    clip_sigma: float = 5.0,
    max_bins: int = 60,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Histogram of residuals; returns ``(bin_edges_us, counts)``.

    The range is clipped to ``+-clip_sigma`` robust sigmas (min +-100 us); values outside
    land in the edge bins. With ``bin_us=None`` the bin is the smallest of 5/10/20/25/50/100
    us that keeps at most ``max_bins`` bins.
    """
    r = np.asarray(residuals_us, dtype=np.float64)
    med = float(np.median(r))
    sigma = 1.4826 * float(np.median(np.abs(r - med))) + 1e-9
    half = max(100.0, clip_sigma * sigma)
    if bin_us is None:
        bin_us = next(
            (b for b in (5.0, 10.0, 20.0, 25.0, 50.0, 100.0) if 2.0 * half / b <= max_bins),
            200.0,
        )
    edges = np.arange(med - half, med + half + bin_us, bin_us)
    counts, edges = np.histogram(np.clip(r, edges[0], edges[-1]), bins=edges)
    return edges, counts


def robust_line_fit(
    times_s: NDArray[np.float64],
    nominal_period_s: float,
    mad_k: float = 5.0,
    iterations: int = 4,
) -> tuple[LineFit, NDArray[np.bool_]]:
    """Line + parity fit that re-fits after dropping residuals beyond ``mad_k`` MADs.

    Returns the fit on the inliers (residuals are reported for ALL beats against it, so a
    far cluster of mis-detected beats stays visible) and the inlier mask.
    """
    t = np.asarray(times_s, dtype=np.float64)
    n = beat_indices_from_times(t, nominal_period_s)
    mask = np.ones(len(t), dtype=bool)
    fit = fit_line_with_parity(t, nominal_period_s, n)
    for _ in range(iterations):
        r = fit.residuals_us
        med = float(np.median(r[mask]))
        mad = max(float(np.median(np.abs(r[mask] - med))), 1.0)
        new_mask = np.abs(r - med) <= mad_k * mad
        if new_mask.sum() < 4 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        inner = fit_line_with_parity(t[mask], nominal_period_s, n[mask])
        design = np.column_stack([n, np.ones(len(t)), n % 2]).astype(np.float64)
        resid = t - (design @ np.array([inner.a_s, inner.b_s, inner.c_s]))
        fit = LineFit(
            a_s=inner.a_s,
            b_s=inner.b_s,
            c_s=inner.c_s,
            rate_spd=inner.rate_spd,
            beat_error_ms=inner.beat_error_ms,
            residuals_us=resid * 1e6,
            indices=n,
        )
    return fit, mask


def _gap_clusters(values: NDArray[np.float64], gap: float) -> list[NDArray[np.float64]]:
    """Split sorted ``values`` wherever consecutive values are more than ``gap`` apart."""
    v = np.sort(values)
    if len(v) == 0:
        return []
    cuts = np.flatnonzero(np.diff(v) > gap) + 1
    return np.split(v, cuts)


def _significant_peaks(dens: NDArray[np.float64], peaks: NDArray[np.int64]) -> NDArray[np.int64]:
    """Merge density maxima not separated by a real valley (<= 50 % of the smaller peak).

    Sampling noise of a kernel density puts small ripples on a single bell; only maxima with
    a clear dip between them are distinct clusters.
    """
    kept: list[int] = []
    for pk in peaks:
        if kept:
            prev = kept[-1]
            valley = float(np.min(dens[prev : pk + 1]))
            if valley > 0.5 * min(float(dens[prev]), float(dens[pk])):
                if dens[pk] > dens[prev]:
                    kept[-1] = int(pk)
                continue
        kept.append(int(pk))
    return np.asarray(kept, dtype=np.int64)


def classify_residuals(
    residuals_us: NDArray[np.float64],
    carrier_hz: float = 0.0,
    bandwidth_us: float | None = None,
    min_cluster_fraction: float = 0.05,
) -> ResidualClassification:
    """Decide whether residuals form one bell or several discrete clusters (steps).

    Two stages:
        1. Coarse: split the sorted residuals at gaps wider than ``max(500 us, 4 sigma)``.
           Several populated groups >= 1 ms apart mean the onset jumps between escapement
           sub-events (P1/P2/P3).
        2. Fine, inside the largest group: Gaussian kernel density on a 1 us grid with a
           narrow kernel (15 % of the carrier period, 5-25 us). Maxima about one carrier
           period apart mean carrier cycle slips.
    """
    r = np.asarray(residuals_us, dtype=np.float64)
    n_total = max(1, len(r))
    med = float(np.median(r))
    sigma = 1.4826 * float(np.median(np.abs(r - med)))
    carrier_period_us = 1e6 / carrier_hz if carrier_hz > 0.0 else 0.0

    groups = [
        g
        for g in _gap_clusters(r, max(500.0, 4.0 * sigma))
        if len(g) / n_total >= min_cluster_fraction
    ]
    if len(groups) >= 2:
        centers = [float(np.median(g)) for g in groups]
        fractions = [len(g) / n_total for g in groups]
        spacing = float(np.median(np.diff(centers)))
        kind = "subevent_steps" if spacing >= 1000.0 else "steps"
        detail = (
            "onset jumps between escapement sub-events P1/P2/P3"
            if kind == "subevent_steps"
            else "discrete timing jumps"
        )
        parts = ", ".join(f"{c / 1000.0:+.2f} ms ({f:.0%})" for c, f in zip(centers, fractions))
        return ResidualClassification(
            kind=kind,
            robust_sigma_us=sigma,
            cluster_centers_us=centers,
            cluster_fractions=fractions,
            carrier_hz=float(carrier_hz),
            summary=f"Steps: {len(groups)} groups {spacing / 1000.0:.2f} ms apart [{parts}] - {detail}",
        )

    main = max(_gap_clusters(r, max(500.0, 4.0 * sigma)), key=len) if len(r) else r
    if bandwidth_us is None:
        bandwidth_us = float(np.clip(0.15 * carrier_period_us, 5.0, 25.0)) if carrier_hz else 10.0
    lo, hi = float(np.min(main)) - 3 * bandwidth_us, float(np.max(main)) + 3 * bandwidth_us
    grid = np.arange(lo, hi + 1.0, 1.0)
    dens = np.exp(-0.5 * ((grid[:, None] - main[None, :]) / bandwidth_us) ** 2).sum(axis=1)
    is_peak = np.r_[False, (dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:]), False]
    peaks = _significant_peaks(dens, np.flatnonzero(is_peak & (dens >= 0.08 * float(dens.max()))))
    centers: list[float] = []
    fractions: list[float] = []
    if len(peaks):
        peak_pos = grid[peaks]
        owner = np.argmin(np.abs(main[:, None] - peak_pos[None, :]), axis=1)
        for k, pos in enumerate(peak_pos):
            frac = float(np.sum(owner == k)) / n_total
            if frac >= min_cluster_fraction:
                centers.append(float(pos))
                fractions.append(frac)

    spacing = float(np.median(np.diff(sorted(centers)))) if len(centers) >= 2 else 0.0
    if (
        len(centers) >= 2
        and carrier_period_us > 0.0
        and abs(spacing / carrier_period_us - 1.0) <= 0.3
    ):
        kind = "carrier_steps"
        summary = (
            f"Steps: {len(centers)} clusters {spacing:.0f} us apart "
            f"~ 1 carrier cycle ({carrier_hz:.0f} Hz -> {carrier_period_us:.0f} us)"
        )
    elif (
        len(centers) >= 2
        and spacing > 4.0 * bandwidth_us
        and (carrier_period_us == 0.0 or sigma < 0.8 * carrier_period_us)
    ):
        kind = "steps"
        summary = f"Steps: {len(centers)} clusters {spacing:.0f} us apart"
    else:
        kind = "bell"
        summary = f"Bell: one cluster, robust sigma {sigma:.1f} us"
        if carrier_period_us > 0.0 and sigma >= 0.8 * carrier_period_us:
            summary += " (too wide to resolve single carrier-cycle slips)"
    return ResidualClassification(
        kind=kind,
        robust_sigma_us=sigma,
        cluster_centers_us=centers,
        cluster_fractions=fractions,
        carrier_hz=float(carrier_hz),
        summary=summary,
    )


def text_histogram(edges: NDArray[np.float64], counts: NDArray[np.int64], width: int = 50) -> str:
    """Render a histogram as text lines (``center_us | ####  count``), without empty tails."""
    nonzero = np.flatnonzero(counts)
    if len(nonzero) == 0:
        return ""
    first, last = int(nonzero[0]), int(nonzero[-1])
    peak = max(1, int(np.max(counts)))
    lines = []
    for lo, hi, c in zip(
        edges[first : last + 1], edges[first + 1 : last + 2], counts[first : last + 1]
    ):
        bar = "#" * int(round(width * int(c) / peak))
        lines.append(f"{0.5 * (lo + hi):+8.0f} us | {bar:<{width}} {int(c)}")
    return "\n".join(lines)

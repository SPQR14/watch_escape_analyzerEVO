"""ADC clock drift estimation and quartz reference calibration for horological telemetry."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class SampleRateDriftEstimator:
    """Estimates and compensates ADC sample clock crystal drift.

    Tracks incoming audio buffer chunks against high-resolution monotonic time
    (``time.perf_counter_ns()``) over a sliding window and performs centered
    Ordinary Least Squares (OLS) linear regression to recover the effective
    sampling rate within sub-ppm precision (< 1.15 ppm for ±0.1 s/d rate accuracy).
    Also supports static calibration from a 1 Hz Lavet stepping motor (3,600 BPH)
    or high-frequency quartz reference watch.
    """

    SECONDS_PER_DAY: float = 86400.0
    DEFAULT_WINDOW_SEC: float = 30.0
    DEFAULT_MIN_DURATION_SEC: float = 5.0
    DEFAULT_MIN_CHUNKS: int = 10
    DEFAULT_MAX_DRIFT_PPM: float = 500.0

    def __init__(
        self,
        nominal_fs: float = 48000.0,
        window_sec: float = DEFAULT_WINDOW_SEC,
        min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
        min_chunks: int = DEFAULT_MIN_CHUNKS,
        max_drift_ppm: float = DEFAULT_MAX_DRIFT_PPM,
        calibration_ppm: float = 0.0,
    ) -> None:
        """Initialize the sample rate drift estimator.

        Args:
            nominal_fs: Nominal ADC sampling frequency in Hz (e.g., 48000.0 or 96000.0).
            window_sec: Sliding regression window duration in seconds (default 30.0s).
            min_duration_sec: Minimum elapsed duration before activating dynamic OLS (default 5.0s).
            min_chunks: Minimum number of audio chunks before activating dynamic OLS (default 10).
            max_drift_ppm: Sanity guardrail threshold in parts per million (default ±500.0 ppm).
            calibration_ppm: Static quartz calibration offset in parts per million.
        """
        if nominal_fs <= 0.0:
            logger.error("Invalid nominal_fs=%.2f (must be positive)", nominal_fs)
            raise ValueError("nominal_fs must be positive")
        if window_sec <= 0.0:
            logger.error("Invalid window_sec=%.2f (must be positive)", window_sec)
            raise ValueError("window_sec must be positive")
        if min_duration_sec < 0.0:
            raise ValueError("min_duration_sec must be non-negative")
        if min_chunks < 2:
            raise ValueError("min_chunks must be at least 2")
        if max_drift_ppm <= 0.0:
            raise ValueError("max_drift_ppm must be positive")

        self.nominal_fs: float = float(nominal_fs)
        self.window_sec: float = float(window_sec)
        self.min_duration_sec: float = float(min_duration_sec)
        self.min_chunks: int = int(min_chunks)
        self.max_drift_ppm: float = float(max_drift_ppm)
        self.calibration_ppm: float = float(calibration_ppm)

        self._window: deque[tuple[float, float]] = deque()
        self._origin_ns: int | None = None
        self._last_elapsed_sec: float | None = None
        self._cumulative_samples: float = 0.0
        self._chunk_count: int = 0
        self._dynamic_drift_ppm: float = 0.0

        logger.info(
            "SampleRateDriftEstimator initialized (nominal_fs=%.1f Hz, window=%.1fs, "
            "min_duration=%.1fs, min_chunks=%d, max_drift_ppm=%.1f, calibration_ppm=%+.3f)",
            self.nominal_fs,
            self.window_sec,
            self.min_duration_sec,
            self.min_chunks,
            self.max_drift_ppm,
            self.calibration_ppm,
        )

    def reset(self) -> None:
        """Clear the dynamic sliding window and reset cumulative sample tracking."""
        self._window.clear()
        self._origin_ns = None
        self._last_elapsed_sec = None
        self._cumulative_samples = 0.0
        self._chunk_count = 0
        self._dynamic_drift_ppm = 0.0
        logger.debug("SampleRateDriftEstimator dynamic tracking state reset")

    @property
    def drift_ppm(self) -> float:
        """Return total clock drift in parts per million (static calibration + dynamic OLS)."""
        return float(self.calibration_ppm + self._dynamic_drift_ppm)

    @property
    def effective_fs(self) -> float:
        """Return the effective sample rate in Hz combining static and dynamic drift."""
        return float(self.nominal_fs * (1.0 + self.drift_ppm * 1e-6))

    @property
    def rate_bias_spd(self) -> float:
        """Return the horological rate bias in seconds per day (86400 * drift)."""
        drift_fraction = self.drift_ppm * 1e-6
        return float(self.SECONDS_PER_DAY * drift_fraction)

    def update(
        self,
        chunk_samples: int | float,
        timestamp_ns: int | None = None,
        elapsed_sec: float | None = None,
    ) -> float:
        """Record a new audio chunk and update the sliding-window OLS drift estimate.

        Args:
            chunk_samples: Number of samples in the newly arrived audio buffer chunk.
            timestamp_ns: Optional monotonic timestamp in nanoseconds (defaults to
                ``time.perf_counter_ns()`` when ``elapsed_sec`` is not provided).
            elapsed_sec: Optional explicit monotonic elapsed time in seconds.

        Returns:
            Updated ``effective_fs`` in Hz.
        """
        n_samp = float(chunk_samples)
        if n_samp < 0.0:
            logger.warning("Rejected negative chunk_samples=%.2f", n_samp)
            return self.effective_fs

        if elapsed_sec is not None:
            t_sec = float(elapsed_sec)
        else:
            now_ns = int(timestamp_ns) if timestamp_ns is not None else time.perf_counter_ns()
            if self._origin_ns is None:
                self._origin_ns = now_ns
            t_sec = (now_ns - self._origin_ns) * 1e-9

        if len(self._window) == 0:
            if t_sec == 0.0:
                self._window.append((0.0, n_samp))
                self._last_elapsed_sec = 0.0
                self._cumulative_samples = n_samp
                self._chunk_count = 1
            elif t_sec > 0.0 and n_samp > 0.0:
                inst_fs = n_samp / t_sec
                inst_ppm = ((inst_fs - self.nominal_fs) / self.nominal_fs) * 1e6
                if abs(inst_ppm) <= self.max_drift_ppm:
                    self._window.append((0.0, 0.0))
                    self._window.append((t_sec, n_samp))
                    self._last_elapsed_sec = t_sec
                    self._cumulative_samples = n_samp
                    self._chunk_count = 1
                else:
                    logger.warning(
                        "Rejected initial chunk glitch (inst_ppm=%+.2f exceeds ±%.1f ppm)",
                        inst_ppm,
                        self.max_drift_ppm,
                    )
                    self._window.append((0.0, 0.0))
                    self._last_elapsed_sec = t_sec
            return self.effective_fs

        last_t = self._last_elapsed_sec if self._last_elapsed_sec is not None else 0.0
        dt = t_sec - last_t
        if dt <= 0.0:
            logger.warning("Rejected non-monotonic clock drift update (dt=%.6f s)", dt)
            return self.effective_fs

        inst_fs = n_samp / dt
        inst_ppm = ((inst_fs - self.nominal_fs) / self.nominal_fs) * 1e6
        if abs(inst_ppm) > self.max_drift_ppm:
            logger.warning(
                "Rejected ADC clock glitch: chunk drift %+.2f ppm exceeds ±%.1f ppm guardrail",
                inst_ppm,
                self.max_drift_ppm,
            )
            self._last_elapsed_sec = t_sec
            return self.effective_fs

        valid_elapsed = self._window[-1][0] + dt
        self._cumulative_samples += n_samp
        self._last_elapsed_sec = t_sec
        self._chunk_count += 1
        self._window.append((valid_elapsed, self._cumulative_samples))

        while len(self._window) > 2 and (valid_elapsed - self._window[0][0]) > self.window_sec:
            self._window.popleft()

        span_sec = self._window[-1][0] - self._window[0][0]
        if (
            self._chunk_count >= self.min_chunks
            and span_sec >= self.min_duration_sec - 1e-9
            and len(self._window) >= 2
        ):
            ols_fs = self._compute_centered_ols_fs()
            ols_ppm = ((ols_fs - self.nominal_fs) / self.nominal_fs) * 1e6
            if abs(ols_ppm) <= self.max_drift_ppm:
                self._dynamic_drift_ppm = float(ols_ppm)
            else:
                logger.warning(
                    "Rejected OLS regression drift %+.2f ppm exceeding ±%.1f ppm guardrail",
                    ols_ppm,
                    self.max_drift_ppm,
                )

        return self.effective_fs

    add_chunk = update
    record_chunk = update

    def _compute_centered_ols_fs(self) -> float:
        """Compute the centered Ordinary Least Squares slope of samples vs monotonic time."""
        data = np.asarray(self._window, dtype=np.float64)
        t = data[:, 0]
        y = data[:, 1]
        t_centered = t - np.mean(t)
        y_centered = y - np.mean(y)
        denom = float(np.dot(t_centered, t_centered))
        if denom <= 1e-18:
            return self.nominal_fs
        return float(np.dot(t_centered, y_centered) / denom)

    def calibrate_from_quartz_intervals(
        self,
        measured_samples_per_tick: NDArray[np.float64],
        nominal_fs: float | None = None,
        target_tick_sec: float = 1.0,
    ) -> float:
        """Compute and set ``calibration_ppm`` from a quartz reference watch.

        Supports a 1 Hz Lavet stepping motor (3,600 BPH, ``target_tick_sec=1.0``) or
        high-frequency quartz reference intervals.

        Args:
            measured_samples_per_tick: 1D array of measured ADC sample counts per quartz tick.
            nominal_fs: Nominal sample rate in Hz (defaults to ``self.nominal_fs``).
            target_tick_sec: Exact reference interval between ticks in seconds (default 1.0s).

        Returns:
            Computed ``calibration_ppm`` in parts per million.
        """
        if target_tick_sec <= 0.0:
            raise ValueError("target_tick_sec must be positive")

        ref_fs = float(nominal_fs) if nominal_fs is not None else self.nominal_fs
        if ref_fs <= 0.0:
            raise ValueError("nominal_fs must be positive")
        self.nominal_fs = ref_fs

        intervals = np.asarray(measured_samples_per_tick, dtype=np.float64).ravel()
        if len(intervals) == 0:
            raise ValueError("measured_samples_per_tick must not be empty")

        expected_samples = ref_fs * target_tick_sec
        interval_ppm = ((intervals - expected_samples) / expected_samples) * 1e6
        valid_mask = np.abs(interval_ppm) <= self.max_drift_ppm
        if not np.any(valid_mask):
            logger.warning(
                "All %d quartz intervals exceeded ±%.1f ppm guardrail; retaining calibration_ppm=%+.3f",
                len(intervals),
                self.max_drift_ppm,
                self.calibration_ppm,
            )
            return self.calibration_ppm

        valid_intervals = intervals[valid_mask]
        mean_samples = float(np.mean(valid_intervals))
        measured_fs = mean_samples / target_tick_sec
        cal_ppm = ((measured_fs - ref_fs) / ref_fs) * 1e6

        self.calibration_ppm = float(cal_ppm)
        logger.info(
            "Quartz calibration completed over %d/%d valid ticks: calibration_ppm=%+.4f ppm "
            "(effective_fs=%.4f Hz, rate_bias=%+.3f s/d)",
            len(valid_intervals),
            len(intervals),
            self.calibration_ppm,
            self.effective_fs,
            self.rate_bias_spd,
        )
        return self.calibration_ppm

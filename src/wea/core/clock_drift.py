"""ADC clock drift estimation, quartz calibration, and HFT-style atomic NTP disciplining."""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections import deque

import numpy as np
from numpy.typing import NDArray

from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class AtomicNTPDiscipline:
    """HFT-style minimum-delay UDP NTP client for disciplining host monotonic clock skew.

    Queries a Stratum-1/Anycast NTP server with a burst of 48-byte NTPv3 packets,
    applies a minimum-delay hull filter to reject asymmetric router queue jitter,
    and estimates the host CPU monotonic oscillator skew (``cpu_skew_ppm``).
    """

    NTP_EPOCH_OFFSET: float = 2208988800.0
    NTP_PACKET_SIZE: int = 48
    NTP_REQUEST_PACKET: bytes = b"\x1b" + 47 * b"\0"
    TWO_POW_32: float = float(1 << 32)
    MIN_ANCHOR_SPAN_SEC: float = 15.0
    MAX_SKEW_PPM: float = 200.0
    EMA_ALPHA: float = 0.3

    def __init__(
        self,
        host: str = "time.cloudflare.com",
        port: int = 123,
        timeout_sec: float = 0.4,
    ) -> None:
        """Initialize the atomic NTP discipline client.

        Args:
            host: NTP server hostname or IP address (default ``"time.cloudflare.com"``).
            port: UDP port for NTP (default 123).
            timeout_sec: Socket timeout per packet in seconds (default 0.4s).
        """
        self.host: str = str(host)
        self.port: int = int(port)
        self.timeout_sec: float = float(timeout_sec)
        self.cpu_skew_ppm: float = 0.0
        self._anchor: tuple[float, float] | None = None

        logger.info(
            "AtomicNTPDiscipline initialized (host='%s', port=%d, timeout=%.2fs)",
            self.host,
            self.port,
            self.timeout_sec,
        )

    def reset(self) -> None:
        """Reset the NTP anchor and accumulated CPU skew estimate."""
        self._anchor = None
        self.cpu_skew_ppm = 0.0

    def sample_burst(self, burst_size: int = 4) -> tuple[float, float] | None:
        """Send a burst of UDP NTPv3 requests and select the minimum-delay packet.

        Args:
            burst_size: Number of back-to-back NTP packets in the burst (default 4).

        Returns:
            Tuple ``(local_monotonic_midpoint, atomic_server_midpoint)`` in seconds
            for the single packet with the lowest positive round-trip delay ``delta``,
            or ``None`` if offline or all packets fail.
        """
        count = max(1, int(burst_size))
        best_delay: float = float("inf")
        best_sample: tuple[float, float] | None = None

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(self.timeout_sec)
                for _ in range(count):
                    try:
                        t1 = time.perf_counter_ns() * 1e-9
                        sock.sendto(self.NTP_REQUEST_PACKET, (self.host, self.port))
                        raw_data, _ = sock.recvfrom(1024)
                        t4 = time.perf_counter_ns() * 1e-9
                    except OSError:
                        continue

                    if len(raw_data) < self.NTP_PACKET_SIZE:
                        continue

                    t2_sec, t2_frac, t3_sec, t3_frac = struct.unpack("!4I", raw_data[32:48])
                    t2 = (float(t2_sec) - self.NTP_EPOCH_OFFSET) + (
                        float(t2_frac) / self.TWO_POW_32
                    )
                    t3 = (float(t3_sec) - self.NTP_EPOCH_OFFSET) + (
                        float(t3_frac) / self.TWO_POW_32
                    )

                    delta = (t4 - t1) - (t3 - t2)
                    if 0.0 < delta < best_delay:
                        best_delay = delta
                        local_mid = 0.5 * (t1 + t4)
                        atomic_mid = 0.5 * (t2 + t3)
                        best_sample = (local_mid, atomic_mid)
        except OSError as exc:
            logger.debug("AtomicNTPDiscipline burst failed (offline or unreachable): %s", exc)
            return None

        return best_sample

    def update_cpu_skew(self, burst_size: int = 4) -> float:
        """Update the EMA-smoothed host CPU monotonic clock skew in ppm.

        Stores the first valid burst midpoint pair as ``self._anchor``. On subsequent
        calls where ``dt_local >= 15.0`` seconds, computes ``raw_skew_ppm``, applies
        a ±200 ppm sanity guardrail, and updates ``self.cpu_skew_ppm`` via EMA
        (``0.3 * raw + 0.7 * prev``).

        Args:
            burst_size: Number of UDP NTP packets per burst (default 4).

        Returns:
            Updated ``self.cpu_skew_ppm`` in parts per million.
        """
        sample = self.sample_burst(burst_size=burst_size)
        if sample is None:
            return self.cpu_skew_ppm

        t_local, t_atomic = sample
        if self._anchor is None:
            self._anchor = (t_local, t_atomic)
            logger.debug(
                "AtomicNTPDiscipline anchor established (t_local=%.6f, t_atomic=%.6f)",
                t_local,
                t_atomic,
            )
            return self.cpu_skew_ppm

        anchor_local, anchor_atomic = self._anchor
        dt_local = t_local - anchor_local
        dt_atomic = t_atomic - anchor_atomic

        if dt_local >= self.MIN_ANCHOR_SPAN_SEC and dt_atomic > 0.0:
            raw_skew_ppm = ((dt_local - dt_atomic) / dt_atomic) * 1e6
            if abs(raw_skew_ppm) < self.MAX_SKEW_PPM:
                self.cpu_skew_ppm = float(
                    self.EMA_ALPHA * raw_skew_ppm + (1.0 - self.EMA_ALPHA) * self.cpu_skew_ppm
                )
                logger.info(
                    "AtomicNTPDiscipline updated cpu_skew_ppm=%+.3f ppm (raw=%+.3f ppm, dt=%.1fs)",
                    self.cpu_skew_ppm,
                    raw_skew_ppm,
                    dt_local,
                )
            else:
                logger.warning(
                    "Rejected anomalous NTP CPU skew %+.2f ppm (exceeds ±%.1f ppm guardrail)",
                    raw_skew_ppm,
                    self.MAX_SKEW_PPM,
                )

        return self.cpu_skew_ppm


class SampleRateDriftEstimator:
    """Estimates and compensates ADC sample clock crystal drift.

    Tracks incoming audio buffer chunks against high-resolution monotonic time
    (``time.perf_counter_ns()``) over a sliding window, evaluates decimated centered
    Ordinary Least Squares (OLS) linear regression smoothed by a first-order EMA,
    and combines dynamic ADC drift, static quartz calibration, and atomic NTP CPU skew.
    """

    SECONDS_PER_DAY: float = 86400.0
    DEFAULT_WINDOW_SEC: float = 30.0
    DEFAULT_MIN_DURATION_SEC: float = 5.0
    DEFAULT_MIN_CHUNKS: int = 10
    DEFAULT_MAX_DRIFT_PPM: float = 500.0
    DEFAULT_RECALC_INTERVAL_SEC: float = 0.5
    DEFAULT_EMA_ALPHA: float = 0.15

    def __init__(
        self,
        nominal_fs: float = 48000.0,
        window_sec: float = DEFAULT_WINDOW_SEC,
        min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
        min_chunks: int = DEFAULT_MIN_CHUNKS,
        max_drift_ppm: float = DEFAULT_MAX_DRIFT_PPM,
        calibration_ppm: float = 0.0,
        recalc_interval_sec: float = DEFAULT_RECALC_INTERVAL_SEC,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        cpu_skew_ppm: float = 0.0,
        dynamic_tracking: bool = True,
    ) -> None:
        """Initialize the sample rate drift estimator.

        Args:
            nominal_fs: Nominal ADC sampling frequency in Hz (e.g., 48000.0 or 96000.0).
            window_sec: Sliding regression window duration in seconds (default 30.0s).
            min_duration_sec: Minimum elapsed duration before activating dynamic OLS (default 5.0s).
            min_chunks: Minimum number of audio chunks before activating dynamic OLS (default 10).
            max_drift_ppm: Sanity guardrail threshold in parts per million (default ±500.0 ppm).
            calibration_ppm: Static quartz calibration offset in parts per million.
            recalc_interval_sec: Minimum elapsed interval between OLS recalculations (default 0.5s).
            ema_alpha: First-order EMA smoothing factor for OLS slope updates, clamped to [0.01, 1.0].
            cpu_skew_ppm: Host CPU monotonic clock skew in ppm measured via AtomicNTPDiscipline.
            dynamic_tracking: If False, ``update()`` does not regress sample counts against
                the host clock and ``effective_fs`` is the nominal rate corrected only by
                ``calibration_ppm``. The host clock is itself an uncalibrated crystal and the
                audio callback timestamps carry scheduling jitter (milliseconds on Windows),
                so without NTP disciplining the "correction" adds error instead of removing it.
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
        if recalc_interval_sec < 0.0:
            raise ValueError("recalc_interval_sec must be non-negative")

        self.nominal_fs: float = float(nominal_fs)
        self.window_sec: float = float(window_sec)
        self.min_duration_sec: float = float(min_duration_sec)
        self.min_chunks: int = int(min_chunks)
        self.max_drift_ppm: float = float(max_drift_ppm)
        self.calibration_ppm: float = float(calibration_ppm)
        self.recalc_interval_sec: float = float(recalc_interval_sec)
        self.ema_alpha: float = float(np.clip(ema_alpha, 0.01, 1.0))
        self.cpu_skew_ppm: float = float(cpu_skew_ppm)
        self.dynamic_tracking: bool = bool(dynamic_tracking)

        self._lock = threading.Lock()
        self._history: deque[tuple[float, float]] = deque()
        self._window: deque[tuple[float, float]] = self._history
        self._origin_ns: int | None = None
        self._last_elapsed_sec: float | None = None
        self._last_recalc_sec: float = -float("inf")
        self._cumulative_samples: float = 0.0
        self._chunk_count: int = 0
        self._ols_eval_count: int = 0
        self._dynamic_fs: float = float(self.nominal_fs)
        self._dynamic_drift_ppm: float = 0.0
        self._is_converged: bool = False

        logger.info(
            "SampleRateDriftEstimator initialized (nominal_fs=%.1f Hz, window=%.1fs, "
            "min_duration=%.1fs, min_chunks=%d, max_drift_ppm=%.1f, calibration_ppm=%+.3f, "
            "recalc_interval=%.2fs, ema_alpha=%.2f, cpu_skew_ppm=%+.3f)",
            self.nominal_fs,
            self.window_sec,
            self.min_duration_sec,
            self.min_chunks,
            self.max_drift_ppm,
            self.calibration_ppm,
            self.recalc_interval_sec,
            self.ema_alpha,
            self.cpu_skew_ppm,
        )

    def reset(self) -> None:
        """Clear the dynamic sliding window and reset cumulative sample and EMA state."""
        with self._lock:
            self._history.clear()
            self._origin_ns = None
            self._last_elapsed_sec = None
            self._last_recalc_sec = -float("inf")
            self._cumulative_samples = 0.0
            self._chunk_count = 0
            self._ols_eval_count = 0
            self._dynamic_fs = float(self.nominal_fs)
            self._dynamic_drift_ppm = 0.0
            self._is_converged = False
        logger.debug("SampleRateDriftEstimator dynamic tracking state reset")

    @property
    def is_converged(self) -> bool:
        """Return True once the dynamic OLS estimator has converged at least once."""
        return self._is_converged

    @property
    def dynamic_fs(self) -> float:
        """Return the EMA-smoothed dynamic sample rate estimate in Hz."""
        return self._dynamic_fs if self._is_converged else self.nominal_fs

    @property
    def effective_fs(self) -> float:
        """Return the effective sample rate in Hz combining dynamic OLS, quartz cal, and NTP skew.

        The NTP CPU skew only corrects a rate that was measured against the host clock, so it
        is applied only once the dynamic OLS estimate is active.
        """
        if self._is_converged:
            base_fs = self._dynamic_fs
            skew_ppm = self.cpu_skew_ppm
        else:
            base_fs = self.nominal_fs
            skew_ppm = 0.0
        return float(base_fs * (1.0 + (self.calibration_ppm - skew_ppm) * 1e-6))

    @property
    def drift_ppm(self) -> float:
        """Return total net clock drift in parts per million relative to ``nominal_fs``."""
        return float(((self.effective_fs - self.nominal_fs) / self.nominal_fs) * 1e6)

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
        """Record a new audio chunk and update the decimated EMA-smoothed OLS drift estimate.

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
        if not self.dynamic_tracking:
            return self.effective_fs

        with self._lock:
            if elapsed_sec is not None:
                t_sec = float(elapsed_sec)
            else:
                now_ns = int(timestamp_ns) if timestamp_ns is not None else time.perf_counter_ns()
                if self._origin_ns is None:
                    self._origin_ns = now_ns
                t_sec = (now_ns - self._origin_ns) * 1e-9

            if len(self._history) == 0:
                self._dynamic_fs = float(self.nominal_fs)
                if t_sec <= 0.0:
                    self._history.append((0.0, n_samp))
                    self._last_elapsed_sec = 0.0
                    self._cumulative_samples = n_samp
                    self._chunk_count = 1
                elif n_samp > 0.0:
                    self._history.append((0.0, 0.0))
                    self._history.append((t_sec, n_samp))
                    self._last_elapsed_sec = t_sec
                    self._cumulative_samples = n_samp
                    self._chunk_count = 1
                return self.effective_fs

            last_t = self._last_elapsed_sec if self._last_elapsed_sec is not None else 0.0
            dt = t_sec - last_t
            if dt <= 0.0:
                return self.effective_fs

            # For coarse synthetic/macro steps (dt >= 0.25s), ignore extreme sample-count glitches
            # silently without logging false warnings on short real-time PipeWire/USB packets.
            if dt >= 0.25:
                inst_fs = n_samp / dt
                inst_ppm = ((inst_fs - self.nominal_fs) / self.nominal_fs) * 1e6
                if abs(inst_ppm) > self.max_drift_ppm:
                    self._last_elapsed_sec = t_sec
                    return self.effective_fs

            valid_elapsed = self._history[-1][0] + dt
            self._cumulative_samples += n_samp
            self._last_elapsed_sec = t_sec
            self._chunk_count += 1
            self._history.append((valid_elapsed, self._cumulative_samples))

            cutoff_elapsed = valid_elapsed - self.window_sec
            while len(self._history) > 2 and self._history[0][0] < cutoff_elapsed:
                self._history.popleft()

            span_sec = self._history[-1][0] - self._history[0][0]
            if (
                self._chunk_count >= self.min_chunks
                and span_sec >= self.min_duration_sec - 1e-9
                and len(self._history) >= 2
                and (valid_elapsed - self._last_recalc_sec) >= self.recalc_interval_sec - 1e-9
            ):
                self._last_recalc_sec = valid_elapsed
                self._ols_eval_count += 1
                raw_slope = self._compute_centered_ols_fs()
                ols_ppm = ((raw_slope - self.nominal_fs) / self.nominal_fs) * 1e6
                if abs(ols_ppm) <= self.max_drift_ppm:
                    if not self._is_converged:
                        self._dynamic_fs = float(raw_slope)
                        self._is_converged = True
                    else:
                        self._dynamic_fs = float(
                            self.ema_alpha * raw_slope + (1.0 - self.ema_alpha) * self._dynamic_fs
                        )
                    self._dynamic_drift_ppm = float(
                        ((self._dynamic_fs - self.nominal_fs) / self.nominal_fs) * 1e6
                    )
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
        """Compute centered OLS slope of samples vs time using zero-overhead buffer unpacking."""
        n_pairs = len(self._history)
        if n_pairs < 2:
            return self.nominal_fs
        data = np.fromiter(
            (val for pair in self._history for val in pair),
            dtype=np.float64,
            count=n_pairs * 2,
        ).reshape(-1, 2)
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
        if not self._is_converged:
            self._dynamic_fs = ref_fs

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

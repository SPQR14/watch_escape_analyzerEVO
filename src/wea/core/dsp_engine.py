"""Multi-threaded acoustic DSP engine for Swiss lever escapement horological telemetry."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Self

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from scipy.signal import butter, hilbert, sosfilt, sosfilt_zi, sosfiltfilt

from wea.core.clock_drift import SampleRateDriftEstimator
from wea.core.models import BeatEvent, BeatTelemetry, WatchConfig
from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class DSPEngine:
    """Multi-core acoustic DSP engine for horological escapement analysis.

    Processes raw piezoelectric microphone audio buffers (48 kHz or 96 kHz) using:
        1. 4th-order Butterworth high-pass SOS filtering (zero-phase ``sosfiltfilt`` for
           complete windows and stateful ``sosfilt`` with ``zi`` for streaming).
        2. Multi-threaded analytic signal extraction via ``scipy.signal.hilbert`` inside
           a ``scipy.fft.set_workers`` context manager (releasing the GIL in PocketFFT).
        3. Persistent ``ThreadPoolExecutor`` parallelizing per-beat micro-impulse
           inspection (unlocking P1, impulse P2, drop P3) across detected beats.
        4. Adaptive median/MAD peak detection with a ~0.7 * T0 refractory dead-time
           window and high-precision derivation of Rate (s/d), Beat Error (ms), and
           Balance Wheel Amplitude (deg).
    """

    SECONDS_PER_DAY: float = 86400.0
    DEFAULT_HIGHPASS_HZ: float = 2500.0
    DEFAULT_FILTER_ORDER: int = 4
    REFRACTORY_FACTOR: float = 0.70
    MIN_AMPLITUDE_DEG: float = 100.0
    MAX_AMPLITUDE_DEG: float = 340.0
    CARRIER_PEAK_ALIGNMENT_SEC: float = 17.5e-6

    def __init__(
        self,
        config: WatchConfig | None = None,
        bph: int = 21600,
        lift_angle: float = 52.0,
        sample_rate: float = 48000.0,
        highpass_cutoff_hz: float = DEFAULT_HIGHPASS_HZ,
        filter_order: int = DEFAULT_FILTER_ORDER,
        workers: int | None = None,
        drift_estimator: SampleRateDriftEstimator | None = None,
        p1_threshold_ratio: float = 0.18,
    ) -> None:
        """Initialize the multi-threaded DSP engine.

        Args:
            config: Optional WatchConfig instance overriding ``bph``, ``lift_angle``,
                and ``sample_rate``.
            bph: Nominal beats per hour (18000, 21600, 25200, 28800, 36000).
            lift_angle: Escapement lift angle in degrees (typically 49.0° to 53.0°).
            sample_rate: Nominal audio sampling rate in Hz (e.g., 48000 or 96000).
            highpass_cutoff_hz: Butterworth high-pass cutoff frequency in Hz (default 2500 Hz).
            filter_order: Butterworth filter order (default 4).
            workers: Number of parallel worker threads for PocketFFT and beat micro-analysis.
                Defaults to ``max(1, (os.cpu_count() or 2) // 2)``.
            drift_estimator: Optional ``SampleRateDriftEstimator`` providing ``effective_fs``.
            p1_threshold_ratio: Relative onset threshold for unlocking impulse P1.
        """
        if config is not None:
            self.config = config
        else:
            self.config = WatchConfig(
                bph=int(bph),
                lift_angle=float(lift_angle),
                sample_rate=int(round(sample_rate)),
            )

        default_workers = max(1, (os.cpu_count() or 2) // 2)
        self.workers: int = max(1, int(workers)) if workers is not None else default_workers
        self.highpass_cutoff_hz: float = float(highpass_cutoff_hz)
        self.filter_order: int = int(filter_order)
        self.drift_estimator: SampleRateDriftEstimator | None = drift_estimator
        self.p1_threshold_ratio: float = float(p1_threshold_ratio)

        self._sos: NDArray[np.float64] = self._design_highpass_sos(
            float(self.config.sample_rate),
            self.highpass_cutoff_hz,
            self.filter_order,
        )
        self._sos_zi: NDArray[np.float64] | None = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=self.workers)

        logger.info(
            "DSPEngine initialized (bph=%d, lift=%.1f deg, fs=%d Hz, fc=%.1f Hz, workers=%d)",
            self.config.bph,
            self.config.lift_angle,
            self.config.sample_rate,
            self.highpass_cutoff_hz,
            self.workers,
        )

    def __enter__(self) -> Self:
        """Enter runtime context."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Cleanly shut down worker pool on context exit."""
        self.shutdown()

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the persistent ThreadPoolExecutor cleanly."""
        if self._executor is not None:
            logger.info("Shutting down DSPEngine ThreadPoolExecutor (workers=%d)", self.workers)
            self._executor.shutdown(wait=wait)
            self._executor = None

    def reset_filter_state(self) -> None:
        """Reset the stateful streaming SOS filter state ``zi``."""
        self._sos_zi = None
        logger.debug("DSPEngine streaming SOS filter state reset")

    @staticmethod
    def _design_highpass_sos(
        sample_rate: float,
        cutoff_hz: float = DEFAULT_HIGHPASS_HZ,
        order: int = DEFAULT_FILTER_ORDER,
    ) -> NDArray[np.float64]:
        """Design a 4th-order Butterworth high-pass filter in SOS format."""
        nyquist = 0.5 * float(sample_rate)
        norm_cutoff = float(np.clip(cutoff_hz / nyquist, 1e-4, 0.99))
        return np.asarray(
            butter(order, norm_cutoff, btype="highpass", output="sos"),
            dtype=np.float64,
        )

    def filter_signal(
        self,
        raw_signal: NDArray[np.float64],
        streaming: bool = False,
        fs: float | None = None,
    ) -> NDArray[np.float64]:
        """Filter low-frequency ambient noise and rotor rumble from raw audio.

        Supports zero-phase ``sosfiltfilt`` for complete windows and stateful
        ``sosfilt`` (with ``zi`` state preservation) for continuous streaming chunks.

        Args:
            raw_signal: 1D array of raw audio samples.
            streaming: If True, apply causal stateful ``sosfilt`` using ``self._sos_zi``.
                If False, apply zero-phase ``sosfiltfilt``.
            fs: Optional sampling rate override in Hz.

        Returns:
            Filtered 1D float64 signal array.
        """
        sig = np.asarray(raw_signal, dtype=np.float64).ravel()
        if len(sig) == 0:
            return sig.copy()

        target_fs = float(fs) if fs is not None else float(self.config.sample_rate)
        if abs(target_fs - float(self.config.sample_rate)) > 1.0:
            sos = self._design_highpass_sos(target_fs, self.highpass_cutoff_hz, self.filter_order)
        else:
            sos = self._sos

        if streaming:
            if self._sos_zi is None:
                zi_base = np.asarray(sosfilt_zi(sos), dtype=np.float64)
                self._sos_zi = zi_base * float(sig[0])
            filtered, self._sos_zi = sosfilt(sos, sig, zi=self._sos_zi)
            return np.asarray(filtered, dtype=np.float64)

        return np.asarray(sosfiltfilt(sos, sig), dtype=np.float64)

    def compute_envelopes(
        self,
        filtered_signal: NDArray[np.float64],
        fs: float | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute macro-smoothed and high-resolution micro-impulse Hilbert envelopes.

        Executes ``scipy.signal.hilbert`` inside ``scipy.fft.set_workers(self.workers)``
        so PocketFFT releases the GIL and utilizes multi-core parallelism.

        Args:
            filtered_signal: High-pass filtered 1D audio buffer.
            fs: Effective sampling rate in Hz.

        Returns:
            Tuple of ``(macro_envelope, micro_envelope)`` as float64 NumPy arrays.
        """
        if len(filtered_signal) == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty

        fs_eff = float(fs) if fs is not None else self._resolve_effective_fs()

        with scipy.fft.set_workers(self.workers):
            analytic = hilbert(filtered_signal)
        inst_mag = np.abs(analytic).astype(np.float64, copy=False)

        # Fast moving-average smoothing window (~0.8 ms, within 0.5 ms - 1.0 ms) for macro-beat timing
        macro_win = max(int(round(0.0008 * fs_eff)), 3)
        macro_kernel = np.ones(macro_win, dtype=np.float64) / float(macro_win)
        macro_env = np.convolve(inst_mag, macro_kernel, mode="same")

        # High-resolution smoothing (~0.12 ms) for micro-impulse inspection (P1, P2, P3)
        micro_win = max(int(round(0.00012 * fs_eff)), 3)
        micro_kernel = np.ones(micro_win, dtype=np.float64) / float(micro_win)
        micro_env = np.convolve(inst_mag, micro_kernel, mode="same")

        return macro_env, micro_env

    def _resolve_effective_fs(self, effective_fs: float | None = None) -> float:
        """Resolve effective sample rate from explicit argument, drift estimator, or config."""
        if effective_fs is not None and effective_fs > 0.0:
            return float(effective_fs)
        if self.drift_estimator is not None:
            return float(self.drift_estimator.effective_fs)
        return float(self.config.sample_rate)

    def detect_macro_beat_peaks(
        self,
        macro_env: NDArray[np.float64],
        bph: int,
        fs_eff: float,
    ) -> list[int]:
        """Detect tick/tock peak indices using robust median+MAD and 0.7*T0 dead-time.

        Args:
            macro_env: Smoothed macro-beat Hilbert amplitude envelope.
            bph: Nominal beats per hour.
            fs_eff: Effective sampling rate in Hz.

        Returns:
            List of sample indices corresponding to dominant beat peaks.
        """
        n_samples = len(macro_env)
        if n_samples == 0:
            return []

        t0 = 3600.0 / float(bph)
        refractory_samples = max(1, int(round(self.REFRACTORY_FACTOR * t0 * fs_eff)))
        pre_samples = int(round(0.012 * fs_eff))
        post_samples = int(round(0.004 * fs_eff))

        med = float(np.median(macro_env))
        mad = float(np.median(np.abs(macro_env - med))) + 1e-12
        robust_sigma = 1.4826 * mad
        peak_ref = float(np.percentile(macro_env, 99.5))

        adaptive_thresh = max(
            med + 6.0 * robust_sigma,
            med + 0.25 * (peak_ref - med),
        )

        peaks: list[int] = []
        i = pre_samples
        limit = n_samples - post_samples
        while i < limit:
            if macro_env[i] > adaptive_thresh:
                win_end = min(n_samples, i + refractory_samples)
                local_max_idx = i + int(np.argmax(macro_env[i:win_end]))
                if pre_samples <= local_max_idx < limit:
                    peaks.append(local_max_idx)
                i = local_max_idx + refractory_samples
            else:
                i += 1

        return peaks

    def _analyze_single_beat(
        self,
        args: tuple[NDArray[np.float64], NDArray[np.float64], int, float],
    ) -> BeatEvent | None:
        """Isolate unlocking (t1), impulse (t2), and drop (t3) within a single beat window."""
        sub_fine, sub_wave, start_idx, fs_eff = args
        if len(sub_fine) < 10:
            return None

        max_fine = float(np.max(sub_fine))
        if max_fine <= 0.0:
            return None

        p3_rel_idx = int(np.argmax(sub_fine))
        p3_exact_rel = self._parabolic_peak_interp(sub_fine, p3_rel_idx)

        baseline_slice = sub_fine[: max(10, p3_rel_idx // 3)]
        local_baseline = float(np.percentile(baseline_slice, 25))
        effective_peak = max_fine - local_baseline
        if effective_peak <= 0.0:
            return None

        onset_thresh = local_baseline + self.p1_threshold_ratio * effective_peak
        above_indices = np.where(sub_fine[: p3_rel_idx + 1] >= onset_thresh)[0]

        if len(above_indices) > 0:
            onset_idx = int(above_indices[0])
            p1_search_end = min(onset_idx + int(round(0.0005 * fs_eff)), p3_rel_idx)
            if p1_search_end > onset_idx:
                p1_rel_idx = onset_idx + int(np.argmax(sub_fine[onset_idx : p1_search_end + 1]))
                p1_exact_rel = self._parabolic_peak_interp(sub_fine, p1_rel_idx)
            else:
                p1_rel_idx = onset_idx
                p1_exact_rel = float(onset_idx)
        else:
            p1_rel_idx = p3_rel_idx
            p1_exact_rel = p3_exact_rel

        if p3_rel_idx - p1_rel_idx > 4:
            mid_start = p1_rel_idx + int(0.2 * (p3_rel_idx - p1_rel_idx))
            mid_end = p1_rel_idx + int(0.8 * (p3_rel_idx - p1_rel_idx))
            if mid_end > mid_start:
                p2_rel_idx = mid_start + int(np.argmax(sub_fine[mid_start:mid_end]))
                p2_exact_rel = self._parabolic_peak_interp(sub_fine, p2_rel_idx)
            else:
                p2_exact_rel = 0.5 * (p1_exact_rel + p3_exact_rel)
        else:
            p2_exact_rel = 0.5 * (p1_exact_rel + p3_exact_rel)

        t_p1 = (start_idx + p1_exact_rel) / fs_eff
        t_p2 = (start_idx + p2_exact_rel) / fs_eff
        t_p3 = (start_idx + p3_exact_rel) / fs_eff

        return BeatEvent(
            timestamp_p1=float(t_p1),
            timestamp_p2=float(t_p2),
            timestamp_p3=float(t_p3),
            peak_energy=float(max_fine**2),
            raw_waveform=sub_wave.copy(),
        )

    @classmethod
    def compute_amplitude_from_lift_time(
        cls,
        t_lift: float,
        lift_angle: float,
        bph: int,
    ) -> float:
        """Compute balance wheel amplitude (degrees) via the horological sine relation.

        Applies ``theta = alpha / sin(pi * f * t_lift)`` where ``alpha`` is ``lift_angle``
        and ``f = bph / 7200.0`` Hz, clamped to realistic bounds [100.0°, 340.0°].
        """
        if t_lift <= 1e-6 or lift_angle <= 0.0 or bph <= 0:
            return cls.MIN_AMPLITUDE_DEG
        f_osc = float(bph) / 7200.0
        arg = float(np.clip(np.pi * f_osc * t_lift, 1e-6, np.pi - 1e-6))
        sin_val = float(np.sin(arg))
        if sin_val <= 1e-6:
            return cls.MAX_AMPLITUDE_DEG
        theta = float(lift_angle) / sin_val
        return float(np.clip(theta, cls.MIN_AMPLITUDE_DEG, cls.MAX_AMPLITUDE_DEG))

    def process_window(
        self,
        audio_buffer: NDArray[np.float64],
        bph: int | None = None,
        lift_angle: float | None = None,
        effective_fs: float | None = None,
        streaming: bool = False,
    ) -> BeatTelemetry:
        """Process a raw audio buffer and extract structured horological telemetry.

        Args:
            audio_buffer: 1D NumPy array of raw acoustic samples.
            bph: Optional BPH override (defaults to ``self.config.bph``).
            lift_angle: Optional lift angle override in degrees (defaults to ``self.config.lift_angle``).
            effective_fs: Optional effective sample rate in Hz (defaults to drift estimator or config).
            streaming: Whether to use stateful streaming filter mode.

        Returns:
            ``BeatTelemetry`` dataclass instance with rate, beat error, amplitude,
            beat timestamps, envelope, and confidence.
        """
        active_bph = int(bph) if bph is not None else self.config.bph
        active_lift = float(lift_angle) if lift_angle is not None else self.config.lift_angle
        fs_eff = self._resolve_effective_fs(effective_fs)

        filtered = self.filter_signal(audio_buffer, streaming=streaming, fs=fs_eff)
        macro_env, micro_env = self.compute_envelopes(filtered, fs=fs_eff)

        peak_indices = self.detect_macro_beat_peaks(macro_env, bph=active_bph, fs_eff=fs_eff)
        n_samples = len(filtered)
        pre_samples = int(round(0.012 * fs_eff))
        post_samples = int(round(0.004 * fs_eff))

        tasks: list[tuple[NDArray[np.float64], NDArray[np.float64], int, float]] = []
        for pk_idx in peak_indices:
            s0 = max(0, pk_idx - pre_samples)
            s1 = min(n_samples, pk_idx + post_samples)
            tasks.append((micro_env[s0:s1], filtered[s0:s1], s0, fs_eff))

        if tasks:
            if self._executor is not None and self.workers > 1 and len(tasks) > 1:
                raw_results = list(self._executor.map(self._analyze_single_beat, tasks))
            else:
                raw_results = [self._analyze_single_beat(task) for task in tasks]
            beats = [b for b in raw_results if b is not None]
        else:
            beats = []

        if len(beats) < 3:
            logger.warning(
                "Insufficient beats detected (%d < 3) for telemetry calculation",
                len(beats),
            )
            return BeatTelemetry(
                rate_spd=0.0,
                beat_error_ms=0.0,
                amplitude_deg=0.0,
                beat_times_sec=np.empty(0, dtype=np.float64),
                envelope=macro_env,
                confidence=0.0,
                snr_db=0.0,
                beats=beats,
                micro_envelope=micro_env,
                filtered_signal=filtered,
                effective_fs=fs_eff,
            )

        p1_times = np.array([b.timestamp_p1 for b in beats], dtype=np.float64)
        p3_times = np.array([b.timestamp_p3 for b in beats], dtype=np.float64)
        beat_times = 0.35 * p1_times + 0.65 * p3_times

        t0 = 3600.0 / float(active_bph)

        # Full oscillations T_{osc, i} = t_{i+2} - t_i cancel asymmetric tick/tock beat error
        t_osc = beat_times[2:] - beat_times[:-2]
        mean_t_osc = self._estimate_oscillation_period(beat_times, t_osc, 2.0 * t0)

        # Rate (s/d): 86400 * ((2 * T0 - T_osc_bar) / (2 * T0))
        rate_spd = self.SECONDS_PER_DAY * ((2.0 * t0 - mean_t_osc) / (2.0 * t0))

        # Beat Error (ms) from alternating tick->tock and tock->tick half-periods.
        # Note: SyntheticEscapeGenerator shifts alternating intervals by ±0.5 * BE, so
        # |T_tick_to_tock - T_tock_to_tick| equals the full beat error in seconds.
        half_periods = np.diff(beat_times)
        even_intervals = half_periods[0::2]
        odd_intervals = half_periods[1::2]
        mean_even = self._trimmed_mean(even_intervals)
        mean_odd = self._trimmed_mean(odd_intervals)
        beat_error_ms = abs(mean_even - mean_odd) * 1000.0

        # Amplitude (deg) from pallet lift duration between unlocking (t1) and drop (t3)
        dt_p1_p3 = (p3_times - p1_times) + self.CARRIER_PEAK_ALIGNMENT_SEC
        median_dt_13 = float(np.median(dt_p1_p3))
        f_osc = float(active_bph) / 7200.0
        actual_beat_period = 0.5 * mean_t_osc
        # Map half-angle impulse delay dt_13 to equivalent full-lift duration t_lift
        # for the horological sine relation theta = alpha / sin(pi * f * t_lift)
        half_sin = float(
            np.sin(np.clip(np.pi * median_dt_13 / actual_beat_period, 1e-6, 0.5 * np.pi - 1e-6))
        )
        full_sin = float(np.clip(2.0 * half_sin, 1e-6, 0.999999))
        t_lift_eff = float(np.arcsin(full_sin) / (np.pi * f_osc))
        amplitude_deg = self.compute_amplitude_from_lift_time(
            t_lift=t_lift_eff,
            lift_angle=active_lift,
            bph=active_bph,
        )

        even_jitter = (
            1.4826 * float(np.median(np.abs(even_intervals - np.median(even_intervals))))
            if len(even_intervals) > 1
            else 0.0
        )
        odd_jitter = (
            1.4826 * float(np.median(np.abs(odd_intervals - np.median(odd_intervals))))
            if len(odd_intervals) > 1
            else 0.0
        )
        jitter = 0.5 * (even_jitter + odd_jitter)
        confidence = float(np.clip(np.exp(-jitter / 0.0005), 0.0, 1.0))

        # Acoustic SNR (dB): ratio of peak beat envelope amplitude to median noise floor
        peak_env = float(np.median([np.sqrt(max(b.peak_energy, 0.0)) for b in beats]))
        noise_floor = float(np.median(macro_env)) + 1e-12
        snr_db = float(
            np.clip(20.0 * np.log10(max(peak_env, noise_floor) / noise_floor), 0.0, 60.0)
        )

        telemetry = BeatTelemetry(
            rate_spd=float(rate_spd),
            beat_error_ms=float(beat_error_ms),
            amplitude_deg=float(amplitude_deg),
            beat_times_sec=beat_times,
            envelope=macro_env,
            confidence=confidence,
            snr_db=snr_db,
            beats=beats,
            micro_envelope=micro_env,
            filtered_signal=filtered,
            effective_fs=fs_eff,
        )
        logger.debug(
            "DSPEngine telemetry: rate=%+.3f s/d, BE=%.4f ms, amp=%.2f deg, "
            "snr=%.1f dB, conf=%.3f (%d beats)",
            telemetry.rate_spd,
            telemetry.beat_error_ms,
            telemetry.amplitude_deg,
            telemetry.snr_db,
            telemetry.confidence,
            len(beats),
        )
        return telemetry

    analyze = process_window
    process = process_window

    @classmethod
    def _estimate_oscillation_period(
        cls,
        beat_times: NDArray[np.float64],
        t_osc: NDArray[np.float64],
        nominal_osc_period: float,
    ) -> float:
        """Compute robust oscillation period using OLS slope across even/odd oscillation tracks."""
        if len(t_osc) == 0:
            return nominal_osc_period

        med_osc = float(np.median(t_osc))
        mad_osc = float(np.median(np.abs(t_osc - med_osc))) + 1e-12
        inliers = np.abs(t_osc - med_osc) <= max(4.0 * 1.4826 * mad_osc, 0.002 * nominal_osc_period)

        if np.all(inliers) and len(beat_times) >= 6:
            even_beats = beat_times[0::2]
            odd_beats = beat_times[1::2]
            slope_even = cls._centered_ols_slope(even_beats)
            slope_odd = cls._centered_ols_slope(odd_beats)
            if slope_even > 0.0 and slope_odd > 0.0:
                return 0.5 * (slope_even + slope_odd)

        valid_osc = t_osc[inliers] if np.any(inliers) else t_osc
        return cls._trimmed_mean(valid_osc)

    @staticmethod
    def _centered_ols_slope(times: NDArray[np.float64]) -> float:
        """Compute centered OLS slope of timestamps against oscillation index k = 0, 1, ..."""
        n = len(times)
        if n < 2:
            return 0.0
        k = np.arange(n, dtype=np.float64)
        k_c = k - np.mean(k)
        t_c = times - np.mean(times)
        denom = float(np.dot(k_c, k_c))
        if denom <= 1e-18:
            return 0.0
        return float(np.dot(k_c, t_c) / denom)

    @staticmethod
    def _trimmed_mean(values: NDArray[np.float64], trim_fraction: float = 0.2) -> float:
        """Compute a symmetric trimmed mean of a 1D array."""
        arr = np.sort(np.asarray(values, dtype=np.float64).ravel())
        n = len(arr)
        if n == 0:
            return 0.0
        if n < 5:
            return float(np.mean(arr))
        k = int(np.floor(n * trim_fraction))
        if 2 * k >= n:
            return float(np.median(arr))
        return float(np.mean(arr[k : n - k]))

    @staticmethod
    def _parabolic_peak_interp(arr: NDArray[np.float64], idx: int) -> float:
        """Sub-sample parabolic interpolation around a discrete local maximum."""
        if idx <= 0 or idx >= len(arr) - 1:
            return float(idx)
        y0 = float(arr[idx - 1])
        y1 = float(arr[idx])
        y2 = float(arr[idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            return float(idx)
        delta = 0.5 * (y0 - y2) / denom
        return float(idx) + float(np.clip(delta, -0.5, 0.5))

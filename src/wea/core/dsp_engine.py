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
    DEFAULT_LOWCUT_HZ: float = 3000.0
    DEFAULT_HIGHCUT_HZ: float = 14000.0
    DEFAULT_HIGHPASS_HZ: float = DEFAULT_LOWCUT_HZ
    DEFAULT_FILTER_ORDER: int = 4
    REFRACTORY_FACTOR: float = 0.70
    MIN_AMPLITUDE_DEG: float = 100.0
    MAX_AMPLITUDE_DEG: float = 340.0
    CARRIER_PEAK_ALIGNMENT_SEC: float = 17.5e-6
    SHOCK_MAD_MULTIPLIER: float = 12.0
    SHOCK_MIN_DURATION_MS: float = 35.0
    AGC_TARGET_PEAK: float = 1.0
    AGC_NOISE_GATE_FLOOR: float = 1e-5
    AGC_MIN_GAIN: float = 0.05
    AGC_MAX_GAIN: float = 500.0

    def __init__(
        self,
        config: WatchConfig | None = None,
        bph: int = 21600,
        lift_angle: float = 52.0,
        sample_rate: float = 48000.0,
        highpass_cutoff_hz: float = DEFAULT_LOWCUT_HZ,
        highcut_hz: float = DEFAULT_HIGHCUT_HZ,
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
            highpass_cutoff_hz: Butterworth bandpass lower cutoff frequency in Hz (default 3000 Hz).
            highcut_hz: Butterworth bandpass upper cutoff frequency in Hz (default 14000 Hz).
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
        self.highcut_hz: float = float(highcut_hz)
        self.filter_order: int = int(filter_order)
        self.drift_estimator: SampleRateDriftEstimator | None = drift_estimator
        self.p1_threshold_ratio: float = float(p1_threshold_ratio)

        self._sos: NDArray[np.float64] = self._design_bandpass_sos(
            float(self.config.sample_rate),
            self.highpass_cutoff_hz,
            self.highcut_hz,
            self.filter_order,
        )
        self._sos_zi: NDArray[np.float64] | None = None
        self._agc_env_level: float | None = None
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
        """Reset the stateful streaming SOS filter state ``zi`` and AGC tracker state."""
        self._sos_zi = None
        self._agc_env_level = None
        logger.debug("DSPEngine streaming SOS filter and AGC state reset")

    @classmethod
    def _design_bandpass_sos(
        cls,
        sample_rate: float,
        lowcut_hz: float = DEFAULT_LOWCUT_HZ,
        highcut_hz: float = DEFAULT_HIGHCUT_HZ,
        order: int = DEFAULT_FILTER_ORDER,
    ) -> NDArray[np.float64]:
        """Design a 4th-order Butterworth bandpass filter (3000–14000 Hz) in SOS format."""
        nyquist = 0.5 * float(sample_rate)
        low = float(np.clip(lowcut_hz / nyquist, 1e-4, 0.88))
        high = float(np.clip(highcut_hz / nyquist, low + 0.02, 0.96))
        return np.asarray(
            butter(order, [low, high], btype="bandpass", output="sos"),
            dtype=np.float64,
        )

    @classmethod
    def _design_highpass_sos(
        cls,
        sample_rate: float,
        cutoff_hz: float = DEFAULT_LOWCUT_HZ,
        order: int = DEFAULT_FILTER_ORDER,
    ) -> NDArray[np.float64]:
        """Design the escapement bandpass SOS filter (alias preserved for compatibility)."""
        return cls._design_bandpass_sos(
            sample_rate=sample_rate,
            lowcut_hz=cutoff_hz,
            highcut_hz=cls.DEFAULT_HIGHCUT_HZ,
            order=order,
        )

    def filter_signal(
        self,
        raw_signal: NDArray[np.float64],
        streaming: bool = False,
        fs: float | None = None,
    ) -> NDArray[np.float64]:
        """Filter raw audio with a 4th-order Butterworth bandpass filter (3000–14000 Hz).

        Supports zero-phase ``sosfiltfilt`` for complete windows and stateful
        ``sosfilt`` (with ``zi`` state preservation) for continuous streaming chunks.

        Args:
            raw_signal: 1D array of raw audio samples.
            streaming: If True, apply causal stateful ``sosfilt`` using ``self._sos_zi``.
                If False, apply zero-phase ``sosfiltfilt``.
            fs: Optional sampling rate override in Hz.

        Returns:
            Bandpass-filtered 1D float64 signal array.
        """
        sig = np.asarray(raw_signal, dtype=np.float64).ravel()
        if len(sig) == 0:
            return sig.copy()

        target_fs = float(fs) if fs is not None else float(self.config.sample_rate)
        if abs(target_fs - float(self.config.sample_rate)) > 1.0:
            sos = self._design_bandpass_sos(
                target_fs,
                self.highpass_cutoff_hz,
                self.highcut_hz,
                self.filter_order,
            )
        else:
            sos = self._sos

        if streaming:
            if self._sos_zi is None:
                zi_base = np.asarray(sosfilt_zi(sos), dtype=np.float64)
                self._sos_zi = zi_base * float(sig[0])
            filtered, self._sos_zi = sosfilt(sos, sig, zi=self._sos_zi)
            return np.asarray(filtered, dtype=np.float64)

        return np.asarray(sosfiltfilt(sos, sig), dtype=np.float64)

    @staticmethod
    def _estimate_clean_beat_peak(envelope: NDArray[np.float64], fs_eff: float) -> float:
        """Estimate the normal escapement beat peak level across 220 ms blocks.

        Uses the 75th percentile of 220 ms block maxima (so partially silent buffers
        do not collapse the beat peak reference to the noise floor, while isolated
        macro-shocks in 1–2 blocks do not inflate it).
        """
        n = len(envelope)
        if n == 0:
            return 0.0
        block_len = max(16, int(round(0.220 * fs_eff)))
        if n < block_len * 2:
            return float(np.percentile(envelope, 98.0))
        n_blocks = n // block_len
        reshaped = envelope[: n_blocks * block_len].reshape(n_blocks, block_len)
        block_maxima = np.max(reshaped, axis=1)
        return float(np.percentile(block_maxima, 75.0))

    @staticmethod
    def _find_contiguous_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
        """Return ``(start_idx, end_idx)`` half-open intervals for contiguous True regions."""
        if len(mask) == 0 or not np.any(mask):
            return []
        padded = np.empty(len(mask) + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = mask.view(np.int8)
        padded[-1] = 0
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def blank_transients(
        self,
        filtered_signal: NDArray[np.float64],
        fs: float | None = None,
        raw_signal: NDArray[np.float64] | None = None,
        mad_multiplier: float = SHOCK_MAD_MULTIPLIER,
        min_duration_ms: float = SHOCK_MIN_DURATION_MS,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_], float]:
        """Detect and blank sustained macro-shocks (> 35 ms) without touching narrow watch beats.

        Genuine horological impulses are brief (< 12 ms). This blanker ONLY suppresses
        segments where:
            1. The smoothed block energy envelope remains continuously saturated above
               threshold (``med + 12 * mad`` and ``0.5 * clean_peak``) for a sustained
               duration ``>= min_duration_ms`` (default 35.0 ms), OR
            2. A 100 ms segment exhibits > 10x the median peak energy of surrounding
               active beat segments and contains a sustained burst (> 15 ms).

        Args:
            filtered_signal: 1D bandpass-filtered audio signal.
            fs: Optional effective sampling rate in Hz.
            raw_signal: Optional raw signal buffer (unused for narrow impulses).
            mad_multiplier: MAD threshold multiplier for shock detection (default 12.0).
            min_duration_ms: Minimum sustained macro-shock duration in ms (default 35.0 ms).

        Returns:
            Tuple of ``(conditioned_signal, blank_mask, blanked_fraction)``.
        """
        _ = raw_signal
        sig = np.asarray(filtered_signal, dtype=np.float64).ravel()
        n = len(sig)
        if n == 0:
            return sig.copy(), np.zeros(0, dtype=np.bool_), 0.0

        fs_eff = float(fs) if fs is not None else self._resolve_effective_fs()
        # 6 ms energy-bridging envelope to merge carrier oscillations inside a continuous bump
        # while keeping genuine watch beats (< 10 ms) narrow.
        short_win = max(5, int(round(0.006 * fs_eff)))
        short_kernel = np.ones(short_win, dtype=np.float64) / float(short_win)
        short_env = np.convolve(np.abs(sig), short_kernel, mode="same")

        med = float(np.median(short_env))
        mad = float(np.median(np.abs(short_env - med))) + 1e-12
        clean_peak = self._estimate_clean_beat_peak(short_env, fs_eff)

        # Enforce that narrow watch impulses (< 15 ms) are NEVER blanked
        effective_min_ms = max(15.0, float(min_duration_ms))
        min_shock_samples = max(4, int(round((effective_min_ms / 1000.0) * fs_eff)))
        sustained_thresh = max(med + mad_multiplier * mad, 0.50 * clean_peak)

        blank_mask = np.zeros(n, dtype=np.bool_)

        # 1. Sustained continuous saturation (> 35 ms continuous block energy)
        for s_idx, e_idx in self._find_contiguous_runs(short_env > sustained_thresh):
            if (e_idx - s_idx) >= min_shock_samples:
                blank_mask[s_idx:e_idx] = True

        # 2. Macro 100 ms segment with > 10x the median peak energy of surrounding beats,
        # blanking only sustained bursts (> 15 ms) inside that shocked 100 ms segment.
        seg_len = max(32, int(round(0.100 * fs_eff)))
        if n >= seg_len * 4 and clean_peak > 0.0:
            n_segs = n // seg_len
            seg_reshaped = short_env[: n_segs * seg_len].reshape(n_segs, seg_len)
            seg_peak_energies = np.max(seg_reshaped, axis=1) ** 2
            active_energies = seg_peak_energies[seg_peak_energies > (med + 3.0 * mad) ** 2]
            if len(active_energies) >= 2:
                median_beat_energy = float(np.median(active_energies))
                min_burst_samples = max(4, int(round(0.015 * fs_eff)))
                for seg_idx, seg_energy in enumerate(seg_peak_energies):
                    if seg_energy > 10.0 * median_beat_energy:
                        b0 = seg_idx * seg_len
                        b1 = min(n, b0 + seg_len)
                        sub_runs = self._find_contiguous_runs(short_env[b0:b1] > sustained_thresh)
                        for rs, re in sub_runs:
                            if (re - rs) >= min_burst_samples:
                                blank_mask[b0 + rs : b0 + re] = True

        if not np.any(blank_mask):
            return sig.copy(), blank_mask, 0.0

        # Expand blanked intervals by a 6 ms guard margin to suppress IIR filter edge ringing
        guard_samples = max(4, int(round(0.006 * fs_eff)))
        expanded_mask = np.zeros(n, dtype=np.bool_)
        for s_idx, e_idx in self._find_contiguous_runs(blank_mask):
            g0 = max(0, s_idx - guard_samples)
            g1 = min(n, e_idx + guard_samples)
            expanded_mask[g0:g1] = True

        conditioned = sig.copy()
        conditioned[expanded_mask] = 0.0

        # Smooth 1 ms raised-cosine taper immediately outside blanked regions
        taper_len = max(4, int(round(0.001 * fs_eff)))
        for s_idx, e_idx in self._find_contiguous_runs(expanded_mask):
            left_start = max(0, s_idx - taper_len)
            if s_idx > left_start:
                w_left = 0.5 * (
                    1.0 + np.cos(np.linspace(0.0, np.pi, s_idx - left_start, dtype=np.float64))
                )
                conditioned[left_start:s_idx] *= w_left
            right_end = min(n, e_idx + taper_len)
            if right_end > e_idx:
                w_right = 0.5 * (
                    1.0 - np.cos(np.linspace(0.0, np.pi, right_end - e_idx, dtype=np.float64))
                )
                conditioned[e_idx:right_end] *= w_right

        blanked_fraction = float(np.mean(expanded_mask))
        logger.info(
            "Transient shock blanking suppressed %.2f%% of window (%d/%d samples)",
            blanked_fraction * 100.0,
            int(np.sum(expanded_mask)),
            n,
        )
        return conditioned, expanded_mask, blanked_fraction

    def apply_agc(
        self,
        signal: NDArray[np.float64],
        fs: float | None = None,
        streaming: bool = False,
        target_peak: float = AGC_TARGET_PEAK,
        noise_gate_floor: float = AGC_NOISE_GATE_FLOOR,
    ) -> NDArray[np.float64]:
        """Apply fast-attack / slow-release Adaptive Gain Control (AGC) with noise-gate floor.

        Tracks the peak envelope over 10 ms sub-blocks using fast attack (to prevent
        clipping on loud vintage movements) and slow release (to boost quiet titanium/diver
        cases without distorting intra-beat P1/P2/P3 relative amplitudes). Holds unity gain
        when the signal lies below ``noise_gate_floor`` or lacks impulsive beats above the
        ambient noise floor.

        Args:
            signal: 1D conditioned audio array.
            fs: Optional effective sampling rate in Hz.
            streaming: Whether to persist envelope tracker state across calls.
            target_peak: Target normalized peak amplitude (default 1.0).
            noise_gate_floor: Minimum peak amplitude required to engage AGC boost.

        Returns:
            Gain-normalized 1D float64 signal array.
        """
        sig = np.asarray(signal, dtype=np.float64).ravel()
        n = len(sig)
        if n == 0:
            return sig.copy()

        fs_eff = float(fs) if fs is not None else self._resolve_effective_fs()
        abs_sig = np.abs(sig)
        clean_peak = self._estimate_clean_beat_peak(abs_sig, fs_eff)

        med = float(np.median(abs_sig))
        mad = float(np.median(np.abs(abs_sig - med))) + 1e-12
        robust_sigma = 1.4826 * mad

        # Noise gate: do not amplify pure silence or stationary noise when no watch is present
        if clean_peak < noise_gate_floor or (
            robust_sigma > 1e-9 and clean_peak < med + 4.5 * robust_sigma
        ):
            return sig.copy()

        sub_len = max(16, int(round(0.010 * fs_eff)))
        n_blocks = max(1, (n + sub_len - 1) // sub_len)

        env_level = (
            self._agc_env_level
            if (streaming and self._agc_env_level is not None and self._agc_env_level > 0.0)
            else clean_peak
        )

        alpha_attack = 0.35
        alpha_release = 0.004
        min_hold_level = max(noise_gate_floor, 0.85 * clean_peak)

        block_gains = np.empty(n_blocks, dtype=np.float64)
        block_centers = np.empty(n_blocks, dtype=np.float64)

        for b_idx in range(n_blocks):
            b0 = b_idx * sub_len
            b1 = min(n, b0 + sub_len)
            block_peak = float(np.max(abs_sig[b0:b1])) if b1 > b0 else 0.0
            if block_peak > env_level:
                env_level = alpha_attack * block_peak + (1.0 - alpha_attack) * env_level
            else:
                env_level = max(min_hold_level, (1.0 - alpha_release) * env_level)

            gain = target_peak / max(env_level, noise_gate_floor)
            block_gains[b_idx] = float(np.clip(gain, self.AGC_MIN_GAIN, self.AGC_MAX_GAIN))
            block_centers[b_idx] = 0.5 * float(b0 + b1 - 1)

        if streaming:
            self._agc_env_level = float(env_level)

        if n_blocks == 1:
            return sig * block_gains[0]

        sample_idx = np.arange(n, dtype=np.float64)
        smooth_gain = np.interp(sample_idx, block_centers, block_gains)
        return np.asarray(sig * smooth_gain, dtype=np.float64)

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
        """Detect tick/tock peak indices using robust median+MAD, 0.70*T0 dead-time, and phase-comb tracking.

        Args:
            macro_env: Smoothed macro-beat Hilbert amplitude envelope (~1000 Hz bandwidth).
            bph: Nominal beats per hour.
            fs_eff: Effective sampling rate in Hz.

        Returns:
            List of sample indices corresponding to dominant beat peaks.
        """
        n_samples = len(macro_env)
        if n_samples == 0:
            return []

        t0 = 3600.0 / float(bph)
        nominal_samples = t0 * fs_eff
        refractory_samples = max(1, int(round(self.REFRACTORY_FACTOR * nominal_samples)))
        pre_samples = int(round(0.012 * fs_eff))
        post_samples = int(round(0.008 * fs_eff))

        med = float(np.median(macro_env))
        mad = float(np.median(np.abs(macro_env - med))) + 1e-12
        robust_sigma = 1.4826 * mad
        clean_peak = self._estimate_clean_beat_peak(macro_env, fs_eff)

        adaptive_thresh = max(
            med + 5.5 * robust_sigma,
            med + 0.22 * (clean_peak - med),
        )

        peaks: list[int] = []
        i = pre_samples
        limit = n_samples - post_samples
        comb_tol = int(round(0.15 * nominal_samples))

        while i < limit:
            # If we have a previous beat, first check the phase-locked ±15% comb window around
            # prev_peak + T0 so a mid-cycle noise spike at 0.70*T0 cannot steal the trigger.
            if peaks:
                expected_idx = peaks[-1] + int(round(nominal_samples))
                c0 = max(i, expected_idx - comb_tol)
                c1 = min(limit, expected_idx + comb_tol)
                if c1 > c0:
                    comb_max_idx = c0 + int(np.argmax(macro_env[c0:c1]))
                    if macro_env[comb_max_idx] > adaptive_thresh:
                        peaks.append(comb_max_idx)
                        i = comb_max_idx + refractory_samples
                        continue

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
        """Isolate unlocking (P1), impulse (P2), and drop (P3) within a single beat window."""
        sub_fine, sub_wave, start_idx, fs_eff = args
        if len(sub_fine) < 10:
            return None

        max_fine = float(np.max(sub_fine))
        if max_fine <= 0.0:
            return None

        dom_idx = int(np.argmax(sub_fine))
        dom_exact = self._parabolic_peak_interp(sub_fine, dom_idx)

        # Bound the companion impulse search to the physical horological lift-time range
        # [155 deg .. 340 deg] so background noise 10 ms ahead of the beat cannot fake P1.
        t0 = 3600.0 / float(max(self.config.bph, 3600))
        lift_deg = float(self.config.lift_angle)
        dt_min = (
            (t0 / np.pi) * float(np.arcsin(np.clip(lift_deg / (2.0 * 340.0), 0.01, 0.99))) * 0.80
        )
        dt_max = (
            (t0 / np.pi) * float(np.arcsin(np.clip(lift_deg / (2.0 * 155.0), 0.01, 0.99))) * 1.25
        )
        n_min = max(2, int(round(dt_min * fs_eff)))
        n_max = max(n_min + 4, int(round(dt_max * fs_eff)))

        baseline_slice = sub_fine[: max(8, min(len(sub_fine) // 4, max(8, dom_idx - n_max)))]
        local_baseline = float(np.percentile(baseline_slice, 25))
        effective_peak = max_fine - local_baseline
        if effective_peak <= 0.0:
            return None

        onset_thresh = local_baseline + self.p1_threshold_ratio * effective_peak

        pre_start = max(0, dom_idx - n_max)
        pre_end = max(pre_start, dom_idx - n_min)
        above_pre = (
            np.where(sub_fine[pre_start : pre_end + 1] >= onset_thresh)[0]
            if pre_end > pre_start
            else np.empty(0, dtype=np.int64)
        )

        post_start = min(len(sub_fine) - 1, dom_idx + n_min)
        post_end = min(len(sub_fine) - 1, dom_idx + n_max)
        above_post = (
            np.where(sub_fine[post_start : post_end + 1] >= onset_thresh)[0]
            if post_end > post_start
            else np.empty(0, dtype=np.int64)
        )

        if len(above_pre) > 0:
            onset_idx = pre_start + int(above_pre[0])
            p1_search_end = min(onset_idx + int(round(0.0005 * fs_eff)), dom_idx)
            if p1_search_end > onset_idx:
                p1_rel_idx = onset_idx + int(np.argmax(sub_fine[onset_idx : p1_search_end + 1]))
                p1_exact_rel = self._parabolic_peak_interp(sub_fine, p1_rel_idx)
            else:
                p1_rel_idx = onset_idx
                p1_exact_rel = float(onset_idx)
            p3_rel_idx = dom_idx
            p3_exact_rel = dom_exact
        elif len(above_post) > 0:
            # Unlock/impulse-dominant piezo beat: dominant peak is P1, drop peak P3 follows in [post_start, post_end]
            p1_rel_idx = dom_idx
            p1_exact_rel = dom_exact
            p3_rel_idx = post_start + int(np.argmax(sub_fine[post_start : post_end + 1]))
            p3_exact_rel = self._parabolic_peak_interp(sub_fine, p3_rel_idx)
        else:
            # Fallback: search within [pre_start, dom_idx]
            fallback_above = np.where(sub_fine[pre_start : dom_idx + 1] >= onset_thresh)[0]
            if len(fallback_above) > 0:
                onset_idx = pre_start + int(fallback_above[0])
                p1_search_end = min(onset_idx + int(round(0.0005 * fs_eff)), dom_idx)
                p1_rel_idx = onset_idx + int(np.argmax(sub_fine[onset_idx : p1_search_end + 1]))
                p1_exact_rel = self._parabolic_peak_interp(sub_fine, p1_rel_idx)
            else:
                p1_rel_idx = dom_idx
                p1_exact_rel = dom_exact
            p3_rel_idx = dom_idx
            p3_exact_rel = dom_exact

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
        conditioned, blank_mask, blank_fraction = self.blank_transients(
            filtered,
            fs=fs_eff,
            raw_signal=audio_buffer,
        )
        normalized = self.apply_agc(conditioned, fs=fs_eff, streaming=streaming)
        macro_env, micro_env = self.compute_envelopes(normalized, fs=fs_eff)

        peak_indices = self.detect_macro_beat_peaks(macro_env, bph=active_bph, fs_eff=fs_eff)
        n_samples = len(normalized)
        pre_samples = int(round(0.012 * fs_eff))
        post_samples = int(round(0.008 * fs_eff))

        tasks: list[tuple[NDArray[np.float64], NDArray[np.float64], int, float]] = []
        for pk_idx in peak_indices:
            s0 = max(0, pk_idx - pre_samples)
            s1 = min(n_samples, pk_idx + post_samples)
            if len(blank_mask) == n_samples and np.any(blank_mask[s0:s1]):
                continue
            tasks.append((micro_env[s0:s1], normalized[s0:s1], s0, fs_eff))

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
                filtered_signal=normalized,
                effective_fs=fs_eff,
            )

        p1_times = np.array([b.timestamp_p1 for b in beats], dtype=np.float64)
        p3_times = np.array([b.timestamp_p3 for b in beats], dtype=np.float64)
        beat_times = 0.35 * p1_times + 0.65 * p3_times

        t0 = 3600.0 / float(active_bph)
        beat_indices = self._assign_beat_indices(beat_times, t0)

        # Full oscillations cancel asymmetric tick/tock beat error even across blanked beat gaps
        t_osc = beat_times[2:] - beat_times[:-2]
        mean_t_osc = self._estimate_oscillation_period(
            beat_times,
            t_osc,
            2.0 * t0,
            beat_indices=beat_indices,
        )

        # Rate (s/d): 86400 * ((2 * T0 - T_osc_bar) / (2 * T0))
        rate_spd = self.SECONDS_PER_DAY * ((2.0 * t0 - mean_t_osc) / (2.0 * t0))

        # Beat Error (ms) from alternating tick->tock and tock->tick single-beat half-periods,
        # clamping/rejecting outlier intervals that deviate by more than ±15% from T0.
        half_periods = np.diff(beat_times)
        index_steps = np.diff(beat_indices)
        single_step_mask = (index_steps == 1) & (np.abs(half_periods - t0) <= 0.15 * t0)
        if np.any(single_step_mask):
            start_parities = beat_indices[:-1] % 2
            even_intervals = half_periods[single_step_mask & (start_parities == 0)]
            odd_intervals = half_periods[single_step_mask & (start_parities == 1)]
        else:
            valid_hp = half_periods[np.abs(half_periods - t0) <= 0.15 * t0]
            even_intervals = valid_hp[0::2]
            odd_intervals = valid_hp[1::2]

        if len(even_intervals) > 0 and len(odd_intervals) > 0:
            mean_even = self._trimmed_mean(even_intervals)
            mean_odd = self._trimmed_mean(odd_intervals)
            beat_error_ms = abs(mean_even - mean_odd) * 1000.0
        else:
            beat_error_ms = 0.0

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
        base_confidence = float(np.clip(np.exp(-jitter / 0.0005), 0.0, 1.0))
        if blank_fraction > 0.0:
            shock_penalty = float(np.clip(1.0 - max(0.08, 2.5 * blank_fraction), 0.10, 0.92))
            confidence = base_confidence * shock_penalty
        else:
            confidence = base_confidence

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
            filtered_signal=normalized,
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

    @staticmethod
    def _assign_beat_indices(
        beat_times: NDArray[np.float64], nominal_beat_period: float
    ) -> NDArray[np.int64]:
        """Assign integer beat indices m_i using ±15% inlier period tracking."""
        n = len(beat_times)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if n == 1:
            return np.zeros(1, dtype=np.int64)

        diffs = np.diff(beat_times)
        single_candidates = diffs[np.abs(diffs - nominal_beat_period) <= 0.15 * nominal_beat_period]
        ref_period = (
            float(np.median(single_candidates))
            if len(single_candidates) >= 2
            else float(nominal_beat_period)
        )
        steps = np.maximum(1, np.rint(diffs / ref_period).astype(np.int64))
        indices = np.empty(n, dtype=np.int64)
        indices[0] = 0
        indices[1:] = np.cumsum(steps)
        return indices

    @classmethod
    def _robust_track_osc_slope(
        cls,
        k_osc: NDArray[np.float64],
        times: NDArray[np.float64],
        nominal_osc_period: float,
    ) -> float:
        """Compute outlier-trimmed OLS oscillation period using phase residuals t_k - k * (2*T0)."""
        n = len(times)
        if n < 2 or len(k_osc) != n:
            return 0.0

        # Phase residuals relative to nominal oscillation period
        residuals = times - k_osc * nominal_osc_period
        dk = np.diff(k_osc)
        dr = np.diff(residuals)
        valid_steps = dk > 0
        if not np.any(valid_steps):
            return 0.0

        step_slopes = dr[valid_steps] / dk[valid_steps]
        # Reject steps deviating by > ±15% of nominal oscillation period
        in_bounds = np.abs(step_slopes) <= 0.15 * nominal_osc_period
        if not np.any(in_bounds):
            return 0.0

        med_slope = float(np.median(step_slopes[in_bounds]))
        detrended = residuals - med_slope * (k_osc - float(np.median(k_osc)))
        med_det = float(np.median(detrended))
        mad_det = float(np.median(np.abs(detrended - med_det))) + 1e-12
        tol = max(3.5 * 1.4826 * mad_det, 0.0015 * nominal_osc_period)
        inlier_pts = np.abs(detrended - med_det) <= tol

        if int(np.sum(inlier_pts)) >= 2:
            slope = cls._ols_slope_with_indices(k_osc[inlier_pts], times[inlier_pts])
            if abs(slope - nominal_osc_period) <= 0.15 * nominal_osc_period:
                return slope

        return float(nominal_osc_period + med_slope)

    @classmethod
    def _estimate_oscillation_period(
        cls,
        beat_times: NDArray[np.float64],
        t_osc: NDArray[np.float64],
        nominal_osc_period: float,
        beat_indices: NDArray[np.int64] | None = None,
    ) -> float:
        """Compute robust oscillation period using phase-residual outlier-trimmed OLS regression."""
        if len(t_osc) == 0:
            return nominal_osc_period

        if beat_indices is None or len(beat_indices) != len(beat_times):
            beat_indices = cls._assign_beat_indices(beat_times, 0.5 * nominal_osc_period)

        even_mask = (beat_indices % 2) == 0
        odd_mask = ~even_mask

        even_times = beat_times[even_mask]
        even_k = (beat_indices[even_mask] // 2).astype(np.float64)
        odd_times = beat_times[odd_mask]
        odd_k = ((beat_indices[odd_mask] - 1) // 2).astype(np.float64)

        slope_even = cls._robust_track_osc_slope(even_k, even_times, nominal_osc_period)
        slope_odd = cls._robust_track_osc_slope(odd_k, odd_times, nominal_osc_period)

        if slope_even > 0.0 and slope_odd > 0.0:
            return 0.5 * (slope_even + slope_odd)
        if slope_even > 0.0:
            return slope_even
        if slope_odd > 0.0:
            return slope_odd

        bounded_osc = t_osc[np.abs(t_osc - nominal_osc_period) <= 0.15 * nominal_osc_period]
        if len(bounded_osc) == 0:
            return nominal_osc_period
        return cls._trimmed_mean(bounded_osc)

    @staticmethod
    def _ols_slope_with_indices(
        indices: NDArray[np.float64],
        times: NDArray[np.float64],
    ) -> float:
        """Compute centered OLS slope of timestamps against explicit oscillation indices k."""
        n = len(times)
        if n < 2 or len(indices) != n:
            return 0.0
        k_c = indices - np.mean(indices)
        t_c = times - np.mean(times)
        denom = float(np.dot(k_c, k_c))
        if denom <= 1e-18:
            return 0.0
        return float(np.dot(k_c, t_c) / denom)

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

"""High-precision DSP pipeline for Swiss lever escapement impulse isolation and telemetry."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from wea.core.models import BeatEvent, TelemetryResult, WatchConfig
from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class EscapeDSPPipeline:
    """Digital Signal Processing pipeline for horological acoustic analysis.

    Implements:
        a) Cascaded zero-phase Butterworth bandpass filtering (4 kHz - 18 kHz) via SOS.
        b) Dual-scale energy envelope extraction (coarse RMS for global beat segmentation
           and fine sub-millisecond Hilbert/RMS envelope to isolate P1, P2, and P3).
        c) High-precision mathematical derivation of Rate (s/d), Beat Error (ms),
           and Balance Wheel Amplitude (deg).
    """

    SECONDS_PER_DAY: float = 86400.0
    LOWCUT_HZ: float = 4000.0
    HIGHCUT_HZ: float = 18000.0
    FILTER_ORDER: int = 4

    def __init__(
        self,
        config: WatchConfig,
        p1_threshold_ratio: float = 0.18,
    ) -> None:
        """Initialize the DSP pipeline for a given watch configuration.

        Args:
            config: WatchConfig instance with BPH, lift_angle, and sample_rate.
            p1_threshold_ratio: Relative envelope threshold (fraction of peak envelope)
                used to detect P1 onset inside each beat window.
        """
        self.config = config
        self.p1_threshold_ratio = p1_threshold_ratio
        self._sos = self._design_bandpass_sos()
        logger.info(
            "EscapeDSPPipeline initialized with sample_rate=%.1f Hz, "
            "nominal_beat_period=%.6f s, lift_angle=%.2f deg",
            self.config.sample_rate,
            self.config.nominal_beat_period,
            self.config.lift_angle,
        )

    def _design_bandpass_sos(self) -> np.ndarray:
        """Design a Butterworth bandpass filter in Second-Order Sections (SOS) format."""
        nyquist = 0.5 * self.config.sample_rate
        low = max(self.LOWCUT_HZ / nyquist, 1e-4)
        high = min(self.HIGHCUT_HZ / nyquist, 0.999)
        return butter(self.FILTER_ORDER, [low, high], btype="bandpass", output="sos")

    def max_lift_time_s(self, min_amplitude_deg: float = 140.0) -> float:
        """Longest plausible P1 -> P3 lift time (s), reached at ``min_amplitude_deg``.

        ``t_lift = arcsin(LA / (2 * A)) / (pi * f_osc)``, with a 10 % margin.
        """
        f_osc = self.config.nominal_oscillation_frequency
        ratio = float(np.clip(self.config.lift_angle / (2.0 * min_amplitude_deg), 1e-6, 0.999))
        return 1.10 * float(np.arcsin(ratio) / (np.pi * f_osc))

    def filter_signal(self, raw_signal: np.ndarray) -> np.ndarray:
        """Apply zero-phase forward-backward SOS Butterworth \
            bandpass filter."""
        logger.debug(
            "Filtering raw signal (%d samples) with bandpass %.1f Hz - %.1f Hz (order %d)",
            len(raw_signal),
            self.LOWCUT_HZ,
            self.HIGHCUT_HZ,
            self.FILTER_ORDER,
        )
        if len(raw_signal) == 0:
            logger.warning("filter_signal received an empty signal array")
            return raw_signal.copy()
        return sosfiltfilt(self._sos, raw_signal)

    def compute_dual_envelopes(self, filtered_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract coarse global RMS envelope and fine sub-millisecond
        envelope.

        Args:
            filtered_signal: Bandpass-filtered 1D acoustic waveform.

        Returns:
            Tuple of (coarse_rms_envelope, fine_hilbert_rms_envelope).
        """
        fs = self.config.sample_rate
        energy = filtered_signal**2

        # Coarse RMS window (~4 ms) to merge P1+P2+P3 into a single macro beat lobe
        coarse_win = max(int(round(0.004 * fs)), 3)
        coarse_kernel = np.ones(coarse_win, dtype=np.float64) / coarse_win
        coarse_env = np.sqrt(np.convolve(energy, coarse_kernel, mode="same"))

        # Fine sub-millisecond analytic envelope via Hilbert transform smoothed with a ~0.12 ms window
        analytic = hilbert(filtered_signal)
        inst_amp = np.abs(analytic)
        fine_win = max(int(round(0.00012 * fs)), 3)
        fine_kernel = np.ones(fine_win, dtype=np.float64) / fine_win
        fine_env = np.convolve(inst_amp, fine_kernel, mode="same")

        return coarse_env, fine_env

    def extract_beats(self, filtered_signal: np.ndarray) -> list[BeatEvent]:
        """Detect beat windows and isolate P1 (unlock), P2 (impulse), and P3 (drop) timestamps."""
        fs = self.config.sample_rate
        coarse_env, fine_env = self.compute_dual_envelopes(filtered_signal)

        n_samples = len(filtered_signal)
        nominal_samples = int(round(self.config.nominal_beat_period * fs))
        min_distance = int(0.65 * nominal_samples)

        # Global threshold using robust percentile statistics
        noise_floor = float(np.percentile(coarse_env, 50))
        peak_level = float(np.percentile(coarse_env, 99.5))
        coarse_threshold = noise_floor + 0.25 * (peak_level - noise_floor)

        # Find coarse macro-beat centers
        coarse_peaks: list[int] = []
        i = min_distance // 2
        while i < n_samples - min_distance // 2:
            if coarse_env[i] > coarse_threshold:
                win_end = min(i + min_distance, n_samples)
                local_max_idx = i + int(np.argmax(coarse_env[i:win_end]))
                coarse_peaks.append(local_max_idx)
                i = local_max_idx + min_distance
            else:
                i += 1

        # A beat spans a full lift time (P1 -> P3), up to ~20 ms at 18000 BPH and low
        # amplitude, plus the decay of the sound. The coarse peak can sit on P1 or P3.
        half_window_s = self.max_lift_time_s() + 0.0035
        pre_samples = int(round(half_window_s * fs))
        post_samples = int(round(half_window_s * fs))

        beats: list[BeatEvent] = []
        for c_idx in coarse_peaks:
            start_idx = max(0, c_idx - pre_samples)
            end_idx = min(n_samples, c_idx + post_samples)
            if end_idx - start_idx < 10:
                continue

            sub_fine = fine_env[start_idx:end_idx]
            sub_wave = filtered_signal[start_idx:end_idx]

            max_fine = float(np.max(sub_fine))
            if max_fine <= 0.0:
                continue

            # Baseline noise ahead of the beat
            local_baseline = float(np.percentile(sub_fine, 15))
            effective_peak = max_fine - local_baseline
            if effective_peak <= 0.0:
                continue

            # P3 is the dominant locking/drop peak near the end of the escapement impulse sequence
            p3_rel_idx = int(np.argmax(sub_fine))
            p3_exact_rel = self._parabolic_peak_interp(sub_fine, p3_rel_idx)

            # P1 is the first local impulse peak after crossing the onset threshold
            onset_thresh = local_baseline + self.p1_threshold_ratio * effective_peak
            above_indices = np.where(sub_fine[: p3_rel_idx + 1] >= onset_thresh)[0]

            if len(above_indices) > 0:
                onset_idx = int(above_indices[0])
                # Search for the local peak of P1 within ~0.6 ms after onset
                p1_search_end = min(onset_idx + int(round(0.0006 * fs)), p3_rel_idx)
                if p1_search_end > onset_idx:
                    p1_rel_idx = onset_idx + int(np.argmax(sub_fine[onset_idx : p1_search_end + 1]))
                    p1_exact_rel = self._parabolic_peak_interp(sub_fine, p1_rel_idx)
                else:
                    p1_rel_idx = onset_idx
                    p1_exact_rel = float(onset_idx)
            else:
                p1_rel_idx = p3_rel_idx
                p1_exact_rel = p3_exact_rel

            # P2 lies between P1 and P3
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

            # Compensate for the intrinsic rise time from impulse start to P1 envelope peak
            # In both synthetic and physical escapements, P1 peak occurs at tau_p1 * 0.25 + filter_group_delay(0)
            # Notice that both P1 and P3 experience the exact same envelope peak delay (~0.045 ms)
            # relative to their impulse timestamps! Thus (t_p3_peak - t_p1_peak) preserves (t_p3 - t_p1)
            # and beat-to-beat differences (t_p1[k+1] - t_p1[k] or t_p3[k+1] - t_p3[k]) cancel constant offsets.
            t_p1 = (start_idx + p1_exact_rel) / fs
            t_p2 = (start_idx + p2_exact_rel) / fs
            t_p3 = (start_idx + p3_exact_rel) / fs

            beats.append(
                BeatEvent(
                    timestamp_p1=float(t_p1),
                    timestamp_p2=float(t_p2),
                    timestamp_p3=float(t_p3),
                    peak_energy=float(max_fine**2),
                    raw_waveform=sub_wave.copy(),
                )
            )

        return beats

    def analyze(self, raw_signal: np.ndarray) -> tuple[TelemetryResult, list[BeatEvent]]:
        """Run the full DSP pipeline on a raw audio buffer and compute horological telemetry."""
        filtered = self.filter_signal(raw_signal)
        beats = self.extract_beats(filtered)

        if len(beats) < 3:
            logger.warning(
                "Insufficient beats detected (%d < 3) for telemetry computation",
                len(beats),
            )
            return (
                TelemetryResult(
                    rate_spd=0.0,
                    beat_error_ms=0.0,
                    amplitude_deg=0.0,
                    confidence=0.0,
                ),
                beats,
            )

        # Use P3 (highest SNR locking impact) and P1 timestamps for beat-to-beat intervals
        p1_times = np.array([b.timestamp_p1 for b in beats], dtype=np.float64)
        p3_times = np.array([b.timestamp_p3 for b in beats], dtype=np.float64)

        # Combine P1 and P3 timestamps (weighted toward P3 due to higher SNR) for ultra-stable beat timing
        beat_times = 0.35 * p1_times + 0.65 * p3_times
        intervals = np.diff(beat_times)

        # Separate even and odd beat intervals
        even_intervals = intervals[0::2]
        odd_intervals = intervals[1::2]

        mean_even = float(np.median(even_intervals))
        mean_odd = float(np.median(odd_intervals))

        # Mean beat period across balanced tick/tock pairs
        measured_beat_period = 0.5 * (mean_even + mean_odd)
        nominal_period = self.config.nominal_beat_period

        # Rate in seconds per day:
        # rate_spd = 86400 * (T_nominal - T_measured) / T_measured
        rate_spd = self.SECONDS_PER_DAY * (
            (nominal_period - measured_beat_period) / measured_beat_period
        )

        # Beat Error in milliseconds (timegrapher convention): intervals alternate
        # P + d / P - d and the beat error is |d|, half the even/odd interval difference.
        beat_error_ms = 0.5 * abs(mean_even - mean_odd) * 1000.0

        # Balance wheel amplitude from the P1 -> P3 lift time t_lift:
        # A = lift_angle / (2 * sin(pi * f_osc * t_lift)), f_osc = 1 / (2 * T_beat)
        # => pi * f_osc * t_lift = pi * t_lift / (2 * T_beat)
        dt_p1_p3_arr = p3_times - p1_times
        median_dt = float(np.median(dt_p1_p3_arr))

        if median_dt > 1e-6:
            sin_arg = np.clip(
                (np.pi * median_dt) / (2.0 * measured_beat_period), 1e-6, np.pi - 1e-6
            )
            sin_val = float(np.sin(sin_arg))
            if sin_val > 1e-6:
                amplitude_deg = float(
                    np.clip(self.config.lift_angle / (2.0 * sin_val), 50.0, 360.0)
                )
            else:
                amplitude_deg = 0.0
        else:
            amplitude_deg = 0.0

        # Compute confidence score from interval consistency
        even_std = float(np.std(even_intervals)) if len(even_intervals) > 1 else 0.0
        odd_std = float(np.std(odd_intervals)) if len(odd_intervals) > 1 else 0.0
        jitter = 0.5 * (even_std + odd_std)
        confidence = float(np.clip(np.exp(-jitter / 0.0005), 0.0, 1.0))

        telemetry = TelemetryResult(
            rate_spd=float(rate_spd),
            beat_error_ms=float(beat_error_ms),
            amplitude_deg=float(amplitude_deg),
            confidence=confidence,
        )
        logger.debug(
            "DSP telemetry computed: rate=%+.2f s/d, BE=%.3f ms, amp=%.1f deg, conf=%.2f (%d beats)",
            telemetry.rate_spd,
            telemetry.beat_error_ms,
            telemetry.amplitude_deg,
            telemetry.confidence,
            len(beats),
        )
        return telemetry, beats

    @staticmethod
    def _parabolic_peak_interp(arr: np.ndarray, idx: int) -> float:
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

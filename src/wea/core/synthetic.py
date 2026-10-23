"""Ground-truth synthetic acoustic signal generator for Swiss lever escapements."""

from __future__ import annotations

import numpy as np

from wea.core.models import WatchConfig
from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class SyntheticEscapeGenerator:
    """Generates realistic synthetic watch escapement acoustic waveforms with exact ground-truth timing.

    Models each beat (tick/tock) as three distinct damped high-frequency harmonic transients:
    - P1: Pallet unlocking impulse.
    - P2: Impulse pin/tooth impact.
    - P3: Escape wheel tooth drop / locking impact.
    """

    SECONDS_PER_DAY: float = 86400.0

    def __init__(self, config: WatchConfig, seed: int | None = 42) -> None:
        """Initialize the generator with a WatchConfig and optional random seed."""
        self.config = config
        self.rng = np.random.default_rng(seed)
        logger.debug(
            "SyntheticEscapeGenerator initialized (bph=%d, lift=%.1f, fs=%d, seed=%s)",
            config.bph,
            config.lift_angle,
            config.sample_rate,
            seed,
        )

    def compute_p1_p3_delay(
        self, amplitude_deg: float, actual_beat_period: float | None = None
    ) -> float:
        """Compute the physical lift time (seconds) between P1 (unlock) and P3 (drop).

        The balance moves as ``theta(t) = A * sin(2 * pi * f_osc * t)``. The impulse pin
        crosses the lift angle ``LA`` symmetrically around the dead point, i.e. from
        ``-LA / 2`` to ``+LA / 2``, so::

            t_lift = 2 * arcsin(LA / (2 * A)) / (2 * pi * f_osc)
                   = arcsin(LA / (2 * A)) / (pi * f_osc)

        With ``f_osc = 1 / (2 * T_beat)`` this is ``(2 * T_beat / pi) * arcsin(LA / (2 * A))``.
        Example: 28800 BPH, A = 280 deg, LA = 52 deg -> t_lift ~= 7.40 ms.

        Args:
            amplitude_deg: Balance amplitude in degrees.
            actual_beat_period: Real beat period in seconds (defaults to the nominal one).

        Returns:
            Lift time ``t_lift`` in seconds.
        """
        t_beat = (
            actual_beat_period
            if actual_beat_period is not None
            else self.config.nominal_beat_period
        )
        ratio = np.clip(self.config.lift_angle / (2.0 * amplitude_deg), 1e-6, 0.999999)
        return float((2.0 * t_beat / np.pi) * np.arcsin(ratio))

    @staticmethod
    def beat_error_half_offset(beat_error_ms: float) -> float:
        """Return the alternating interval offset ``delta`` (seconds) for a beat error.

        Timegrapher convention: beat intervals alternate ``P + delta`` / ``P - delta`` and the
        reported beat error is ``|delta|`` (half the tick/tock interval difference).
        """
        return float(beat_error_ms) * 1e-3

    def generate(
        self,
        duration_s: float = 5.0,
        true_rate_spd: float = 0.0,
        true_beat_error_ms: float = 0.0,
        true_amplitude_deg: float = 270.0,
        snr_db: float | None = 40.0,
    ) -> np.ndarray:
        """Synthesize an acoustic signal with ground-truth horological telemetry parameters.

        Args:
            duration_s: Total signal duration in seconds.
            true_rate_spd: Ground-truth rate deviation in seconds per day (+ fast, - slow).
            true_beat_error_ms: Ground-truth beat error in milliseconds.
            true_amplitude_deg: Ground-truth balance amplitude in degrees.
            snr_db: Signal-to-noise ratio in decibels (None for noiseless).

        Returns:
            1D NumPy array of float64 audio samples.
        """
        fs = self.config.sample_rate
        n_samples = int(round(duration_s * fs))
        signal = np.zeros(n_samples, dtype=np.float64)

        t0 = self.config.nominal_beat_period
        rate_factor = 1.0 + (true_rate_spd / self.SECONDS_PER_DAY)
        mean_beat_period = t0 / rate_factor

        half_be_s = self.beat_error_half_offset(true_beat_error_ms)

        dt_p1_p3 = self.compute_p1_p3_delay(true_amplitude_deg, actual_beat_period=mean_beat_period)
        dt_p1_p2 = 0.28 * dt_p1_p3

        kernel_len = int(0.0025 * fs)
        t_k = np.arange(kernel_len, dtype=np.float64) / fs

        tau_p1 = 0.00015
        tau_p2 = 0.00018
        tau_p3 = 0.00020

        env_p1 = (t_k / (tau_p1 * 0.25)) * np.exp(-t_k / tau_p1)
        env_p1 /= np.max(env_p1) + 1e-12

        env_p2 = (t_k / (tau_p2 * 0.25)) * np.exp(-t_k / tau_p2)
        env_p2 /= np.max(env_p2) + 1e-12

        env_p3 = (t_k / (tau_p3 * 0.25)) * np.exp(-t_k / tau_p3)
        env_p3 /= np.max(env_p3) + 1e-12

        carrier_p1 = 0.6 * np.sin(2.0 * np.pi * 7200.0 * t_k) + 0.4 * np.sin(
            2.0 * np.pi * 12400.0 * t_k
        )
        carrier_p2 = 0.5 * np.sin(2.0 * np.pi * 8500.0 * t_k) + 0.5 * np.sin(
            2.0 * np.pi * 14200.0 * t_k
        )
        carrier_p3 = 0.55 * np.sin(2.0 * np.pi * 9600.0 * t_k) + 0.45 * np.sin(
            2.0 * np.pi * 15500.0 * t_k
        )

        amp_p1 = 0.55
        amp_p2 = 0.45
        amp_p3 = 1.00

        wave_p1 = amp_p1 * env_p1 * carrier_p1
        wave_p2 = amp_p2 * env_p2 * carrier_p2
        wave_p3 = amp_p3 * env_p3 * carrier_p3

        current_p1 = 0.050
        beat_idx = 0

        while current_p1 + dt_p1_p3 + 0.005 < duration_s:
            t_p1 = current_p1
            t_p2 = current_p1 + dt_p1_p2
            t_p3 = current_p1 + dt_p1_p3

            self._add_sub_sample_impulse(signal, t_p1, wave_p1, fs)
            self._add_sub_sample_impulse(signal, t_p2, wave_p2, fs)
            self._add_sub_sample_impulse(signal, t_p3, wave_p3, fs)

            if beat_idx % 2 == 0:
                interval = mean_beat_period + half_be_s
            else:
                interval = mean_beat_period - half_be_s

            current_p1 += interval
            beat_idx += 1

        if snr_db is not None:
            sig_power = np.mean(signal**2)
            if sig_power > 0:
                noise_power = sig_power / (10.0 ** (snr_db / 10.0))
                noise = self.rng.normal(0.0, np.sqrt(noise_power), size=n_samples)
                signal = signal + noise

        return signal

    @staticmethod
    def _add_sub_sample_impulse(
        buffer: np.ndarray,
        timestamp_s: float,
        kernel: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Insert a kernel at fractional sample position using linear interpolation."""
        exact_idx = timestamp_s * sample_rate
        base_idx = int(np.floor(exact_idx))
        frac = exact_idx - base_idx

        k_len = len(kernel)
        end_idx = base_idx + k_len + 1
        if base_idx < 0 or end_idx > len(buffer):
            return

        buffer[base_idx : base_idx + k_len] += (1.0 - frac) * kernel
        buffer[base_idx + 1 : base_idx + k_len + 1] += frac * kernel


def synthesize_escapement(
    bph: int = 21600,
    rate_spd: float = 5.0,
    beat_error_ms: float = 0.3,
    amplitude_deg: float = 290.0,
    lift_angle: float = 51.0,
    seconds: float = 30.0,
    fs: int = 48000,
    snr_db: float = 25.0,
    carrier_hz: float = 6000.0,
    second_mode_ratio: float = 1.32,
    sound_levels: tuple[float, float, float] = (0.45, 0.35, 1.0),
    decay_s: float = 0.0006,
    jitter_us: float = 5.0,
    missing_fraction: float = 0.0,
    hum_level: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a contact-microphone escapement recording with a resonant carrier.

    Each beat has three sounds - unlock P1 at the beat instant, impulse P2 at ``0.45 t_lift``
    and drop P3 at ``t_lift`` - with ``t_lift = arcsin(LA / (2 A)) / (pi f_osc)``. Each sound
    is a two-mode damped burst (``carrier_hz`` and ``second_mode_ratio * carrier_hz``) with a
    0.1 ms attack and random phases, so the envelope ripple differs from beat to beat, as with
    a real case resonance. Beat instants follow ``t_k = t_0 + k P +- d / 2`` (beat error ``d``)
    plus Gaussian jitter.

    Args:
        bph: Beats per hour.
        rate_spd: True rate in s/d (positive = gains, i.e. shorter period).
        beat_error_ms: True beat error ``d`` (intervals alternate ``P + d`` / ``P - d``).
        amplitude_deg: Balance amplitude.
        lift_angle: Lift angle in degrees.
        seconds: Duration.
        fs: Sample rate in Hz.
        snr_db: Burst RMS to white-noise RMS ratio.
        carrier_hz: Main resonance of the escapement sounds.
        second_mode_ratio: Frequency ratio of the second resonance.
        sound_levels: Relative peak levels of P1, P2, P3.
        decay_s: Exponential decay time constant of each sound.
        jitter_us: Standard deviation of the per-beat timing jitter.
        missing_fraction: Fraction of beats removed at random (dropouts).
        hum_level: 50 Hz hum level relative to the burst RMS.
        seed: Random seed.

    Returns:
        ``(signal, onsets_s)`` - the audio and the true P1 instant of every present beat.
    """
    rng = np.random.default_rng(seed)
    period_s = 3600.0 / bph / (1.0 + rate_spd / 86400.0)
    f_osc = bph / 7200.0
    t_lift = float(np.arcsin(lift_angle / (2.0 * amplitude_deg)) / (np.pi * f_osc))

    n_total = int(round(seconds * fs))
    x = np.zeros(n_total + fs, dtype=np.float64)
    burst_len = int(round(8.0 * decay_s * fs)) + 1
    t_b = np.arange(burst_len, dtype=np.float64) / fs
    attack = 1.0 - np.exp(-t_b / 1e-4)
    decay = np.exp(-t_b / decay_s)

    onsets: list[float] = []
    n_beats = int((seconds - 0.1) / period_s)
    for k in range(n_beats):
        t0 = 0.05 + k * period_s + 0.5 * beat_error_ms * 1e-3 * (1.0 if k % 2 == 0 else -1.0)
        t0 += rng.normal(0.0, jitter_us * 1e-6)
        if missing_fraction > 0.0 and rng.random() < missing_fraction:
            continue
        if t0 + t_lift + burst_len / fs >= seconds:
            break
        onsets.append(t0)
        for frac, level in zip((0.0, 0.45, 1.0), sound_levels):
            ph1, ph2 = rng.uniform(0.0, 2.0 * np.pi, size=2)
            burst = (
                level
                * attack
                * decay
                * (
                    np.sin(2.0 * np.pi * carrier_hz * t_b + ph1)
                    + 0.6 * np.sin(2.0 * np.pi * second_mode_ratio * carrier_hz * t_b + ph2)
                )
            )
            pos = (t0 + frac * t_lift) * fs
            i0 = int(np.floor(pos))
            w = pos - i0
            x[i0 : i0 + burst_len] += (1.0 - w) * burst
            x[i0 + 1 : i0 + burst_len + 1] += w * burst

    x = x[:n_total]
    active = x[np.abs(x) > 1e-9]
    rms = float(np.sqrt(np.mean(active**2))) if active.size else 1.0
    t_all = np.arange(n_total, dtype=np.float64) / fs
    noisy = (
        x
        + rng.normal(0.0, rms / (10.0 ** (snr_db / 20.0)), n_total)
        + hum_level * rms * np.sin(2.0 * np.pi * 50.0 * t_all)
    )
    return noisy, np.asarray(onsets, dtype=np.float64)

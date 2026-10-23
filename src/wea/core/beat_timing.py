"""Onset-based beat timing: P1 (unlock) threshold crossing with a phase-locked tracking gate.

Why onsets and not peaks: the envelope peak of a resonant escapement sound moves with the
random phase of the case modes (~100-150 us of jitter), and when the unlock sound is weak the
loudest sub-event changes from beat to beat (jumps of milliseconds). The rising edge of the
first sub-event is far more stable.

Pipeline (offline and streaming are the same code):
    1. Butterworth bandpass (order 4, 2-10 kHz by default) with zero-phase ``sosfiltfilt``.
    2. Hilbert envelope, smoothed by a centered ~0.2 ms box filter.
    3. For each beat: noise = median envelope over the 20 ms before the search gate,
       threshold = noise + 0.25 (peak - noise), onset = first upward crossing BEFORE the
       peak, with linear sub-sample interpolation. Optionally (default) a second pass
       re-thresholds at 25 % of that first sub-event's own peak (see ``OnsetConfig``).
    4. Tracking gate: once locked, beat ``m`` is only searched within +-3 ms of
       ``t_same + (m - n_same) T``, where ``t_same`` is the last beat of the same parity
       (tick or tock), so the beat error never pushes beats out of the gate.
    5. Streaming: audio is appended block by block; the envelope of the kept buffer is
       recomputed and only positions at least ``edge_margin_ms`` away from both buffer ends
       are used, which avoids filtfilt/Hilbert edge transients. Beats are produced strictly
       in index order, so the overlap between successive buffers can never duplicate a beat.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.fft
from numpy.typing import NDArray
from scipy.signal import butter, hilbert, oaconvolve, sosfiltfilt

from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


@dataclass(slots=True)
class OnsetConfig:
    """Tunable parameters of the onset detector (defaults follow the timegrapher spec)."""

    band_low_hz: float = 2000.0
    band_high_hz: float = 10000.0
    filter_order: int = 4
    smooth_ms: float = 0.2
    threshold_frac: float = 0.25
    noise_window_ms: float = 20.0
    gate_ms: float = 3.0
    # Gate for the first beat after acquisition (no same-parity reference yet): +-8 % of T
    acquisition_gate_frac: float = 0.08
    # Beat peak must exceed the pre-beat noise by this much to be accepted
    min_snr_db: float = 8.0
    # ...and reach this fraction of the median peak of recent beats. The maximum of plain
    # noise over a search region reaches ~2.6x its median (8 dB), so without this a missing
    # beat would be replaced by a noise "beat".
    min_peak_ratio: float = 0.35
    # Consecutive gate misses before the lock is dropped and the beat re-acquired
    max_consecutive_misses: int = 12
    # Envelope positions closer than this to a buffer end are not trusted
    edge_margin_ms: float = 30.0
    # Decay tail of the sound appended after the lift window when looking for the peak
    tail_ms: float = 4.0
    # Second pass: re-threshold at threshold_frac of the first sub-event's own peak (max
    # within subevent_peak_ms after the first crossing). The beat peak is usually the drop
    # P3, whose level wanders with the case-mode phases; normalizing to the sub-event that
    # actually starts the beat roughly halves the onset jitter on resonant signals.
    refine_subevent_peak: bool = True
    subevent_peak_ms: float = 0.6


@dataclass(slots=True)
class BeatOnset:
    """One detected beat.

    Attributes:
        index: Integer beat index ``n``; consecutive beats differ by 1, gaps by more.
        sample: Sub-sample onset position, counted from the start of the stream.
        peak: Envelope peak of the beat.
        noise: Median envelope over the 20 ms before the gate.
        segment: Continuous-audio segment id (incremented after each discontinuity).
    """

    index: int
    sample: float
    peak: float
    noise: float
    segment: int = 0


def beat_snr_db(
    peaks: NDArray[np.float64] | list[float], noises: NDArray[np.float64] | list[float]
) -> float:
    """Single SNR definition used across the app.

    ``SNR = 20 log10(median(beat envelope peaks) / median(pre-beat noise envelope))`` where the
    noise of each beat is the median envelope over the 20 ms before it. Clipped to [0, 60] dB.
    """
    p = np.asarray(peaks, dtype=np.float64)
    q = np.asarray(noises, dtype=np.float64)
    if len(p) == 0 or len(q) == 0:
        return 0.0
    noise = max(float(np.median(q)), 1e-12)
    return float(np.clip(20.0 * np.log10(max(float(np.median(p)), noise) / noise), 0.0, 60.0))


@dataclass
class _TrackState:
    locked: bool = False
    misses: int = 0
    next_index: int = 0
    cursor: float = 0.0  # acquisition scan position (absolute samples)
    history: list[BeatOnset] = field(default_factory=list)
    recent_peaks: list[float] = field(default_factory=list)


class OnsetBeatTracker:
    """Streaming P1-onset beat tracker. See the module docstring for the method."""

    def __init__(
        self,
        fs: float,
        bph: int,
        lift_angle: float = 52.0,
        config: OnsetConfig | None = None,
    ) -> None:
        """Create a tracker.

        Args:
            fs: Nominal sample rate in Hz (sample positions are reported in this clock).
            bph: Beats per hour.
            lift_angle: Lift angle in degrees (sizes the onset-before-peak search).
            config: Detector parameters.
        """
        self.fs = float(fs)
        self.bph = int(bph)
        self.lift_angle = float(lift_angle)
        self.config = config if config is not None else OnsetConfig()
        self._sos = self._design_sos()
        self.segment = 0
        self._buf = np.empty(0, dtype=np.float64)
        self._buf_start = 0  # absolute sample index of _buf[0]
        self._state = _TrackState()

    # ------------------------------------------------------------------ properties
    @property
    def period_samples(self) -> float:
        """Nominal beat period in samples."""
        return 3600.0 / float(self.bph) * self.fs

    @property
    def next_sample(self) -> int:
        """Absolute index of the next sample expected by :meth:`process`."""
        return self._buf_start + len(self._buf)

    @property
    def locked(self) -> bool:
        """True while the tracking gate is locked on the beat."""
        return self._state.locked

    def _ms(self, ms: float) -> float:
        return ms * 1e-3 * self.fs

    def _lift_max_samples(self) -> float:
        # Longest lift time expected (amplitude 140 deg) with a 10 % margin
        f_osc = self.bph / 7200.0
        ratio = float(np.clip(self.lift_angle / (2.0 * 140.0), 1e-6, 0.999))
        return 1.10 * float(np.arcsin(ratio) / (np.pi * f_osc)) * self.fs

    def _design_sos(self) -> NDArray[np.float64]:
        nyq = 0.5 * self.fs
        low = float(np.clip(self.config.band_low_hz / nyq, 1e-4, 0.90))
        high = float(np.clip(self.config.band_high_hz / nyq, low + 0.02, 0.98))
        return np.asarray(
            butter(self.config.filter_order, [low, high], btype="bandpass", output="sos")
        )

    # ------------------------------------------------------------------ public API
    def reset(self, start_sample: int = 0) -> None:
        """Forget all audio and tracking state (segment id is kept)."""
        self._buf = np.empty(0, dtype=np.float64)
        self._buf_start = int(start_sample)
        self._state = _TrackState(cursor=float(start_sample))

    def mark_discontinuity(self, start_sample: int | None = None) -> None:
        """Audio before this point is not contiguous with what follows: start a new segment."""
        self.segment += 1
        self.reset(self.next_sample if start_sample is None else start_sample)

    def envelope(self, signal: NDArray[np.float64]) -> NDArray[np.float64]:
        """Bandpass (zero phase) -> |Hilbert| -> centered box smoothing of ``smooth_ms``."""
        x = np.asarray(signal, dtype=np.float64)
        n = len(x)
        if n < 32:
            return np.zeros(n, dtype=np.float64)
        filtered = sosfiltfilt(self._sos, x)
        nfft = int(scipy.fft.next_fast_len(n))
        env = np.abs(hilbert(filtered, N=nfft)[:n])
        win = max(1, int(round(self._ms(self.config.smooth_ms))))
        if win > 1:
            full = oaconvolve(env, np.full(win, 1.0 / win), mode="full")
            env = full[(win - 1) // 2 : (win - 1) // 2 + n]
        return np.asarray(env, dtype=np.float64)

    def process(
        self, block: NDArray[np.float64], start_sample: int | None = None
    ) -> list[BeatOnset]:
        """Append ``block`` and return the beats that became final.

        Args:
            block: New mono samples.
            start_sample: Absolute index of ``block[0]``. If it does not continue the stream
                (samples were lost or the source restarted), a discontinuity is marked.
        """
        samples = np.asarray(block, dtype=np.float64).ravel()
        if start_sample is not None and int(start_sample) != self.next_sample:
            self.mark_discontinuity(int(start_sample))
        if len(samples):
            self._buf = np.concatenate([self._buf, samples]) if len(self._buf) else samples.copy()
        return self._scan(final=False)

    def flush(self) -> list[BeatOnset]:
        """Process what is left at the end of a finite recording."""
        return self._scan(final=True)

    def detect(self, signal: NDArray[np.float64]) -> list[BeatOnset]:
        """Offline convenience: track a whole recording from scratch."""
        self.segment = 0
        self.reset(0)
        beats = self.process(signal, start_sample=0)
        beats.extend(self.flush())
        return beats

    # ------------------------------------------------------------------ internals
    def _predict(self, m: int) -> tuple[float, float]:
        """Return ``(predicted onset sample, gate half-width)`` for beat index ``m``."""
        st = self._state
        period = self.period_samples
        same = next((b for b in reversed(st.history) if (b.index - m) % 2 == 0), None)
        if same is not None:
            return same.sample + (m - same.index) * period, self._ms(self.config.gate_ms)
        last = st.history[-1]
        return last.sample + (m - last.index) * period, self.config.acquisition_gate_frac * period

    def _measure(
        self,
        env: NDArray[np.float64],
        gate_lo: float,
        gate_hi: float,
        peak_hi: float,
    ) -> tuple[float, float, float] | None:
        """Find the onset in ``[gate_lo, gate_hi]`` (absolute samples) for the beat whose peak
        lies in ``[gate_lo, peak_hi]``; returns ``(onset, peak, noise)`` or ``None``."""
        b0 = self._buf_start
        lo = int(np.floor(gate_lo)) - b0
        hi = int(np.ceil(gate_hi)) - b0
        ph = int(np.ceil(peak_hi)) - b0
        nw = int(round(self._ms(self.config.noise_window_ms)))
        if lo - nw < 0 or ph > len(env) or ph <= lo:
            return None
        pk = lo + int(np.argmax(env[lo:ph]))
        peak = float(env[pk])
        noise = float(np.median(env[lo - nw : lo]))
        if peak <= noise * 10.0 ** (self.config.min_snr_db / 20.0):
            return None
        recent = self._state.recent_peaks
        if len(recent) >= 3 and peak < self.config.min_peak_ratio * float(np.median(recent)):
            return None
        thr = noise + self.config.threshold_frac * (peak - noise)
        end = min(hi, pk)
        seg = env[max(lo, 1) - 1 : end + 1]
        up = np.flatnonzero((seg[:-1] < thr) & (seg[1:] >= thr))
        if len(up) == 0:
            return None
        i = max(lo, 1) + int(up[0])  # env[i - 1] < thr <= env[i]
        if self.config.refine_subevent_peak:
            j = min(len(env), i + max(2, int(round(self._ms(self.config.subevent_peak_ms)))))
            k = i + int(np.argmax(env[i:j]))
            thr2 = noise + self.config.threshold_frac * (float(env[k]) - noise)
            seg2 = env[i - 1 : k + 1]
            up2 = np.flatnonzero((seg2[:-1] < thr2) & (seg2[1:] >= thr2))
            if len(up2):
                # Last upward crossing before the sub-event peak (ripple-safe)
                i = i + int(up2[-1])
                thr = thr2
            elif thr2 < thr:
                # Sub-event threshold lies below env[i - 1]: walk back along the rising edge
                w = i - 1
                while w > lo and env[w - 1] >= thr2:
                    w -= 1
                if w > lo:
                    i, thr = w, thr2
        y0, y1 = float(env[i - 1]), float(env[i])
        onset_rel = (i - 1) + ((thr - y0) / (y1 - y0) if y1 != y0 else 1.0)
        return b0 + onset_rel, peak, noise

    def _scan(self, final: bool) -> list[BeatOnset]:
        cfg = self.config
        st = self._state
        out: list[BeatOnset] = []
        if len(self._buf) == 0:
            return out
        env = self.envelope(self._buf)
        edge = self._ms(cfg.edge_margin_ms)
        valid_end = self.next_sample - (self._ms(1.0) if final else edge)
        period = self.period_samples
        lift = self._lift_max_samples()
        tail = self._ms(cfg.tail_ms)
        nw = self._ms(cfg.noise_window_ms)
        valid_start = self._buf_start + (0.0 if self._buf_start == 0 else edge)

        while True:
            if st.locked:
                m = st.next_index
                pred, half = self._predict(m)
                gate_lo, gate_hi = pred - half, pred + half
                peak_hi = gate_hi + lift + tail
                if peak_hi > valid_end:
                    break
                if gate_lo - nw < valid_start:
                    # Not enough history before this beat (stream start): skip it
                    res = None
                else:
                    res = self._measure(env, gate_lo, gate_hi, peak_hi)
                st.next_index = m + 1
                if res is not None:
                    beat = BeatOnset(m, res[0], res[1], res[2], self.segment)
                    self._accept(beat)
                    out.append(beat)
                else:
                    st.misses += 1
                    if st.misses > cfg.max_consecutive_misses:
                        st.locked = False
                        st.cursor = gate_hi
                        logger.debug("Onset tracker lost lock at beat %d; re-acquiring", m)
            else:
                start = max(st.cursor, valid_start + nw + lift + tail)
                stop = start + period
                if stop + 2.0 * (lift + tail) > valid_end:
                    break
                i0 = int(start) - self._buf_start
                i1 = int(stop) - self._buf_start
                pk_abs = self._buf_start + i0 + int(np.argmax(env[i0:i1]))
                res = self._measure(env, pk_abs - lift - tail, pk_abs, pk_abs + lift + tail)
                if res is None:
                    st.cursor = start + 0.5 * period
                    continue
                onset = res[0]
                if st.history:
                    last = st.history[-1]
                    idx = last.index + int(np.rint((onset - last.sample) / period))
                    if idx <= last.index:
                        st.cursor = onset + 0.5 * period
                        continue
                else:
                    idx = 0
                beat = BeatOnset(idx, onset, res[1], res[2], self.segment)
                self._accept(beat)
                st.locked = True
                st.next_index = idx + 1
                out.append(beat)

        self._trim(nw + lift + tail + edge + self._ms(cfg.gate_ms) + 0.1 * period)
        return out

    def _accept(self, beat: BeatOnset) -> None:
        st = self._state
        st.history.append(beat)
        del st.history[:-4]
        st.recent_peaks.append(beat.peak)
        del st.recent_peaks[:-9]
        st.misses = 0

    def _trim(self, context: float) -> None:
        """Drop audio that no future beat can need (keeps ``context`` samples of history)."""
        st = self._state
        if st.locked:
            pred, half = self._predict(st.next_index)
            need_from = pred - half - context
        else:
            need_from = st.cursor - context
        cut = int(need_from) - self._buf_start
        if cut > 0:
            cut = min(cut, len(self._buf))
            self._buf = self._buf[cut:]
            self._buf_start += cut


def detect_onsets(
    signal: NDArray[np.float64],
    fs: float,
    bph: int,
    lift_angle: float = 52.0,
    config: OnsetConfig | None = None,
) -> list[BeatOnset]:
    """Offline helper: P1 onsets of a whole recording."""
    return OnsetBeatTracker(fs, bph, lift_angle, config).detect(signal)

"""Non-blocking real-time Qt DSP worker bridging AudioRingBuffer, SampleRateDriftEstimator, and DSPEngine."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from wea.core.clock_drift import AtomicNTPDiscipline, SampleRateDriftEstimator
from wea.core.dsp_engine import DSPEngine
from wea.core.models import BeatTelemetry, WatchConfig
from wea.core.ring_buffer import AudioRingBuffer
from wea.core.synthetic import SyntheticEscapeGenerator
from wea.utils.audio_io import HardwareAudioCapture
from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class DSPStreamWorker(QObject):
    """Real-time DSP stream worker designed to run inside a dedicated QThread via moveToThread.

    Coordinates:
        - ``AudioRingBuffer`` for pre-allocated circular sample storage.
        - ``SampleRateDriftEstimator`` for dynamic OLS and static quartz clock drift compensation.
        - ``DSPEngine`` for multi-threaded Hilbert envelope and escapement telemetry extraction.
        - ``SyntheticEscapeGenerator`` for phase-continuous real-time synthetic simulation when
          hardware capture is not active.
    """

    telemetry_ready = pyqtSignal(object)
    waveform_ready = pyqtSignal(object, object)
    drift_updated = pyqtSignal(float, float, float)
    error_occurred = pyqtSignal(str)
    session_finished = pyqtSignal()

    _sig_start_timer = pyqtSignal(int)
    _sig_stop_timer = pyqtSignal()
    _sig_start_telemetry_timer = pyqtSignal(int)
    _sig_stop_telemetry_timer = pyqtSignal()
    _sig_start_ntp_timer = pyqtSignal(int)
    _sig_stop_ntp_timer = pyqtSignal()
    _sig_schedule_eval = pyqtSignal()

    DEFAULT_BUFFER_DURATION_S: float = 8.0
    DEFAULT_WINDOW_DURATION_S: float = 3.0
    DEFAULT_CHUNK_SIZE: int = 4096
    VISUAL_TIMER_INTERVAL_MS: int = 33
    TELEMETRY_TIMER_INTERVAL_MS: int = 250
    WAVEFORM_WINDOW_MS: float = 600.0
    NTP_DISCIPLINE_INTERVAL_MS: int = 30000

    def __init__(
        self,
        config: WatchConfig | None = None,
        *,
        device_id: int | None = None,
        buffer_duration_s: float = DEFAULT_BUFFER_DURATION_S,
        window_duration_s: float = DEFAULT_WINDOW_DURATION_S,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        calibration_ppm: float = 0.0,
        synthetic_mode: bool = True,
        synthetic_rate_spd: float = 4.5,
        synthetic_beat_error_ms: float = 0.35,
        synthetic_amplitude_deg: float = 280.0,
        synthetic_snr_db: float = 35.0,
        workers: int | None = None,
        adc_dynamic_drift: bool = False,
        analysis_in_background: bool = False,
        parent: QObject | None = None,
    ) -> None:
        """Initialize the real-time DSP worker and its underlying mathematical components.

        Args:
            config: Watch movement configuration (BPH, lift angle, sample rate).
            device_id: Hardware input device index, or ``None`` for the synthetic generator.
            buffer_duration_s: Ring buffer capacity in seconds (default 8.0s).
            window_duration_s: Sliding DSP analysis window in seconds (default 4.0s).
            chunk_size: Number of audio samples per streaming chunk (default 4096).
            calibration_ppm: Initial static clock drift calibration offset in ppm.
            synthetic_mode: Whether ``start_stream()`` feeds synthetic chunks on a timer.
            synthetic_rate_spd: Ground-truth rate (s/d) used in synthetic simulation mode.
            synthetic_beat_error_ms: Ground-truth beat error (ms) used in synthetic mode.
            synthetic_amplitude_deg: Ground-truth amplitude (deg) used in synthetic mode.
            synthetic_snr_db: SNR (dB) used in synthetic simulation mode.
            workers: Number of worker threads inside ``DSPEngine``.
            adc_dynamic_drift: Track the ADC rate against the host clock (OLS over callback
                timestamps, plus NTP disciplining). Off by default: the host clock is not a
                better reference than the ADC crystal and callback jitter adds rate noise.
                Use ``calibration_ppm`` for a measured ADC correction instead.
            analysis_in_background: Run ``DSPEngine.process_window`` on a dedicated analysis
                thread so this worker's event loop keeps emitting 30 FPS waveforms while a
                window is being analyzed. Windows arriving while an analysis is in flight are
                skipped (the next tick analyzes the newest audio).
            parent: Optional Qt parent object.
        """
        super().__init__(parent)

        self.config: WatchConfig = config if config is not None else WatchConfig()
        self.device_id: int | None = self._normalize_device_id(device_id)
        self.buffer_duration_s: float = max(2.0, float(buffer_duration_s))
        self.window_duration_s: float = max(
            1.0, min(float(window_duration_s), self.buffer_duration_s)
        )
        self.chunk_size: int = max(32, int(chunk_size))
        self.synthetic_mode: bool = bool(synthetic_mode) if self.device_id is None else False

        self.synthetic_rate_spd: float = float(synthetic_rate_spd)
        self.synthetic_beat_error_ms: float = float(synthetic_beat_error_ms)
        self.synthetic_amplitude_deg: float = float(synthetic_amplitude_deg)
        self.synthetic_snr_db: float = float(synthetic_snr_db)

        capacity_samples = int(round(self.buffer_duration_s * self.config.sample_rate))
        self.ring_buffer = AudioRingBuffer(capacity_samples=capacity_samples)
        self.adc_dynamic_drift: bool = bool(adc_dynamic_drift)
        self.drift_estimator = SampleRateDriftEstimator(
            nominal_fs=float(self.config.sample_rate),
            calibration_ppm=float(calibration_ppm),
            dynamic_tracking=self.adc_dynamic_drift,
        )
        self.ntp_discipline = AtomicNTPDiscipline()
        self.dsp_engine = DSPEngine(
            config=self.config,
            workers=workers,
            drift_estimator=self.drift_estimator,
        )
        self._hardware_capture: HardwareAudioCapture | None = None
        self._ntp_busy: bool = False
        self._overflows_seen: int = 0
        self.discontinuity_count: int = 0

        self.analysis_in_background: bool = bool(analysis_in_background)
        self._analysis_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="WEA-Analysis")
            if self.analysis_in_background
            else None
        )
        self._analysis_future: Future | None = None
        # Bumped on start/stop/config changes so stale background results are dropped.
        self._analysis_generation: int = 0

        self._window_samples: int = 0
        self._min_analysis_samples: int = 0
        self._analysis_hop_samples: int = 0
        self._samples_since_last_analysis: int = 0
        self._has_emitted_telemetry: bool = False
        self._eval_queued: bool = False
        self._recompute_window_thresholds()

        self._running: bool = False
        self._receiving_external_chunks: bool = False
        self._synth_seed: int = 1000
        self._synth_buffer: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self._synth_cursor: int = 0
        self._synth_next_p1_s: float = 0.050
        self._synth_beat_idx: int = 0
        self._waveform_step_idx: int = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_visual_timer_tick)
        self._sig_start_timer.connect(self._timer.start)
        self._sig_stop_timer.connect(self._timer.stop)

        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._on_telemetry_timer_tick)
        self._sig_start_telemetry_timer.connect(self._telemetry_timer.start)
        self._sig_stop_telemetry_timer.connect(self._telemetry_timer.stop)

        self._ntp_timer = QTimer(self)
        self._ntp_timer.timeout.connect(self.trigger_ntp_discipline_update)
        self._sig_start_ntp_timer.connect(self._ntp_timer.start)
        self._sig_stop_ntp_timer.connect(self._ntp_timer.stop)

        self._sig_schedule_eval.connect(self._on_scheduled_eval)

        logger.info(
            "DSPStreamWorker initialized (bph=%d, lift=%.1f deg, fs=%d Hz, "
            "capacity=%d samples, chunk_size=%d, device_id=%s, synthetic_mode=%s)",
            self.config.bph,
            self.config.lift_angle,
            self.config.sample_rate,
            self.ring_buffer.capacity_samples,
            self.chunk_size,
            self.device_id,
            self.synthetic_mode,
        )

    @property
    def engine(self) -> DSPEngine:
        """Return the underlying DSPEngine instance."""
        return self.dsp_engine

    def isRunning(self) -> bool:  # noqa: N802
        """Return True if the stream worker is currently active."""
        return self._running

    def _recompute_window_thresholds(self) -> None:
        """Compute sample thresholds for sliding-window DSP execution."""
        fs = self.config.sample_rate
        self._window_samples = min(
            self.ring_buffer.capacity_samples,
            int(round(self.window_duration_s * fs)),
        )
        min_beats_sec = max(1.0, 5.0 * self.config.nominal_beat_period)
        self._min_analysis_samples = min(
            self._window_samples,
            int(round(min_beats_sec * fs)),
        )
        self._analysis_hop_samples = max(1, int(round(0.25 * fs)))

    def _timer_interval_ms(self) -> int:
        """Return visual timer interval in milliseconds (~30 FPS)."""
        return self.VISUAL_TIMER_INTERVAL_MS

    def _reset_synthetic_state(self) -> None:
        """Reset phase-continuous synthetic generator state."""
        self._synth_buffer = np.empty(0, dtype=np.float64)
        self._synth_cursor = 0
        self._synth_next_p1_s = 0.050
        self._synth_beat_idx = 0
        self._waveform_step_idx = 0

    def _generate_phase_continuous_block(self, min_duration_s: float = 6.0) -> NDArray[np.float64]:
        """Synthesize a phase-continuous escapement audio block preserving exact beat timing."""
        fs = self.config.sample_rate
        gen = SyntheticEscapeGenerator(config=self.config, seed=self._synth_seed)
        self._synth_seed += 1

        t0 = self.config.nominal_beat_period
        rate_factor = 1.0 + (self.synthetic_rate_spd / SyntheticEscapeGenerator.SECONDS_PER_DAY)
        mean_beat_period = t0 / rate_factor
        half_be_s = SyntheticEscapeGenerator.beat_error_half_offset(self.synthetic_beat_error_ms)

        dt_p1_p3 = gen.compute_p1_p3_delay(
            self.synthetic_amplitude_deg,
            actual_beat_period=mean_beat_period,
        )
        dt_p1_p2 = 0.28 * dt_p1_p3

        n_samples = int(round(max(min_duration_s, 2.0) * fs))
        signal = np.zeros(n_samples, dtype=np.float64)
        block_duration_s = n_samples / float(fs)

        kernel_len = int(0.0025 * fs)
        t_k = np.arange(kernel_len, dtype=np.float64) / fs

        tau_p1, tau_p2, tau_p3 = 0.00015, 0.00018, 0.00020
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

        wave_p1 = 0.55 * env_p1 * carrier_p1
        wave_p2 = 0.45 * env_p2 * carrier_p2
        wave_p3 = 1.00 * env_p3 * carrier_p3

        current_p1 = self._synth_next_p1_s
        beat_idx = self._synth_beat_idx

        while current_p1 + dt_p1_p3 + 0.005 < block_duration_s:
            gen._add_sub_sample_impulse(signal, current_p1, wave_p1, fs)
            gen._add_sub_sample_impulse(signal, current_p1 + dt_p1_p2, wave_p2, fs)
            gen._add_sub_sample_impulse(signal, current_p1 + dt_p1_p3, wave_p3, fs)

            interval = (
                mean_beat_period + half_be_s if beat_idx % 2 == 0 else mean_beat_period - half_be_s
            )
            current_p1 += interval
            beat_idx += 1

        # Cut the block in the quiet inter-beat region right before the next unplaced beat
        cut_s = max(0.5, min(block_duration_s, current_p1 - 0.010))
        cut_samples = max(1, min(n_samples, int(np.floor(cut_s * fs))))
        actual_cut_s = cut_samples / float(fs)
        signal = signal[:cut_samples]

        self._synth_next_p1_s = max(0.005, current_p1 - actual_cut_s)
        self._synth_beat_idx = beat_idx

        if self.synthetic_snr_db is not None:
            sig_power = float(np.mean(signal**2))
            if sig_power > 0.0:
                noise_power = sig_power / (10.0 ** (self.synthetic_snr_db / 10.0))
                signal = signal + gen.rng.normal(0.0, np.sqrt(noise_power), size=len(signal))

        return signal

    def _next_synthetic_chunk(self, num_samples: int) -> NDArray[np.float64]:
        """Extract the next ``num_samples`` from the phase-continuous synthetic stream."""
        remaining = len(self._synth_buffer) - self._synth_cursor
        if remaining < num_samples:
            leftover = (
                self._synth_buffer[self._synth_cursor :]
                if remaining > 0
                else np.empty(0, dtype=np.float64)
            )
            need_s = max(6.0, (num_samples / float(self.config.sample_rate)) + 2.0)
            fresh = self._generate_phase_continuous_block(min_duration_s=need_s)
            if len(leftover) > 0:
                combined = np.empty(len(leftover) + len(fresh), dtype=np.float64)
                combined[: len(leftover)] = leftover
                combined[len(leftover) :] = fresh
                self._synth_buffer = combined
            else:
                self._synth_buffer = fresh
            self._synth_cursor = 0

        chunk = self._synth_buffer[self._synth_cursor : self._synth_cursor + num_samples].copy()
        self._synth_cursor += num_samples
        return chunk

    @staticmethod
    def _normalize_device_id(device_id: object) -> int | None:
        """Normalize UI or caller device_id into ``int | None`` (``None`` for synthetic)."""
        if device_id is None or device_id == "synthetic" or device_id == "":
            return None
        if isinstance(device_id, bool):
            return None
        try:
            parsed = int(device_id)  # type: ignore[arg-type]
            return parsed if parsed >= 0 else None
        except (TypeError, ValueError):
            return None

    def _stop_hardware_capture(self) -> None:
        """Stop and release any active HardwareAudioCapture stream."""
        cap = self._hardware_capture
        self._hardware_capture = None
        if cap is not None:
            cap.stop()

    @pyqtSlot(object)
    def set_input_device(self, device_id: int | None) -> None:
        """Switch between SyntheticEscapeGenerator (``None``) and HardwareAudioCapture (``int``)."""
        norm_id = self._normalize_device_id(device_id)
        changed = norm_id != self.device_id
        self.device_id = norm_id
        self.synthetic_mode = norm_id is None
        logger.info(
            "DSPStreamWorker input device set to device_id=%s (synthetic_mode=%s)",
            self.device_id,
            self.synthetic_mode,
        )
        if self._running and changed:
            self.stop_stream()
            self.start_stream()

    @pyqtSlot()
    def trigger_ntp_discipline_update(self) -> None:
        """Trigger a non-blocking background AtomicNTPDiscipline skew update."""
        if self._ntp_busy:
            return
        self._ntp_busy = True

        def _worker_fn() -> None:
            try:
                skew_ppm = self.ntp_discipline.update_cpu_skew()
                self.set_cpu_skew_ppm(skew_ppm)
            except Exception as exc:
                logger.debug("Background AtomicNTPDiscipline check skipped: %s", exc)
            finally:
                self._ntp_busy = False

        threading.Thread(
            target=_worker_fn,
            name="WEA-AtomicNTPDiscipline",
            daemon=True,
        ).start()

    @pyqtSlot()
    def start_stream(self) -> None:
        """Start streaming analysis (synthetic timer or hardware PortAudio capture)."""
        logger.info(
            "DSPStreamWorker starting stream (device_id=%s, synthetic_mode=%s, bph=%d, lift=%.1f, fs=%d)",
            self.device_id,
            self.synthetic_mode,
            self.config.bph,
            self.config.lift_angle,
            self.config.sample_rate,
        )
        self._stop_hardware_capture()
        self._sig_stop_timer.emit()
        self._sig_stop_telemetry_timer.emit()
        self._sig_stop_ntp_timer.emit()

        self._running = True
        self._analysis_generation += 1
        self._receiving_external_chunks = False
        self._overflows_seen = 0
        self._samples_since_last_analysis = 0
        self._has_emitted_telemetry = False
        self._eval_queued = False
        self.ring_buffer.clear()
        self.drift_estimator.reset()
        self.dsp_engine.reset_filter_state()
        self._reset_synthetic_state()

        if self.device_id is not None:
            try:
                self._hardware_capture = HardwareAudioCapture(
                    device_id=self.device_id,
                    sample_rate=float(self.config.sample_rate),
                    blocksize=self.chunk_size,
                    callback=self._on_hardware_audio_chunk,
                )
                self._hardware_capture.start()
                self._sig_start_timer.emit(self.VISUAL_TIMER_INTERVAL_MS)
                self._sig_start_telemetry_timer.emit(self.TELEMETRY_TIMER_INTERVAL_MS)
                # The NTP CPU skew only matters when the ADC is tracked against the host clock
                if self.adc_dynamic_drift:
                    self._sig_start_ntp_timer.emit(self.NTP_DISCIPLINE_INTERVAL_MS)
            except Exception as exc:
                msg = (
                    f"Failed to open hardware audio device {self.device_id} "
                    f"at {self.config.sample_rate} Hz: {exc}"
                )
                logger.error(msg)
                self._stop_hardware_capture()
                self._running = False
                self.error_occurred.emit(msg)
                self.session_finished.emit()
                return
        elif self.synthetic_mode:
            self._sig_start_timer.emit(self.VISUAL_TIMER_INTERVAL_MS)
            self._sig_start_telemetry_timer.emit(self.TELEMETRY_TIMER_INTERVAL_MS)

    @pyqtSlot()
    def stop_stream(self) -> None:
        """Stop the active stream and halt synthetic timer or hardware capture."""
        was_running = self._running
        self._running = False
        self._analysis_generation += 1
        self._eval_queued = False
        self._sig_stop_timer.emit()
        self._sig_stop_telemetry_timer.emit()
        self._sig_stop_ntp_timer.emit()
        self._stop_hardware_capture()
        logger.info("DSPStreamWorker stream stopped")
        if was_running:
            self.session_finished.emit()

    stop = stop_stream

    @pyqtSlot(object)
    def update_watch_config(self, config: WatchConfig) -> None:
        """Update WatchConfig (BPH, lift angle, sample rate) in real time."""
        try:
            if not isinstance(config, WatchConfig):
                raise TypeError(f"Expected WatchConfig instance, got {type(config).__name__}")

            fs_changed = config.sample_rate != self.config.sample_rate
            self.config = config

            if fs_changed:
                new_capacity = int(round(self.buffer_duration_s * config.sample_rate))
                self.ring_buffer = AudioRingBuffer(capacity_samples=new_capacity)
                self.drift_estimator.nominal_fs = float(config.sample_rate)
                self.drift_estimator.reset()
                self.dsp_engine._sos = self.dsp_engine._design_bandpass_sos(
                    float(config.sample_rate),
                    self.dsp_engine.highpass_cutoff_hz,
                    self.dsp_engine.highcut_hz,
                    self.dsp_engine.filter_order,
                )

            self.dsp_engine.config = config
            self.dsp_engine.reset_filter_state()
            self._analysis_generation += 1
            self._recompute_window_thresholds()
            self._samples_since_last_analysis = 0
            self._eval_queued = False
            self._reset_synthetic_state()

            if self._running:
                if self.device_id is not None and fs_changed:
                    self._stop_hardware_capture()
                    self.ring_buffer.clear()
                    try:
                        self._hardware_capture = HardwareAudioCapture(
                            device_id=self.device_id,
                            sample_rate=float(config.sample_rate),
                            blocksize=self.chunk_size,
                            callback=self._on_hardware_audio_chunk,
                        )
                        self._hardware_capture.start()
                    except Exception as exc:
                        msg = (
                            f"Failed to reopen hardware audio device {self.device_id} "
                            f"at {config.sample_rate} Hz: {exc}"
                        )
                        logger.error(msg)
                        self._stop_hardware_capture()
                        self._running = False
                        self.error_occurred.emit(msg)
                        self.session_finished.emit()
                        return
                elif self.synthetic_mode and not self._receiving_external_chunks:
                    self.ring_buffer.clear()
                    self._sig_start_timer.emit(self.VISUAL_TIMER_INTERVAL_MS)
                    self._sig_start_telemetry_timer.emit(self.TELEMETRY_TIMER_INTERVAL_MS)

            logger.info(
                "DSPStreamWorker updated WatchConfig (bph=%d, lift=%.1f deg, fs=%d Hz)",
                config.bph,
                config.lift_angle,
                config.sample_rate,
            )
        except Exception as exc:
            logger.error("Failed to update WatchConfig in DSPStreamWorker: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    @pyqtSlot(float)
    def set_calibration_ppm(self, ppm: float) -> None:
        """Update static clock drift calibration offset in parts per million."""
        try:
            self.drift_estimator.calibration_ppm = float(ppm)
            eff_fs = self.drift_estimator.effective_fs
            total_ppm = self.drift_estimator.drift_ppm
            bias_spd = self.drift_estimator.rate_bias_spd
            logger.info(
                "DSPStreamWorker calibration_ppm updated to %+.3f ppm (effective_fs=%.4f Hz)",
                self.drift_estimator.calibration_ppm,
                eff_fs,
            )
            self.drift_updated.emit(eff_fs, total_ppm, bias_spd)
        except Exception as exc:
            logger.error("Failed to set calibration_ppm in DSPStreamWorker: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    @pyqtSlot(float)
    def set_cpu_skew_ppm(self, skew_ppm: float) -> None:
        """Update host CPU monotonic clock skew offset (from AtomicNTPDiscipline) in ppm."""
        try:
            self.drift_estimator.cpu_skew_ppm = float(skew_ppm)
            eff_fs = self.drift_estimator.effective_fs
            total_ppm = self.drift_estimator.drift_ppm
            bias_spd = self.drift_estimator.rate_bias_spd
            logger.info(
                "DSPStreamWorker cpu_skew_ppm updated to %+.3f ppm (effective_fs=%.4f Hz)",
                self.drift_estimator.cpu_skew_ppm,
                eff_fs,
            )
            self.drift_updated.emit(eff_fs, total_ppm, bias_spd)
        except Exception as exc:
            logger.error("Failed to set cpu_skew_ppm in DSPStreamWorker: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    def _on_hardware_audio_chunk(
        self,
        chunk: NDArray[np.float64],
        timestamp_ns: int | None = None,
    ) -> None:
        """Non-blocking PortAudio ingestion callback: writes to ring buffer with zero DSP on audio thread."""
        if chunk is None:
            return
        samples = np.asarray(chunk, dtype=np.float64).ravel()
        n_samples = int(samples.size)
        if n_samples == 0:
            return

        self._receiving_external_chunks = True

        # An input overflow means PortAudio dropped samples: the buffer now holds two
        # stretches of audio glued together with an unknown time gap, which turns into
        # absurd rate/beat-error values. Discard pre-gap audio and warm up again.
        cap = self._hardware_capture
        overflows = int(cap.overflow_count) if cap is not None else 0
        if overflows > self._overflows_seen:
            self._overflows_seen = overflows
            self._handle_stream_discontinuity()

        self.ring_buffer.write(samples)
        self.drift_estimator.update(n_samples, timestamp_ns=timestamp_ns)
        self._samples_since_last_analysis += n_samples

        if (
            self.ring_buffer.available_samples >= self._min_analysis_samples
            and self._samples_since_last_analysis >= self._analysis_hop_samples
            and not self._eval_queued
        ):
            self._eval_queued = True
            self._sig_schedule_eval.emit()

    def _handle_stream_discontinuity(self) -> None:
        """Drop buffered audio after a gap in the sample stream (e.g. input overflow)."""
        self.discontinuity_count += 1
        self.ring_buffer.clear()
        self._samples_since_last_analysis = 0
        self.dsp_engine.reset_filter_state()
        self._analysis_generation += 1
        if self.discontinuity_count == 1 or self.discontinuity_count % 20 == 0:
            logger.warning(
                "Audio stream discontinuity #%d (input overflow): analysis buffer reset",
                self.discontinuity_count,
            )

    @pyqtSlot()
    def _on_scheduled_eval(self) -> None:
        """Run a telemetry evaluation requested by the audio callback on the worker QThread.

        Waveforms are produced by the 30 FPS visual timer; running them here as well (and
        letting both the timer and this slot analyze the same audio) doubled the DSP load.
        """
        self._eval_queued = False
        self._on_telemetry_timer_tick()

    @pyqtSlot()
    def _on_visual_timer_tick(self) -> None:
        """Fast 30 FPS (33 ms) visual timer: feeds synthetic audio (if active) and emits waveform_ready."""
        if not self._running:
            return
        try:
            if self.synthetic_mode and not self._receiving_external_chunks:
                if self.ring_buffer.available_samples < self._min_analysis_samples:
                    warmup_needed = max(
                        0,
                        self._min_analysis_samples - self.ring_buffer.available_samples,
                    )
                    if warmup_needed > 0:
                        warmup_chunk = self._next_synthetic_chunk(warmup_needed)
                        self.ring_buffer.write(warmup_chunk)
                        self.drift_estimator.update(len(warmup_chunk))
                        self._samples_since_last_analysis += len(warmup_chunk)

                visual_samples = max(
                    256,
                    int(round((self.VISUAL_TIMER_INTERVAL_MS / 1000.0) * self.config.sample_rate)),
                )
                synth_chunk = self._next_synthetic_chunk(visual_samples)
                self.ring_buffer.write(synth_chunk)
                self.drift_estimator.update(len(synth_chunk))
                self._samples_since_last_analysis += len(synth_chunk)

                if not self._has_emitted_telemetry:
                    self._on_telemetry_timer_tick()

            avail = self.ring_buffer.available_samples
            if avail <= 0:
                return

            effective_fs = float(self.drift_estimator.effective_fs)
            wave_samples = min(
                avail,
                max(64, int(round((self.WAVEFORM_WINDOW_MS / 1000.0) * effective_fs))),
            )
            latest_wave = self.ring_buffer.read_latest(wave_samples)
            time_axis, wave_win = self._extract_waveform_window(
                latest_wave,
                effective_fs=effective_fs,
                step_idx=self._waveform_step_idx,
                window_ms=self.WAVEFORM_WINDOW_MS,
            )
            self._waveform_step_idx += 1
            if len(time_axis) > 0:
                self.waveform_ready.emit(time_axis, wave_win)
        except Exception as exc:
            logger.error("Error in visual timer tick: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    @pyqtSlot()
    def _on_telemetry_timer_tick(self) -> None:
        """Decoupled 4 Hz (250 ms) DSP timer: runs DSPEngine.process_window on the worker QThread."""
        if not self._running:
            return
        try:
            effective_fs = float(self.drift_estimator.effective_fs)
            self.drift_updated.emit(
                effective_fs,
                float(self.drift_estimator.drift_ppm),
                float(self.drift_estimator.rate_bias_spd),
            )

            hop_ready = self._samples_since_last_analysis >= self._analysis_hop_samples or (
                not self._has_emitted_telemetry and self._samples_since_last_analysis > 0
            )
            if self.ring_buffer.available_samples >= self._min_analysis_samples and hop_ready:
                if self._analysis_busy():
                    return
                self._samples_since_last_analysis = 0
                window = self.ring_buffer.read_latest(self._window_samples)
                self._dispatch_analysis(window, effective_fs)
        except Exception as exc:
            logger.error("Error in telemetry timer tick: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    def _analysis_busy(self) -> bool:
        """Return True while a background analysis is still running."""
        fut = self._analysis_future
        return fut is not None and not fut.done()

    def _dispatch_analysis(self, window: NDArray[np.float64], effective_fs: float) -> None:
        """Analyze ``window`` inline or on the background analysis thread."""
        bph = int(self.config.bph)
        lift = float(self.config.lift_angle)
        generation = self._analysis_generation
        if self._analysis_executor is None:
            self._run_analysis(window, bph, lift, effective_fs, generation)
            return
        self._analysis_future = self._analysis_executor.submit(
            self._run_analysis, window, bph, lift, effective_fs, generation
        )

    def _run_analysis(
        self,
        window: NDArray[np.float64],
        bph: int,
        lift_angle: float,
        effective_fs: float,
        generation: int,
    ) -> None:
        """Run ``DSPEngine.process_window`` and emit the telemetry if still current."""
        try:
            telemetry: BeatTelemetry = self.dsp_engine.process_window(
                window,
                bph=bph,
                lift_angle=lift_angle,
                effective_fs=effective_fs,
                streaming=False,
            )
        except Exception as exc:
            logger.error("Error in DSP analysis: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))
            return
        if generation != self._analysis_generation or not self._running:
            return
        self._has_emitted_telemetry = True
        self.telemetry_ready.emit(telemetry)

    _on_synthetic_timer_tick = _on_visual_timer_tick

    @pyqtSlot(object)
    def push_audio_chunk(
        self,
        chunk: NDArray[np.float64],
        timestamp_ns: int | None = None,
    ) -> None:
        """Ingest an external audio chunk into the ring buffer and trigger DSP analysis when hop is met.

        Args:
            chunk: 1D float64 NumPy array of audio samples.
            timestamp_ns: Optional monotonic timestamp in nanoseconds for drift estimation.
        """
        self._receiving_external_chunks = True
        self._ingest_chunk(chunk, timestamp_ns=timestamp_ns)

    def _ingest_chunk(
        self,
        chunk: NDArray[np.float64],
        timestamp_ns: int | None = None,
    ) -> None:
        """Write chunk to ring buffer, update clock drift estimator, and run DSP at hop interval."""
        try:
            if chunk is None:
                raise ValueError("Audio chunk must not be None")

            samples = np.asarray(chunk, dtype=np.float64).ravel()
            n_samples = int(samples.size)
            if n_samples == 0:
                return

            self.ring_buffer.write(samples)
            effective_fs = self.drift_estimator.update(n_samples, timestamp_ns=timestamp_ns)
            self._samples_since_last_analysis += n_samples

            if (
                self.ring_buffer.available_samples >= self._min_analysis_samples
                and self._samples_since_last_analysis >= self._analysis_hop_samples
            ):
                self._samples_since_last_analysis = 0
                self.drift_updated.emit(
                    float(effective_fs),
                    float(self.drift_estimator.drift_ppm),
                    float(self.drift_estimator.rate_bias_spd),
                )
                window = self.ring_buffer.read_latest(self._window_samples)
                telemetry: BeatTelemetry = self.dsp_engine.process_window(
                    window,
                    bph=self.config.bph,
                    lift_angle=self.config.lift_angle,
                    effective_fs=effective_fs,
                    streaming=False,
                )
                self._has_emitted_telemetry = True
                self.telemetry_ready.emit(telemetry)

                time_axis, wave_win = self._extract_waveform_window(
                    window,
                    effective_fs=effective_fs,
                    step_idx=self._waveform_step_idx,
                    window_ms=self.WAVEFORM_WINDOW_MS,
                )
                self._waveform_step_idx += 1
                self.waveform_ready.emit(time_axis, wave_win)
            elif not self._has_emitted_telemetry and n_samples >= 1024:
                self.drift_updated.emit(
                    float(effective_fs),
                    float(self.drift_estimator.drift_ppm),
                    float(self.drift_estimator.rate_bias_spd),
                )
                time_axis, wave_win = self._extract_waveform_window(
                    samples,
                    effective_fs=effective_fs,
                    step_idx=self._waveform_step_idx,
                    window_ms=self.WAVEFORM_WINDOW_MS,
                )
                self._waveform_step_idx += 1
                self.waveform_ready.emit(time_axis, wave_win)

        except Exception as exc:
            logger.error("DSPStreamWorker chunk processing error: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    @staticmethod
    def _extract_waveform_window(
        signal: NDArray[np.float64],
        effective_fs: float,
        step_idx: int = 0,
        window_ms: float = 600.0,
        max_bins: int = 2000,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract a scrolling audio waveform window normalized to [-1.0, 1.0] with peak decimation."""
        n_pts = int(signal.size)
        fs = max(1.0, float(effective_fs))
        if n_pts == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty

        chunk_samples = max(16, int(round((window_ms / 1000.0) * fs)))
        if n_pts <= chunk_samples:
            seg = signal
        else:
            max_start = max(1, n_pts - chunk_samples)
            start_idx = (step_idx * int(round(0.035 * fs))) % max_start
            seg = signal[start_idx : start_idx + chunk_samples]

        peak = float(np.max(np.abs(seg)))
        norm_stream = (seg / peak) if peak > 1e-12 else np.zeros_like(seg)
        n_seg = len(norm_stream)

        if n_seg > max_bins * 2:
            bin_size = n_seg // max_bins
            n_trim = bin_size * max_bins
            reshaped = norm_stream[:n_trim].reshape(max_bins, bin_size)
            y_min = np.min(reshaped, axis=1)
            y_max = np.max(reshaped, axis=1)
            y_dec = np.empty(max_bins * 2, dtype=np.float64)
            y_dec[0::2] = y_min
            y_dec[1::2] = y_max
            t_ms = np.linspace(0.0, (n_trim / fs) * 1000.0, max_bins * 2, dtype=np.float64)
            return t_ms, y_dec

        t_ms = (np.arange(n_seg, dtype=np.float64) / fs) * 1000.0
        return t_ms, norm_stream.astype(np.float64, copy=True)

    @pyqtSlot()
    def shutdown(self) -> None:
        """Stop streaming and cleanly shut down the DSPEngine worker pool."""
        self.stop_stream()
        if self._analysis_executor is not None:
            self._analysis_executor.shutdown(wait=True)
            self._analysis_executor = None
            self._analysis_future = None
        self.dsp_engine.shutdown(wait=True)
        logger.info("DSPStreamWorker shut down cleanly")

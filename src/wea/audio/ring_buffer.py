"""Thread-safe circular audio buffer backed by a pre-allocated NumPy array."""

from __future__ import annotations

import threading

import numpy as np

from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class AudioRingBuffer:
    """Thread-safe ring buffer over NumPy for continuous non-blocking audio streams.

    Designed for real-time producer-consumer decoupled pipelines where a low-latency
    audio callback writes chunks and a DSP worker or UI thread reads sliding windows.
    """

    def __init__(self, capacity: int, dtype: np.dtype | type = np.float64) -> None:
        """Initialize the circular buffer with a fixed sample capacity.

        Args:
            capacity: Maximum number of samples stored in the ring buffer.
            dtype: NumPy data type of the samples (default: np.float64).
        """
        if capacity <= 0:
            logger.error("Failed to initialize AudioRingBuffer: invalid capacity=%d", capacity)
            raise ValueError("capacity must be a positive integer")
        self._capacity = int(capacity)
        self._dtype = np.dtype(dtype)
        self._buffer = np.zeros(self._capacity, dtype=self._dtype)
        self._write_pos = 0
        self._size = 0
        self._lock = threading.Lock()
        logger.info(
            "AudioRingBuffer initialized with capacity=%d, dtype=%s",
            self._capacity,
            self._dtype,
        )

    @property
    def capacity(self) -> int:
        """Return total sample capacity of the ring buffer."""
        return self._capacity

    @property
    def size(self) -> int:
        """Return current number of valid readable samples in the buffer."""
        with self._lock:
            return self._size

    def clear(self) -> None:
        """Reset the buffer pointers and zero out memory."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._size = 0
        logger.debug("AudioRingBuffer cleared (capacity=%d)", self._capacity)

    def write(self, data: np.ndarray) -> None:
        """Write a 1D array of samples into the circular buffer, overwriting oldest if full."""
        samples = np.asarray(data, dtype=self._dtype).ravel()
        n = len(samples)
        if n == 0:
            return

        with self._lock:
            if n >= self._capacity:
                # Keep only the most recent `capacity` samples
                self._buffer[:] = samples[-self._capacity :]
                self._write_pos = 0
                self._size = self._capacity
                return

            first_part = min(n, self._capacity - self._write_pos)
            self._buffer[self._write_pos : self._write_pos + first_part] = samples[:first_part]

            second_part = n - first_part
            if second_part > 0:
                self._buffer[:second_part] = samples[first_part:]

            self._write_pos = (self._write_pos + n) % self._capacity
            self._size = min(self._capacity, self._size + n)

    def read_latest(self, num_samples: int) -> np.ndarray:
        """Return a contiguous copy of the most recent `num_samples` samples.

        If fewer than `num_samples` have been written so far, returns all available samples.
        """
        if num_samples <= 0:
            return np.empty(0, dtype=self._dtype)

        with self._lock:
            count = min(int(num_samples), self._size)
            if count == 0:
                return np.empty(0, dtype=self._dtype)

            start_pos = (self._write_pos - count) % self._capacity
            if start_pos + count <= self._capacity:
                return self._buffer[start_pos : start_pos + count].copy()

            first_chunk = self._buffer[start_pos:]
            second_chunk = self._buffer[: self._write_pos]
            return np.concatenate((first_chunk, second_chunk))

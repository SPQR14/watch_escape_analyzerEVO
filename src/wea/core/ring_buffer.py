"""Thread-safe pre-allocated circular audio buffer in pure NumPy."""

from __future__ import annotations

import threading

import numpy as np
from numpy.typing import DTypeLike, NDArray

from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)


class AudioRingBuffer:
    """Pre-allocated circular ring buffer in pure NumPy for real-time audio streaming.

    Avoids dynamic array concatenations during steady-state streaming by using
    pre-allocated backing arrays, slice assignments, and modular index arithmetic
    protected by a lightweight mutex lock.
    """

    DEFAULT_CAPACITY_SAMPLES: int = 480000

    def __init__(
        self,
        capacity_samples: int | None = None,
        *,
        capacity: int | None = None,
        dtype: DTypeLike = np.float64,
    ) -> None:
        """Initialize the circular buffer with a fixed sample capacity.

        Args:
            capacity_samples: Maximum number of samples stored in the ring buffer
                (e.g., 5 to 10 seconds of audio at 48 kHz / 96 kHz).
            capacity: Optional keyword alias for ``capacity_samples`` for backward compatibility.
            dtype: NumPy data type of the samples (defaults to ``np.float64``).
        """
        resolved_capacity = (
            capacity_samples
            if capacity_samples is not None
            else (capacity if capacity is not None else self.DEFAULT_CAPACITY_SAMPLES)
        )
        if resolved_capacity <= 0:
            logger.error(
                "Failed to initialize AudioRingBuffer: invalid capacity_samples=%d",
                resolved_capacity,
            )
            raise ValueError("capacity_samples must be a positive integer")

        self._capacity: int = int(resolved_capacity)
        self._dtype = np.dtype(dtype)
        self._buffer: NDArray[np.float64] = np.zeros(self._capacity, dtype=self._dtype)
        self._read_scratch: NDArray[np.float64] = np.zeros(self._capacity, dtype=self._dtype)
        self._write_pos: int = 0
        self._available_samples: int = 0
        self._total_samples_written: int = 0
        self._lock = threading.Lock()

        logger.info(
            "AudioRingBuffer initialized (capacity_samples=%d, dtype=%s)",
            self._capacity,
            self._dtype,
        )

    @property
    def capacity_samples(self) -> int:
        """Return total sample capacity of the ring buffer."""
        return self._capacity

    @property
    def capacity(self) -> int:
        """Return total sample capacity of the ring buffer (alias for ``capacity_samples``)."""
        return self._capacity

    @property
    def available_samples(self) -> int:
        """Return current number of valid readable samples in the buffer."""
        with self._lock:
            return self._available_samples

    @property
    def size(self) -> int:
        """Return current number of valid readable samples (alias for ``available_samples``)."""
        with self._lock:
            return self._available_samples

    @property
    def total_samples_written(self) -> int:
        """Return cumulative number of samples written since initialization or last reset."""
        with self._lock:
            return self._total_samples_written

    def clear(self) -> None:
        """Reset buffer pointers, counters, and zero out pre-allocated memory."""
        with self._lock:
            self._buffer.fill(0.0)
            self._read_scratch.fill(0.0)
            self._write_pos = 0
            self._available_samples = 0
            self._total_samples_written = 0
        logger.debug("AudioRingBuffer cleared (capacity_samples=%d)", self._capacity)

    reset = clear

    def write(self, chunk: NDArray[np.float64]) -> None:
        """Write a 1D chunk of audio samples into the ring buffer using slice assignments.

        If ``len(chunk)`` exceeds ``capacity_samples``, only the latest ``capacity_samples``
        are retained in the circular buffer while ``total_samples_written`` advances by the
        full chunk length.

        Args:
            chunk: 1D array of incoming audio samples.
        """
        samples = np.asarray(chunk, dtype=self._dtype).ravel()
        n = int(samples.size)
        if n == 0:
            return

        with self._lock:
            self._total_samples_written += n
            if n >= self._capacity:
                self._buffer[:] = samples[n - self._capacity :]
                self._write_pos = 0
                self._available_samples = self._capacity
                return

            first_part = min(n, self._capacity - self._write_pos)
            self._buffer[self._write_pos : self._write_pos + first_part] = samples[:first_part]

            second_part = n - first_part
            if second_part > 0:
                self._buffer[:second_part] = samples[first_part:]

            self._write_pos = (self._write_pos + n) % self._capacity
            self._available_samples = min(self._capacity, self._available_samples + n)

    def read_latest(self, num_samples: int | None = None) -> NDArray[np.float64]:
        """Read the most recent ``num_samples`` in chronological order without concatenation.

        Copies wrap-around segments into a pre-allocated scratch array using slice
        assignments and modular arithmetic before returning a contiguous view copy.

        Args:
            num_samples: Number of latest samples to retrieve. If None, returns all
                currently available samples.

        Returns:
            Contiguous 1D ``np.float64`` array of up to ``num_samples`` latest samples.
        """
        with self._lock:
            req = self._available_samples if num_samples is None else int(num_samples)
            if req <= 0 or self._available_samples == 0:
                return np.empty(0, dtype=self._dtype)

            count = min(req, self._available_samples)
            start_pos = (self._write_pos - count) % self._capacity

            first_part = min(count, self._capacity - start_pos)
            self._read_scratch[:first_part] = self._buffer[start_pos : start_pos + first_part]

            second_part = count - first_part
            if second_part > 0:
                self._read_scratch[first_part:count] = self._buffer[:second_part]

            return self._read_scratch[:count].copy()

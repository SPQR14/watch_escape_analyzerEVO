"""Tests for the P1-onset beat tracker (wea.core.beat_timing).

Tolerances: on the two-mode resonant synthesizer (6 kHz carrier, random mode phases, SNR
20-25 dB) the onset jitter measured over 30 s is ~40 us sigma, <= 90 us worst case.
"""

from __future__ import annotations

import numpy as np
import pytest

from wea.core.beat_timing import (
    BeatOnset,
    OnsetBeatTracker,
    OnsetConfig,
    beat_snr_db,
    detect_onsets,
)
from wea.core.synthetic import synthesize_escapement

FS = 48000
BPH = 21600
PERIOD = 3600.0 / BPH


def _times(beats: list[BeatOnset]) -> np.ndarray:
    return np.array([b.sample for b in beats]) / FS


def _match_errors_us(det_s: np.ndarray, truth_s: np.ndarray) -> np.ndarray:
    j = np.argmin(np.abs(det_s[:, None] - truth_s[None, :]), axis=1)
    return (det_s - truth_s[j]) * 1e6


@pytest.fixture(scope="module")
def clean_case() -> tuple[np.ndarray, np.ndarray]:
    return synthesize_escapement(seconds=20.0, snr_db=25.0, seed=3)


def test_onsets_hit_the_unlock_sound(clean_case) -> None:
    x, truth = clean_case
    beats = detect_onsets(x, FS, BPH, 51.0)
    assert len(beats) == len(truth)
    err = _match_errors_us(_times(beats), truth)
    assert abs(float(np.median(err))) < 30.0
    assert 1.4826 * float(np.median(np.abs(err - np.median(err)))) < 60.0
    assert float(np.max(np.abs(err - np.median(err)))) < 150.0
    np.testing.assert_array_equal([b.index for b in beats], np.arange(len(beats)))


@pytest.mark.parametrize("block", [128, 1024, 4096, 12000, "random"])
def test_streaming_equals_offline_without_duplicates(clean_case, block) -> None:
    """Block-wise processing with overlap gives exactly the offline beats, none duplicated."""
    x, _ = clean_case
    offline = detect_onsets(x, FS, BPH, 51.0)
    tracker = OnsetBeatTracker(FS, BPH, 51.0)
    rng = np.random.default_rng(0)
    pos, streamed = 0, []
    while pos < len(x):
        size = int(rng.integers(64, 20000)) if block == "random" else int(block)
        streamed += tracker.process(x[pos : pos + size], start_sample=pos)
        pos += size
    streamed += tracker.flush()
    assert [b.index for b in streamed] == [b.index for b in offline]
    np.testing.assert_allclose([b.sample for b in streamed], [b.sample for b in offline], atol=1e-6)
    assert np.all(np.diff([b.sample for b in streamed]) > 0.5 * PERIOD * FS)


def test_missing_beats_skip_indices_and_create_no_false_beats() -> None:
    x, truth = synthesize_escapement(seconds=30.0, rate_spd=-20.0, missing_fraction=0.1, seed=3)
    beats = detect_onsets(x, FS, BPH, 51.0)
    det = _times(beats)
    assert len(beats) == len(truth)
    assert np.max(np.abs(_match_errors_us(det, truth))) < 300.0
    steps = np.diff([b.index for b in beats])
    assert set(np.unique(steps)) <= {1, 2, 3, 4}
    assert np.sum(steps > 1) >= 5


def test_gate_rejects_mid_cycle_click_and_displaced_beat() -> None:
    x, truth = synthesize_escapement(seconds=10.0, snr_db=30.0, seed=4)
    x = x.copy()
    # Loud click halfway between two beats
    click_at = int((truth[20] + 0.5 * PERIOD) * FS)
    x[click_at : click_at + 48] += 5.0 * np.hanning(48)
    beats = detect_onsets(x, FS, BPH, 51.0)
    det = _times(beats)
    assert np.min(np.abs(det - click_at / FS)) > 0.02

    # Move beat 30 by +5 ms (outside the +-3 ms gate): it must be dropped, not accepted
    y, truth_y = synthesize_escapement(seconds=10.0, snr_db=30.0, seed=4)
    t30 = int(truth_y[30] * FS)
    seg = y[t30 - 200 : t30 + int(0.03 * FS)].copy()
    y[t30 - 200 : t30 + int(0.03 * FS)] = np.random.default_rng(0).normal(0, 1e-4, len(seg))
    y[t30 - 200 + 240 : t30 + int(0.03 * FS) + 240] += seg
    beats_y = detect_onsets(y, FS, BPH, 51.0)
    assert 30 not in [b.index for b in beats_y]
    assert 29 in [b.index for b in beats_y] and 31 in [b.index for b in beats_y]


def test_weak_unlock_locks_consistently_on_next_subevent() -> None:
    """P1 below the 25 % threshold: every onset sits on P2, at a constant offset."""
    x, truth = synthesize_escapement(
        seconds=20.0, sound_levels=(0.15, 0.6, 1.0), snr_db=20.0, decay_s=0.002, seed=5
    )
    err = _match_errors_us(_times(detect_onsets(x, FS, BPH, 51.0)), truth)
    assert np.median(err) > 1000.0  # locked on P2, a few ms after P1
    assert 1.4826 * float(np.median(np.abs(err - np.median(err)))) < 80.0


def test_discontinuity_starts_new_segment_and_reacquires(clean_case) -> None:
    x, _ = clean_case
    tracker = OnsetBeatTracker(FS, BPH, 51.0)
    first = tracker.process(x[: 5 * FS], start_sample=0)
    # Jump: 0.4 s of audio lost
    second = tracker.process(x[int(5.4 * FS) : 12 * FS], start_sample=int(5.4 * FS))
    second += tracker.flush()
    assert first and second
    assert {b.segment for b in first} == {0}
    assert {b.segment for b in second} == {1}
    assert len(second) >= int(6.0 / PERIOD) - 3


def test_band_is_configurable() -> None:
    narrow = OnsetBeatTracker(FS, BPH, config=OnsetConfig(band_low_hz=1000.0, band_high_hz=8000.0))
    default = OnsetBeatTracker(FS, BPH)
    assert not np.allclose(narrow._sos, default._sos)


def test_beat_snr_definition() -> None:
    assert beat_snr_db([10.0, 10.0, 10.0], [0.1, 0.1, 0.1]) == pytest.approx(40.0)
    assert beat_snr_db([], [1.0]) == 0.0
    assert beat_snr_db([1.0], [2.0]) == 0.0  # clipped at 0 dB


def test_onsets_carry_snr_inputs(clean_case) -> None:
    x, _ = clean_case
    beats = detect_onsets(x, FS, BPH, 51.0)
    snr = beat_snr_db([b.peak for b in beats], [b.noise for b in beats])
    assert 15.0 < snr < 60.0


def test_never_locks_on_noise_when_first_beats_are_missing() -> None:
    """Regression: with beat 0 missing the 1-period acquisition window held only noise
    (8 dB peaks) and the tracker locked mid-cycle forever (+160 s/d)."""
    x, truth = synthesize_escapement(seconds=30.0, snr_db=20.0, rate_spd=-20.0, seed=9)
    x = x.copy()
    rng = np.random.default_rng(3)
    holes = [0, 7, 15] + [k for k in range(16, len(truth)) if rng.random() < 0.12]
    for k in holes:
        a = int((truth[k] - 0.004) * FS)
        x[a : a + int(0.030 * FS)] = rng.normal(0.0, 1e-4, int(0.030 * FS))  # quieter hole
    present = np.delete(truth, holes)
    beats = detect_onsets(x, FS, BPH, 51.0)
    err = _match_errors_us(_times(beats), present)
    assert len(beats) >= len(present) - 2
    assert np.max(np.abs(err - np.median(err))) < 300.0

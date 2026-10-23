#!/usr/bin/env python3
"""Beat-timing diagnostic: record or load ~30 s, extract beat timestamps, fit a line and
show the residual histogram (us) to tell a timing-noise bell from discrete steps.

Examples:
    # Record 30 s from input device 2 (list devices with --list-devices)
    python scripts/diagnose_beats.py --record 30 --device 2 --bph 21600 --lift 51

    # Re-process a saved recording (.npy float audio or .wav)
    python scripts/diagnose_beats.py --file diagnostics/mido/audio.npy --bph 21600 --lift 51

    # Synthetic stand-in reproducing a weak-unlock, long-ringing case
    python scripts/diagnose_beats.py --synthetic mido_like --out tests/data/mido_like

Outputs in ``--out``: ``audio.npy`` (+ ``audio.wav`` when recording), ``beats_<detector>.npy``
(beat timestamps in s, for regression tests), ``residuals_<detector>.npz`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wea.core.beat_diagnostics import (  # noqa: E402
    classify_residuals,
    estimate_carrier_hz,
    fit_line_with_parity,
    residual_histogram,
    robust_line_fit,
    text_histogram,
)
from wea.core.beat_timing import detect_onsets  # noqa: E402
from wea.core.dsp_engine import DSPEngine  # noqa: E402
from wea.core.rate_regression import fit_rate_be  # noqa: E402
from wea.core.synthetic import synthesize_escapement  # noqa: E402

SYNTHETIC_PRESETS: dict[str, dict] = {
    # Weak unlock sound (20 % of the drop) and 2 ms case ring: reproduces +-30 s/d windows
    "mido_like": dict(
        rate_spd=5.0,
        beat_error_ms=0.23,
        amplitude_deg=299.0,
        sound_levels=(0.2, 0.5, 1.0),
        snr_db=15.0,
        decay_s=0.002,
    ),
    "clean": dict(rate_spd=5.0, beat_error_ms=0.3, amplitude_deg=290.0, snr_db=25.0),
}


def load_audio(path: Path) -> tuple[np.ndarray, int | None]:
    """Load mono float audio from .npy or .wav; returns (samples, fs or None)."""
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float64).ravel(), None
    from scipy.io import wavfile

    fs, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        # Use the channel with most energy (averaging channels can cancel)
        data = data[:, int(np.argmax(np.sum(data.astype(np.float64) ** 2, axis=0)))]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    return data.astype(np.float64), int(fs)


def record_audio(seconds: float, device: int | None, fs: int) -> np.ndarray:
    """Record mono audio with sounddevice (high latency: robustness over latency)."""
    import sounddevice as sd

    print(f"Recording {seconds:.0f} s from device {device} at {fs} Hz ...", flush=True)
    data = sd.rec(
        int(round(seconds * fs)),
        samplerate=fs,
        channels=1,
        dtype="float32",
        device=device,
        latency="high",
    )
    sd.wait()
    return data[:, 0].astype(np.float64)


def engine_beats(audio: np.ndarray, fs: int, bph: int, lift: float) -> np.ndarray:
    """Beat timestamps of the window-based DSPEngine (0.35 P1 + 0.65 P3 envelope peaks)."""
    with DSPEngine(bph=bph, lift_angle=lift, sample_rate=fs, auto_bph=False) as engine:
        return np.asarray(engine.process_window(audio).beat_times_sec, dtype=np.float64)


def engine_window_rates(audio: np.ndarray, fs: int, bph: int, lift: float) -> np.ndarray:
    """Rate of every 3 s window stepped by 250 ms, exactly as the live app computes it."""
    rates = []
    with DSPEngine(bph=bph, lift_angle=lift, sample_rate=fs, auto_bph=False) as engine:
        for start in range(0, len(audio) - 3 * fs + 1, fs // 4):
            rates.append(engine.process_window(audio[start : start + 3 * fs]).rate_spd)
    return np.asarray(rates, dtype=np.float64)


def onset_beats(audio: np.ndarray, fs: int, bph: int, lift: float) -> np.ndarray:
    """P1 onset timestamps of the gated onset tracker (wea.core.beat_timing)."""
    return np.asarray([b.sample for b in detect_onsets(audio, fs, bph, lift)]) / float(fs)


WINDOW_S = 15.0


def onset_window_rates(audio: np.ndarray, fs: int, bph: int, lift: float) -> np.ndarray:
    """Regression rate (5-MAD rejection) over every ``WINDOW_S`` window stepped by 250 ms,
    as the live app now computes it (``wea.core.rate_regression.fit_rate_be``)."""
    times = onset_beats(audio, fs, bph, lift)
    period = 3600.0 / bph
    rates = []
    for start in np.arange(0.0, len(audio) / fs - WINDOW_S + 1e-9, 0.25):
        sel = times[(times >= start) & (times < start + WINDOW_S)]
        fit = fit_rate_be(sel, period)
        if fit is not None:
            rates.append(fit.rate_spd)
    return np.asarray(rates, dtype=np.float64)


DETECTORS = {
    "engine": (engine_beats, engine_window_rates),
    "onset": (onset_beats, onset_window_rates),
}


def analyze(audio: np.ndarray, fs: int, bph: int, lift: float, detector: str, out: Path) -> dict:
    """Run one detector, print the diagnosis and save its artefacts."""
    beats_fn, rates_fn = DETECTORS[detector]
    period = 3600.0 / bph
    times = beats_fn(audio, fs, bph, lift)
    np.save(out / f"beats_{detector}.npy", times)

    ols = fit_line_with_parity(times, period)
    fit, inliers = robust_line_fit(times, period)
    carrier = estimate_carrier_hz(audio, fs)
    cls = classify_residuals(fit.residuals_us, carrier_hz=carrier)
    edges, counts = residual_histogram(fit.residuals_us[inliers])
    outliers = fit.residuals_us[~inliers]
    np.savez(out / f"residuals_{detector}.npz", n=fit.indices, t=times, r_us=fit.residuals_us)
    rates = rates_fn(audio, fs, bph, lift)

    print(f"\n=== detector: {detector} ===")
    print(
        f"beats: {len(times)}  ({len(times) / (len(audio) / fs):.2f}/s, nominal {bph / 3600:.2f}/s)"
    )
    print(
        f"line fit over {len(audio) / fs:.1f} s (OLS, all beats): rate {ols.rate_spd:+.2f} s/d, "
        f"BE {ols.beat_error_ms:.3f} ms"
    )
    print(
        f"robust fit ({int(inliers.sum())}/{len(times)} inliers, 5 MAD): rate "
        f"{fit.rate_spd:+.2f} s/d, BE {fit.beat_error_ms:.3f} ms, "
        f"residual sigma {cls.robust_sigma_us:.1f} us"
    )
    print(
        f"windowed rate ({len(rates)} windows): mean {np.mean(rates):+.2f}  std {np.std(rates):.2f}"
        f"  min {np.min(rates):+.2f}  max {np.max(rates):+.2f} s/d"
    )
    print(f"carrier estimate: {carrier:.0f} Hz (period {1e6 / max(carrier, 1.0):.0f} us)")
    print(f"VERDICT: {cls.summary}")
    print(f"residual histogram of inliers ({edges[1] - edges[0]:.0f} us bins):")
    print(text_histogram(edges, counts))
    if len(outliers):
        print(
            f"outliers: {len(outliers)} beats, residuals {np.min(outliers) / 1000:+.2f} .. "
            f"{np.max(outliers) / 1000:+.2f} ms (median {np.median(outliers) / 1000:+.2f} ms)"
        )
    return {
        "detector": detector,
        "n_beats": int(len(times)),
        "ols_rate_spd": ols.rate_spd,
        "fit_rate_spd": fit.rate_spd,
        "n_inliers": int(inliers.sum()),
        "fit_beat_error_ms": fit.beat_error_ms,
        "residual_sigma_us": cls.robust_sigma_us,
        "classification": cls.kind,
        "summary": cls.summary,
        "cluster_centers_us": cls.cluster_centers_us,
        "cluster_fractions": cls.cluster_fractions,
        "carrier_hz": carrier,
        "window_rate_mean_spd": float(np.mean(rates)),
        "window_rate_std_spd": float(np.std(rates)),
        "window_rate_min_spd": float(np.min(rates)),
        "window_rate_max_spd": float(np.max(rates)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--record", type=float, metavar="SECONDS", help="record from --device")
    src.add_argument("--file", type=Path, help=".npy (float audio) or .wav recording")
    src.add_argument("--synthetic", choices=sorted(SYNTHETIC_PRESETS), help="synthetic preset")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--fs", type=int, default=48000, help="sample rate (record/.npy input)")
    ap.add_argument("--bph", type=int, default=21600)
    ap.add_argument("--lift", type=float, default=51.0, help="lift angle in degrees")
    ap.add_argument("--seconds", type=float, default=30.0, help="synthetic duration")
    ap.add_argument("--detector", choices=[*DETECTORS, "all"], default="all")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--window", type=float, default=15.0, help="onset regression window in s (default 15)"
    )
    args = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    global WINDOW_S
    WINDOW_S = float(args.window)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    out = args.out or ROOT / "diagnostics" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    fs = int(args.fs)
    meta: dict = {"bph": args.bph, "lift_angle": args.lift}
    if args.record:
        audio = record_audio(args.record, args.device, fs)
        from scipy.io import wavfile

        wavfile.write(out / "audio.wav", fs, audio.astype(np.float32))
        meta["source"] = f"recorded device {args.device}"
    elif args.file:
        audio, file_fs = load_audio(args.file)
        fs = file_fs or fs
        meta["source"] = str(args.file)
    else:
        preset = args.synthetic or "mido_like"
        audio, _ = synthesize_escapement(
            bph=args.bph, lift_angle=args.lift, seconds=args.seconds, fs=fs, seed=1,
            **SYNTHETIC_PRESETS[preset],
        )  # fmt: skip
        meta["source"] = f"SYNTHETIC preset '{preset}' (not a real recording)"
        meta["synthetic_truth"] = SYNTHETIC_PRESETS[preset]
    np.save(out / "audio.npy", audio.astype(np.float32))
    meta.update(fs=fs, seconds=len(audio) / fs)
    print(f"source: {meta['source']}  ({meta['seconds']:.1f} s @ {fs} Hz) -> {out}")

    detectors = list(DETECTORS) if args.detector == "all" else [args.detector]
    meta["results"] = [analyze(audio, fs, args.bph, args.lift, d, out) for d in detectors]
    (out / "report.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

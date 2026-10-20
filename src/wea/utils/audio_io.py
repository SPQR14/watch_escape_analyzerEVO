"""Audio encoding and persistence utilities supporting FLAC (default) and Ogg Vorbis (fallback)."""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from wea.utils.logger_setup import LoggerSetup

logger = LoggerSetup.setup_logger(__name__)

# libsndfile format constants (from sndfile.h)
SF_FORMAT_FLAC: int = 0x170000
SF_FORMAT_OGG: int = 0x200000
SF_FORMAT_PCM_16: int = 0x0002
SF_FORMAT_VORBIS: int = 0x0060
SFM_WRITE: int = 0x20


class _SF_INFO(ctypes.Structure):  # noqa: N801
    """ctypes mapping for libsndfile ``SF_INFO`` struct."""

    _fields_ = [
        ("frames", ctypes.c_int64),
        ("samplerate", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("format", ctypes.c_int),
        ("sections", ctypes.c_int),
        ("seekable", ctypes.c_int),
    ]


def _load_libsndfile() -> ctypes.CDLL | None:
    """Locate and load ``libsndfile`` shared library if available on the host system."""
    candidates: list[str] = []
    found = ctypes.util.find_library("sndfile")
    if found:
        candidates.append(found)
    candidates.extend(
        [
            "/opt/homebrew/lib/libsndfile.dylib",
            "/usr/local/lib/libsndfile.dylib",
            "libsndfile.dylib",
            "libsndfile.so.1",
            "libsndfile.so",
            "sndfile.dll",
        ]
    )
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            lib.sf_format_check.argtypes = [ctypes.POINTER(_SF_INFO)]
            lib.sf_format_check.restype = ctypes.c_int
            lib.sf_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(_SF_INFO)]
            lib.sf_open.restype = ctypes.c_void_p
            lib.sf_writef_double.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int64,
            ]
            lib.sf_writef_double.restype = ctypes.c_int64
            lib.sf_close.argtypes = [ctypes.c_void_p]
            lib.sf_close.restype = ctypes.c_int
            lib.sf_strerror.argtypes = [ctypes.c_void_p]
            lib.sf_strerror.restype = ctypes.c_char_p
            return lib
        except OSError:
            continue
    return None


_SNDFILE_LIB: ctypes.CDLL | None = _load_libsndfile()


def is_flac_supported() -> bool:
    """Return True if FLAC audio encoding is supported on the current system."""
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        if "FLAC" in sf.available_formats():
            return True
    except Exception:
        pass

    if _SNDFILE_LIB is not None:
        info = _SF_INFO(
            frames=0,
            samplerate=48000,
            channels=1,
            format=SF_FORMAT_FLAC | SF_FORMAT_PCM_16,
            sections=0,
            seekable=0,
        )
        if _SNDFILE_LIB.sf_format_check(ctypes.byref(info)) != 0:
            return True

    if shutil.which("flac") or shutil.which("ffmpeg"):
        return True

    return False


def is_vorbis_supported() -> bool:
    """Return True if Ogg Vorbis audio encoding is supported on the current system."""
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        if "OGG" in sf.available_formats() and "VORBIS" in sf.available_subtypes("OGG"):
            return True
    except Exception:
        pass

    if _SNDFILE_LIB is not None:
        info = _SF_INFO(
            frames=0,
            samplerate=48000,
            channels=1,
            format=SF_FORMAT_OGG | SF_FORMAT_VORBIS,
            sections=0,
            seekable=0,
        )
        if _SNDFILE_LIB.sf_format_check(ctypes.byref(info)) != 0:
            return True

    if shutil.which("ffmpeg") or shutil.which("oggenc"):
        return True

    return False


def preferred_audio_format() -> str:
    """Return ``'flac'`` if FLAC encoding is available, otherwise ``'ogg'`` (Vorbis)."""
    if is_flac_supported():
        return "flac"
    return "ogg"


def _normalize_samples(samples: np.ndarray) -> np.ndarray:
    """Ensure 1D contiguous float64 array scaled safely within [-1.0, 1.0]."""
    arr = np.asarray(samples, dtype=np.float64).ravel()
    if len(arr) == 0:
        return np.zeros(0, dtype=np.float64)
    peak = float(np.max(np.abs(arr)))
    if peak > 1.0:
        arr = arr / peak
    return np.ascontiguousarray(np.clip(arr, -1.0, 1.0), dtype=np.float64)


def _write_via_libsndfile(
    out_path: Path, samples: np.ndarray, sample_rate: int, fmt_flag: int
) -> bool:
    """Write audio samples using ``libsndfile`` via ctypes."""
    if _SNDFILE_LIB is None:
        return False
    info = _SF_INFO(
        frames=0,
        samplerate=int(sample_rate),
        channels=1,
        format=fmt_flag,
        sections=0,
        seekable=0,
    )
    if _SNDFILE_LIB.sf_format_check(ctypes.byref(info)) == 0:
        return False

    snd = _SNDFILE_LIB.sf_open(str(out_path).encode("utf-8"), SFM_WRITE, ctypes.byref(info))
    if not snd:
        err_msg = _SNDFILE_LIB.sf_strerror(None)
        logger.error(
            "libsndfile failed to open '%s': %s",
            out_path,
            err_msg.decode("utf-8", errors="replace") if err_msg else "unknown error",
        )
        return False

    try:
        ptr = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        written = _SNDFILE_LIB.sf_writef_double(snd, ptr, len(samples))
        if written != len(samples):
            logger.warning(
                "libsndfile partial write to '%s': %d / %d frames",
                out_path,
                written,
                len(samples),
            )
            return False
        return True
    finally:
        _SNDFILE_LIB.sf_close(snd)


def _write_via_cli(out_path: Path, samples: np.ndarray, sample_rate: int, target_fmt: str) -> bool:
    """Fallback encoder using ``flac``, ``oggenc``, or ``ffmpeg`` CLI tools."""
    pcm16 = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        with wave.open(str(tmp_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())

        if target_fmt == "flac" and shutil.which("flac"):
            cmd = ["flac", "-f", "-s", str(tmp_wav), "-o", str(out_path)]
            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0 and out_path.exists():
                return True

        if target_fmt == "ogg" and shutil.which("oggenc"):
            cmd = ["oggenc", "-Q", str(tmp_wav), "-o", str(out_path)]
            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0 and out_path.exists():
                return True

        if shutil.which("ffmpeg"):
            codec = "flac" if target_fmt == "flac" else "libvorbis"
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(tmp_wav),
                "-c:a",
                codec,
                str(out_path),
            ]
            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0 and out_path.exists():
                return True
    finally:
        tmp_wav.unlink(missing_ok=True)

    return False


def save_audio_recording(
    file_path: Path | str,
    samples: np.ndarray,
    sample_rate: int,
    prefer_flac: bool = True,
) -> Path:
    """Encode and save a 1D audio buffer in FLAC (default) or Ogg Vorbis (fallback).

    Args:
        file_path: Destination file path.
        samples: 1D NumPy array of audio samples.
        sample_rate: Sampling frequency in Hz.
        prefer_flac: If True, defaults to FLAC when supported, falling back to Vorbis.

    Returns:
        Resolved ``Path`` of the saved audio file.
    """
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm = _normalize_samples(samples)

    ext = out_path.suffix.lower().lstrip(".")
    if ext not in ("flac", "ogg"):
        ext = "flac" if (prefer_flac and is_flac_supported()) else "ogg"
        out_path = out_path.with_suffix(f".{ext}")
    elif ext == "flac" and not is_flac_supported():
        logger.warning("FLAC encoding unavailable; falling back to Ogg Vorbis for '%s'", out_path)
        ext = "ogg"
        out_path = out_path.with_suffix(".ogg")

    logger.info(
        "Saving audio recording (%d samples @ %d Hz, %.2f s) to '%s' [format=%s]",
        len(norm),
        sample_rate,
        len(norm) / float(sample_rate) if sample_rate > 0 else 0.0,
        out_path,
        ext.upper(),
    )

    # 1. Try soundfile package if installed
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        subtype = "PCM_16" if ext == "flac" else "VORBIS"
        sf.write(str(out_path), norm, int(sample_rate), format=ext.upper(), subtype=subtype)
        logger.info("Audio recording saved via soundfile to '%s'", out_path)
        return out_path
    except Exception:
        pass

    # 2. Try libsndfile via ctypes
    if ext == "flac":
        if _write_via_libsndfile(out_path, norm, sample_rate, SF_FORMAT_FLAC | SF_FORMAT_PCM_16):
            logger.info("Audio recording saved in FLAC via libsndfile to '%s'", out_path)
            return out_path
        if _write_via_cli(out_path, norm, sample_rate, "flac"):
            logger.info("Audio recording saved in FLAC via CLI encoder to '%s'", out_path)
            return out_path
        logger.warning("FLAC write failed; falling back to Ogg Vorbis")
        ext = "ogg"
        out_path = out_path.with_suffix(".ogg")

    if ext == "ogg":
        if _write_via_libsndfile(out_path, norm, sample_rate, SF_FORMAT_OGG | SF_FORMAT_VORBIS):
            logger.info("Audio recording saved in Ogg Vorbis via libsndfile to '%s'", out_path)
            return out_path
        if _write_via_cli(out_path, norm, sample_rate, "ogg"):
            logger.info("Audio recording saved in Ogg Vorbis via CLI encoder to '%s'", out_path)
            return out_path

    msg = f"Could not encode audio to FLAC or Ogg Vorbis at '{out_path}'"
    logger.error(msg)
    raise RuntimeError(msg)

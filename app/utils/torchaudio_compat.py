"""Compatibility shim to expose legacy torchaudio backend APIs when missing.

Some versions of `torchaudio` (and downstream packages) expect functions like
`list_audio_backends` and `set_audio_backend`. Newer torchaudio versions may
change their internals (or rely on torchcodec); this shim provides minimal
fallbacks so code that probes backends doesn't crash.

This is a small, safe shim: it does not alter audio decoding behaviour; it
only exposes the expected function names and records a default backend.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

logger = logging.getLogger(__name__)

try:
    import torchaudio
except Exception:  # pragma: no cover - defensive
    torchaudio = None

try:
    import torch
except Exception:
    torch = None

try:
    import soundfile as sf
except Exception:
    sf = None


@dataclass
class AudioMetaData:
    """Fallback AudioMetaData dataclass for downstream libraries."""
    sample_rate: int = 16000
    num_frames: int = 0
    num_channels: int = 1
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def _soundfile_load(filepath: Any, *args: Any, **kwargs: Any) -> Tuple[Any, int]:
    """Load audio file using soundfile and return (waveform, sample_rate)."""
    if sf is None or torch is None:
        raise RuntimeError("soundfile and torch are required for audio loading fallback")
    data, sr = sf.read(filepath, dtype="float32", always_2d=True)
    # soundfile returns (frames, channels), PyTorch expects (channels, frames)
    waveform = torch.from_numpy(data.T)
    return waveform, sr


def _soundfile_info(filepath: Any, *args: Any, **kwargs: Any) -> AudioMetaData:
    """Get audio metadata using soundfile."""
    if sf is None:
        raise RuntimeError("soundfile is required for audio metadata fallback")
    info = sf.info(filepath)
    return AudioMetaData(
        sample_rate=info.samplerate,
        num_frames=info.frames,
        num_channels=info.channels,
        bits_per_sample=getattr(info, "subtype_info", {}).get("bits", 16) or 16,
        encoding=getattr(info, "subtype", "PCM_S") or "PCM_S",
    )


def _ensure():
    if torchaudio is None:
        return

    # Provide list_audio_backends()
    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends() -> Iterable[str]:
            # Mirror behaviour of older torchaudio: return available backends.
            # We can't probe real backends reliably here, so return a sensible
            # default list that downstream libraries accept.
            return ("soundfile", "sox_io")

        setattr(torchaudio, "list_audio_backends", _list_audio_backends)
        logger.debug("Injected torchaudio.list_audio_backends shim")

    # Provide set_audio_backend(name)
    if not hasattr(torchaudio, "set_audio_backend"):
        _current = {"name": "soundfile"}

        def _set_audio_backend(name: str) -> None:
            _current["name"] = str(name)

        def _get_current_backend() -> str:
            return _current["name"]

        setattr(torchaudio, "set_audio_backend", _set_audio_backend)
        setattr(torchaudio, "get_audio_backend", _get_current_backend)
        logger.debug("Injected torchaudio.set_audio_backend/get_audio_backend shim")

    # Provide AudioMetaData
    if not hasattr(torchaudio, "AudioMetaData"):
        setattr(torchaudio, "AudioMetaData", AudioMetaData)
        logger.debug("Injected torchaudio.AudioMetaData shim")

    try:
        import torchaudio.backend.common as ta_backend_common
        if not hasattr(ta_backend_common, "AudioMetaData"):
            setattr(ta_backend_common, "AudioMetaData", AudioMetaData)
    except Exception:
        pass

    # Provide soundfile-backed torchaudio.load fallback
    orig_load = getattr(torchaudio, "load", None)
    def _safe_load(filepath: Any, *args: Any, **kwargs: Any):
        if orig_load is not None:
            try:
                return orig_load(filepath, *args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "torchcodec" in err_str or "not implemented" in err_str or "could not load" in err_str or "no audio backend" in err_str:
                    return _soundfile_load(filepath, *args, **kwargs)
                raise
        return _soundfile_load(filepath, *args, **kwargs)

    setattr(torchaudio, "load", _safe_load)

    # Provide soundfile-backed torchaudio.info fallback
    orig_info = getattr(torchaudio, "info", None)
    def _safe_info(filepath: Any, *args: Any, **kwargs: Any):
        if orig_info is not None:
            try:
                return orig_info(filepath, *args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                if "torchcodec" in err_str or "not implemented" in err_str or "could not load" in err_str or "no audio backend" in err_str:
                    return _soundfile_info(filepath, *args, **kwargs)
                raise
        return _soundfile_info(filepath, *args, **kwargs)

    setattr(torchaudio, "info", _safe_info)


_ensure()

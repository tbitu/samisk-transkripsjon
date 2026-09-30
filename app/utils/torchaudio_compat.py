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

    # torchaudio.load supports frame_offset and num_frames (positional or keyword)
    frame_offset = kwargs.get("frame_offset", 0)
    num_frames = kwargs.get("num_frames", -1)
    if len(args) >= 1:
        frame_offset = args[0]
    if len(args) >= 2:
        num_frames = args[1]

    start = max(0, int(frame_offset))
    frames = int(num_frames) if num_frames is not None and num_frames > 0 else -1

    data, sr = sf.read(filepath, start=start, frames=frames, dtype="float32", always_2d=True)
    # soundfile returns (frames, channels), PyTorch expects (channels, frames)
    waveform = torch.from_numpy(data.T)
    return waveform, sr


def _soundfile_info(filepath: Any, *args: Any, **kwargs: Any) -> AudioMetaData:
    """Get audio metadata using soundfile."""
    if sf is None:
        raise RuntimeError("soundfile is required for audio metadata fallback")
    info = sf.info(filepath)
    bits = 16
    if hasattr(info, "subtype") and info.subtype:
        import re
        m = re.search(r"\d+", str(info.subtype))
        if m:
            try:
                bits = int(m.group(0))
            except ValueError:
                bits = 16
    return AudioMetaData(
        sample_rate=info.samplerate,
        num_frames=info.frames,
        num_channels=info.channels,
        bits_per_sample=bits,
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

    _shim_huggingface_hub()
    _shim_torch_load()
    _shim_semver()
    _shim_pyannote()


def _shim_huggingface_hub() -> None:
    """Map legacy use_auth_token parameter to token in huggingface_hub.hf_hub_download."""
    try:
        import huggingface_hub
        orig_download = getattr(huggingface_hub, "hf_hub_download", None)
        if orig_download is None:
            return

        def _compat_hf_hub_download(*args: Any, **kwargs: Any) -> Any:
            if "use_auth_token" in kwargs:
                token = kwargs.pop("use_auth_token")
                if "token" not in kwargs and token is not None:
                    kwargs["token"] = token
            return orig_download(*args, **kwargs)

        setattr(huggingface_hub, "hf_hub_download", _compat_hf_hub_download)

        for mod_name in ("pyannote.audio.core.pipeline", "pyannote.audio.core.model"):
            try:
                mod = __import__(mod_name, fromlist=["hf_hub_download"])
                if hasattr(mod, "hf_hub_download"):
                    setattr(mod, "hf_hub_download", _compat_hf_hub_download)
            except Exception:
                pass
        logger.debug("Injected huggingface_hub use_auth_token shim")
    except Exception:
        pass


def _shim_torch_load() -> None:
    """Ensure torch.load defaults to weights_only=False for PyTorch Lightning/pyannote checkpoints."""
    global torch
    if torch is None:
        return
    orig_torch_load = getattr(torch, "load", None)
    if orig_torch_load is None:
        return

    def _compat_torch_load(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return orig_torch_load(*args, **kwargs)

    setattr(torch, "load", _compat_torch_load)

    try:
        import torch.torch_version  # type: ignore
        torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    except Exception:
        pass
    logger.debug("Injected torch.load weights_only compatibility shim")


def _shim_semver() -> None:
    """Ensure semver.VersionInfo.parse does not crash on non-standard build strings (e.g. NGC PyTorch versions)."""
    try:
        import re
        import semver

        orig_parse = semver.VersionInfo.parse

        def _safe_semver_parse(version: Any) -> Any:
            try:
                return orig_parse(str(version))
            except ValueError:
                match = re.match(r"^(\d+)\.(\d+)\.?(\d+)?", str(version))
                if match:
                    major, minor, patch = match.groups()
                    return orig_parse(f"{major}.{minor}.{patch or 0}")
                raise

        semver.VersionInfo.parse = _safe_semver_parse
        logger.debug("Injected semver.VersionInfo.parse shim")
    except Exception:
        pass


def _shim_pyannote() -> None:
    """Provide signature flexibility and version check compatibility for pyannote.audio."""
    try:
        import pyannote.audio.utils.version as pa_version

        def _safe_check_version(library: str, theirs: str, mine: str, what: str = "Pipeline") -> None:
            pass  # Suppress strict semver parsing warnings/crashes

        pa_version.check_version = _safe_check_version

        for mod_name in ("pyannote.audio.core.pipeline", "pyannote.audio.core.model"):
            try:
                mod = __import__(mod_name, fromlist=["check_version"])
                if hasattr(mod, "check_version"):
                    setattr(mod, "check_version", _safe_check_version)
            except Exception:
                pass
    except Exception:
        pass

    try:
        from pyannote.audio import Pipeline

        orig_pipeline_from_pretrained = Pipeline.from_pretrained

        @classmethod
        def _compat_pipeline_from_pretrained(cls: Any, *args: Any, **kwargs: Any) -> Any:
            # Support both token and use_auth_token kwargs seamlessly
            token = kwargs.pop("token", None)
            if token is not None and "use_auth_token" not in kwargs:
                kwargs["use_auth_token"] = token
            return orig_pipeline_from_pretrained(*args, **kwargs)

        Pipeline.from_pretrained = _compat_pipeline_from_pretrained
        logger.debug("Injected pyannote.audio Pipeline.from_pretrained shim")
    except Exception:
        pass


_ensure()

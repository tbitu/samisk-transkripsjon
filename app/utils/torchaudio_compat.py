"""Compatibility shim to expose legacy torchaudio backend APIs when missing.

Some versions of `torchaudio` (and downstream packages) expect functions like
`list_audio_backends` and `set_audio_backend`. Newer torchaudio versions may
change their internals (or rely on torchcodec); this shim provides minimal
fallbacks so code that probes backends doesn't crash.

This is a small, safe shim: it does not alter audio decoding behaviour; it
only exposes the expected function names and records a default backend.
"""
from __future__ import annotations

import types
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

try:
    import torchaudio
except Exception:  # pragma: no cover - defensive
    torchaudio = None


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


_ensure()

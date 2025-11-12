"""Sentence-level speech segmentation utilities with speaker diarization support."""
from __future__ import annotations

import logging
import math
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import ffmpeg
import numpy as np
import soundfile as sf
import torch

from .audio_chunker import AudioChunk, chunk_audio_file

logger = logging.getLogger(__name__)

DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
TARGET_SAMPLE_RATE = 16_000
MAX_SENTENCE_DURATION_MS = 20_000
MIN_SENTENCE_DURATION_MS = 1_200
QUESTION_PITCH_DELTA = 18.0
QUESTION_MIN_FRAMES = 5


@dataclass(frozen=True)
class SentenceChunk:
    """Metadata about an extracted single-sentence audio snippet."""

    path: Path
    start_ms: int
    end_ms: int
    speaker: Optional[str]
    is_question: bool

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def prepare_sentence_chunks(source: Path, output_dir: Path) -> List[SentenceChunk]:
    """Extract sentence-sized chunks with speaker labels when available."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Allow disabling diarization via environment when desired. Default: enabled.
    enable = os.environ.get("ENABLE_DIARIZATION", "1")
    if enable.lower() in ("0", "false", "no"):
        logger.info("Diarization disabled via ENABLE_DIARIZATION; using fallback segmentation")
        return _fallback_chunks(source, output_dir)

    try:
        diarization = _run_diarization(source)
        diarized_chunks = list(_chunks_from_diarization(source, output_dir, diarization))
        if diarized_chunks:
            return diarized_chunks
        logger.warning("Diarization yielded no chunks; falling back to silence-based split")
    except Exception as exc:  # pragma: no cover - best-effort fallback
        # If diarization is explicitly required and we appear to have a token,
        # prefer to fail loudly so the calling job can surface the error instead
        # of silently falling back.
        required = os.environ.get("DIARIZATION_REQUIRED", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        if required and _has_hf_token():
            logger.error("Diarization required but failed: %s", exc)
            raise

        logger.warning("Speaker diarization unavailable (%s). Using fallback segmentation.", exc)

    return _fallback_chunks(source, output_dir)


def _has_hf_token() -> bool:
    """Return True if an HF token is available via env or ./hf_token file."""
    token = (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if token:
        return True
    token_file = Path(__file__).resolve().parents[1] / "hf_token"
    return token_file.exists() and token_file.read_text().strip() != ""


def _run_diarization(source: Path):
    pipeline = _load_diarization_pipeline()
    return pipeline(str(source))


def _iter_diarization_turns(diarization) -> Iterable[tuple[float, float, str]]:
    """Yield (start_s, end_s, speaker) tuples for different pyannote outputs.

    Supports older `Annotation` objects (itertracks) and newer
    `DiarizeOutput`-like dicts with 'segments' or 'annotated' fields.
    """
    # Preferred attributes on modern pyannote pipelines.
    if hasattr(diarization, "speaker_diarization"):
        annotation = getattr(diarization, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = diarization.speaker_diarization
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            yield float(turn.start), float(turn.end), speaker
        return

    # If the object has itertracks (older API), use it directly
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            yield float(turn.start), float(turn.end), speaker
        return

    # If it's a mapping-like object with 'segments'
    if isinstance(diarization, dict):
        if "segments" in diarization:
            for seg in diarization["segments"]:
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                sp = seg.get("speaker") or seg.get("label") or "unknown"
                yield s, e, sp
            return

    # A DiarizeOutput may expose .annotated or .labels; try common attributes
    for attr in ("annotated", "segments", "labels", "labels_"):
        if hasattr(diarization, attr):
            container = getattr(diarization, attr)
            try:
                for item in container:
                    # item may be (segment, label) or dict-like
                    if isinstance(item, tuple) and len(item) >= 2:
                        seg, label = item[0], item[-1]
                        s = float(getattr(seg, "start", seg.get("start", 0.0)))
                        e = float(getattr(seg, "end", seg.get("end", 0.0)))
                        yield s, e, label
                    elif isinstance(item, dict):
                        s = float(item.get("start", 0.0))
                        e = float(item.get("end", 0.0))
                        sp = item.get("speaker") or item.get("label") or "unknown"
                        yield s, e, sp
            except Exception:
                # Fallthrough to last-ditch attempt below
                pass

    # If nothing matched, raise a helpful error so caller can fallback
    raise RuntimeError("Unsupported diarization output format: %r" % (type(diarization),))


@lru_cache(maxsize=1)
def _load_diarization_pipeline():  # pragma: no cover - external model loading
    try:
        from pyannote.audio import Pipeline
    except Exception as exc:  # ImportError or other runtime issues
        raise RuntimeError("pyannote.audio is required for speaker diarization") from exc

    token = (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if not token:
        # fall back to a repository-local token file if present (hf_token)
        token_file = Path(__file__).resolve().parents[1] / "hf_token"
        if token_file.exists():
            token = token_file.read_text().strip()

    if not token:
        raise RuntimeError(
            "Missing authentication token for pyannote.audio. Set PYANNOTE_AUTH_TOKEN or place token in ./hf_token."
        )

    # Instead of guessing keyword names, register the token with the
    # Hugging Face hub client which lets Pipeline.from_pretrained() use the
    # authenticated session regardless of the function signature. This
    # avoids mismatch issues between `token` and `use_auth_token` kwargs.
    if token:
        try:
            # Import locally to keep the dependency optional for users who
            # won't use diarization.
            from huggingface_hub import login as _hf_login

            _hf_login(token)
        except Exception:
            # If global login fails, proceed and let the subsequent download
            # raise a clear error that will be handled by the caller.
            logger.debug("Failed to register HF token via huggingface_hub.login")

    # Now call Pipeline.from_pretrained without passing auth kwargs so we
    # avoid signature mismatches in different pyannote versions.
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


def _chunks_from_diarization(source: Path, output_dir: Path, diarization) -> Iterable[SentenceChunk]:
    total_duration_ms = int(round(_probe_duration_ms(source)))
    chunk_index = 0

    for start_s, end_s, speaker in _iter_diarization_turns(diarization):
        start_ms = int(round(max(0.0, start_s) * 1000))
        end_ms = int(round(min(end_s, total_duration_ms / 1000.0) * 1000))

        if end_ms - start_ms < MIN_SENTENCE_DURATION_MS:
            continue

        for sub_start, sub_end in _segment_turn(start_ms, end_ms):
            safe_start, safe_end = _clamp(s=sub_start, e=sub_end, limit=total_duration_ms)
            if safe_end - safe_start < MIN_SENTENCE_DURATION_MS:
                continue

            target_path = output_dir / f"sentence_{chunk_index:04d}.wav"
            _extract_segment(source, safe_start, safe_end, target_path)
            is_question = _estimate_question(target_path)

            yield SentenceChunk(
                path=target_path,
                start_ms=safe_start,
                end_ms=safe_end,
                speaker=speaker,
                is_question=is_question,
            )
            chunk_index += 1


def _segment_turn(start_ms: int, end_ms: int) -> Iterable[tuple[int, int]]:
    duration = end_ms - start_ms
    if duration <= MAX_SENTENCE_DURATION_MS:
        yield start_ms, end_ms
        return

    segments = int(math.ceil(duration / MAX_SENTENCE_DURATION_MS))
    step = duration / segments
    for i in range(segments):
        seg_start = int(round(start_ms + i * step))
        seg_end = int(round(min(end_ms, seg_start + step)))
        yield seg_start, seg_end


def _clamp(*, s: int, e: int, limit: int) -> tuple[int, int]:
    start = max(0, min(s, limit))
    end = max(start + MIN_SENTENCE_DURATION_MS, min(e, limit))
    return start, end


def _extract_segment(source: Path, start_ms: int, end_ms: int, target: Path) -> None:
    start_s = max(0.0, start_ms / 1000.0)
    duration_s = max(0.05, (end_ms - start_ms) / 1000.0)

    process = (
        ffmpeg.input(str(source), ss=start_s, t=duration_s)
        .output(
            str(target),
            format="wav",
            ac=1,
            ar=TARGET_SAMPLE_RATE,
        )
        .overwrite_output()
        .global_args("-loglevel", "error")
    )

    try:
        process.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as exc:  # pragma: no cover - ffmpeg surfaces the message
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"Failed to extract audio segment: {stderr}") from exc


def _fallback_chunks(source: Path, output_dir: Path) -> List[SentenceChunk]:
    temp_dir = output_dir / "fallback"
    temp_dir.mkdir(parents=True, exist_ok=True)

    chunks: List[AudioChunk] = chunk_audio_file(
        source,
        temp_dir,
        min_chunk_ms=max(MIN_SENTENCE_DURATION_MS, 2_000),
        max_chunk_ms=MAX_SENTENCE_DURATION_MS,
        lookahead_ms=2_000,
        min_silence_len_ms=250,
        keep_silence_ms=120,
    )

    sentence_chunks: List[SentenceChunk] = []
    for index, chunk in enumerate(chunks):
        target = output_dir / f"sentence_fallback_{index:04d}.wav"
        shutil.move(str(chunk.path), target)
        sentence_chunks.append(
            SentenceChunk(
                path=target,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                speaker=None,
                is_question=_estimate_question(target),
            )
        )

    try:
        temp_dir.rmdir()
    except OSError:
        pass

    return sentence_chunks


def _estimate_question(path: Path) -> bool:
    try:
        import librosa  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return False

    try:
        audio, sr = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception:
        return False

    if audio.size < int(0.4 * TARGET_SAMPLE_RATE):
        return False

    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=TARGET_SAMPLE_RATE,
        )
    except Exception:
        return False

    if f0 is None:
        return False

    f0 = np.asarray(f0)
    mask = ~np.isnan(f0)
    if mask.sum() < QUESTION_MIN_FRAMES:
        return False

    voiced = f0[mask]
    half = max(1, voiced.size // 4)
    head = float(np.mean(voiced[:half]))
    tail = float(np.mean(voiced[-half:]))

    if math.isnan(head) or math.isnan(tail):
        return False

    return (tail - head) >= QUESTION_PITCH_DELTA


def _probe_duration_ms(source: Path) -> float:
    info = sf.info(str(source))
    if not info.duration:
        return 0.0
    return info.duration * 1000.0

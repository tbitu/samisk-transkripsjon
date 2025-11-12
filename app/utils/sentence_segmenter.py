from __future__ import annotations

"""Sentence-level speech segmentation utilities with speaker diarization support."""
from . import torchaudio_compat  # ensure torchaudio API shims are applied early

import logging
import math
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

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


def prepare_sentence_chunks(
    source: Path,
    output_dir: Path,
    *,
    progress: Optional[Callable[[float, Optional[Dict[str, object]]], None]] = None,
) -> List[SentenceChunk]:
    """Extract sentence-sized chunks with speaker labels when available."""

    output_dir.mkdir(parents=True, exist_ok=True)

    progress_adapter = _ProgressAdapter(progress) if progress else None

    if progress_adapter:
        progress_adapter.emit_start()

    # Diarization is mandatory: always attempt diarization and fail loudly
    try:
        diarization = _run_diarization(
            source,
            progress=progress_adapter.emit_diarization if progress_adapter else None,
        )
    except Exception as exc:  # pragma: no cover - external failures should surface
        logger.error("Speaker diarization failed: %s", exc)
        raise

    segments_to_extract = _collect_sentence_segments(source, diarization)

    if not segments_to_extract:
        logger.error("Diarization completed but produced no chunks for %s", source)
        raise RuntimeError("Diarization produced no chunks")

    chunks: List[Optional[SentenceChunk]] = [None] * len(segments_to_extract)
    total = len(segments_to_extract)

    if total == 0:
        return []

    worker_count = _suggest_sentence_workers(total)

    if progress_adapter:
        progress_adapter.emit_chunking_reset(total)

    if worker_count <= 1:
        for index, (start_ms, end_ms, speaker) in enumerate(segments_to_extract):
            chunk_index, chunk = _build_sentence_chunk(
                source,
                output_dir,
                index,
                start_ms,
                end_ms,
                speaker,
            )
            chunks[chunk_index] = chunk
            if progress_adapter:
                progress_adapter.emit_chunking_progress(index + 1)
    else:
        completed = 0
        counter_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _build_sentence_chunk,
                    source,
                    output_dir,
                    index,
                    start_ms,
                    end_ms,
                    speaker,
                )
                for index, (start_ms, end_ms, speaker) in enumerate(segments_to_extract)
            ]

            for future in as_completed(futures):
                chunk_index, chunk = future.result()
                chunks[chunk_index] = chunk

                if progress_adapter:
                    with counter_lock:
                        completed += 1
                        progress_adapter.emit_chunking_progress(completed)

    prepared_chunks = [chunk for chunk in chunks if chunk is not None]
    _offload_diarization_pipeline()
    return prepared_chunks


class _ProgressAdapter:
    """Normalize diarization + chunking progress into a single monotonic track."""

    _DIARIZATION_WEIGHT = 0.7
    _CHUNKING_WEIGHT = 1.0 - _DIARIZATION_WEIGHT

    def __init__(
        self,
        callback: Callable[[float, Optional[Dict[str, object]]], None],
    ) -> None:
        self._callback = callback
        self._lock = threading.Lock()
        self._diarization_fraction = 0.0
        self._chunking_fraction = 0.0
        self._total_chunks = 0
        self._last_combined = 0.0

    def emit_start(self) -> None:
        self._safe_emit(0.0, {"phase": "diarization", "stage": "start"})

    def emit_diarization(
        self,
        fraction: float,
        meta: Optional[Dict[str, object]] = None,
    ) -> None:
        bounded = self._bound(fraction)
        with self._lock:
            self._diarization_fraction = max(self._diarization_fraction, bounded)
            combined = self._combined_progress()
        payload = dict(meta or {})
        payload.setdefault("phase", "pyannote")
        self._safe_emit(combined, payload)

    def emit_chunking_reset(self, total_chunks: int) -> None:
        with self._lock:
            self._total_chunks = max(0, total_chunks)
            self._chunking_fraction = 0.0

    def emit_chunking_progress(self, completed: int) -> None:
        with self._lock:
            if self._total_chunks <= 0:
                fraction = 1.0
            else:
                fraction = self._bound(completed / self._total_chunks)
            self._chunking_fraction = max(self._chunking_fraction, fraction)
            combined = self._combined_progress()
        payload: Dict[str, object] = {
            "phase": "chunking",
            "completed": completed,
            "total": self._total_chunks,
        }
        self._safe_emit(combined, payload)

    def _combined_progress(self) -> float:
        combined = (
            self._diarization_fraction * self._DIARIZATION_WEIGHT
            + self._chunking_fraction * self._CHUNKING_WEIGHT
        )
        if combined < self._last_combined:
            combined = self._last_combined
        self._last_combined = min(1.0, combined)
        return self._last_combined

    def _safe_emit(
        self,
        fraction: float,
        meta: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self._callback:
            return
        bounded = self._bound(fraction)
        payload = meta or {}
        try:
            self._callback(bounded, payload)
        except Exception:  # pragma: no cover - defensive best effort
            logger.debug("Progress callback failed", exc_info=True)

    @staticmethod
    def _bound(value: float) -> float:
        if not isinstance(value, (float, int)):
            return 0.0
        return max(0.0, min(1.0, float(value)))


def _has_hf_token() -> bool:
    """Return True if an HF token is available via env or ./hf_token file."""
    token = (
        os.environ.get("PYANNOTE_AUTH_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    if token:
        return True
    token_file = Path(__file__).resolve().parents[2] / "hf_token"
    return token_file.exists() and token_file.read_text().strip() != ""


def _run_diarization(source: Path, progress: Optional[Callable[[float, Optional[Dict[str, object]]], None]] = None):
    pipeline = _load_diarization_pipeline()
    _ensure_diarization_device(pipeline)

    with _prepare_diarization_source(source) as normalized_source:
        if not progress:
            return pipeline(str(normalized_source))

        class _HookAdapter:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __call__(
                self,
                step_name: str,
                step_artifact,
                file: Optional[Mapping] = None,
                total: Optional[int] = None,
                completed: Optional[int] = None,
            ) -> None:
                try:
                    if total and completed is not None and total > 0:
                        fraction = float(completed) / float(total)
                    elif completed is not None:
                        fraction = 1.0 if completed > 0 else 0.0
                    else:
                        fraction = 0.0

                    meta: Dict[str, object] = {
                        "stage": step_name,
                        "completed": completed,
                        "total": total,
                    }
                    progress(fraction, meta)
                except Exception:
                    logger.debug("Hook adapter failed to emit progress", exc_info=True)

        return pipeline(str(normalized_source), hook=_HookAdapter())


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
        token_file = Path(__file__).resolve().parents[2] / "hf_token"
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
    _configure_diarization_inference(pipeline)
    return pipeline


@contextmanager
def _prepare_diarization_source(source: Path) -> Iterable[Path]:
    """Yield a path to 16 kHz mono PCM audio suitable for pyannote."""

    try:
        info = sf.info(str(source))
    except Exception:
        info = None

    if (
        info
        and info.samplerate == TARGET_SAMPLE_RATE
        and getattr(info, "channels", 0) == 1
    ):
        # Already in the desired shape.
        yield source
        return

    tmpdir = TemporaryDirectory(prefix="pyannote_norm_")
    try:
        target = Path(tmpdir.name) / "normalized.wav"
        process = (
            ffmpeg.input(str(source))
            .output(
                str(target),
                format="wav",
                ac=1,
                ar=TARGET_SAMPLE_RATE,
                sample_fmt="s16",
            )
            .overwrite_output()
            .global_args("-loglevel", "error")
        )

        try:
            process.run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
            raise RuntimeError(f"Failed to normalise audio for diarization: {stderr}") from exc

        yield target
    finally:
        tmpdir.cleanup()


def _configure_diarization_inference(pipeline) -> None:
    """Increase inference parallelism and batch duration when possible."""

    target_workers = max(1, (os.cpu_count() or 1) // 2)
    target_batch_size = 32
    target_duration = 15.0

    def _tune(obj) -> None:
        if obj is None:
            return
        try:
            if hasattr(obj, "num_workers"):
                current = getattr(obj, "num_workers", 0) or 0
                if current < target_workers:
                    obj.num_workers = target_workers
            if hasattr(obj, "batch_size"):
                current = getattr(obj, "batch_size", None)
                if current is None or current < target_batch_size:
                    obj.batch_size = target_batch_size
            if hasattr(obj, "duration"):
                current = getattr(obj, "duration", None)
                if current is None or current < target_duration:
                    obj.duration = target_duration
        except Exception:
            logger.debug("Failed tuning diarization inference %r", obj, exc_info=True)

    # Known attribute layouts in pyannote 3.x pipelines.
    candidates = [
        getattr(pipeline, name, None)
        for name in (
            "segmentation",
            "speech_turn_segmentation",
            "speaker_segmentation",
            "_segmentation_inference",
            "_speech_turn_segmentation_inference",
        )
    ]

    for item in candidates:
        if item is None:
            continue
        if hasattr(item, "inference"):
            _tune(getattr(item, "inference"))
        elif hasattr(item, "_inference"):
            _tune(getattr(item, "_inference"))
        else:
            _tune(item)

    inference_map = getattr(pipeline, "_inference", None)
    if isinstance(inference_map, dict):
        for inference in inference_map.values():
            _tune(inference)


def _ensure_diarization_device(pipeline) -> None:
    if not torch.cuda.is_available():
        return
    try:
        pipeline.to("cuda")
    except Exception:
        logger.debug("Failed moving diarization pipeline to CUDA", exc_info=True)


def _offload_diarization_pipeline() -> None:
    if not torch.cuda.is_available():
        return
    try:
        pipeline = _load_diarization_pipeline()
    except Exception:
        return

    try:
        pipeline.to("cpu")
        torch.cuda.empty_cache()
    except Exception:
        logger.debug("Failed offloading diarization pipeline to CPU", exc_info=True)


def _collect_sentence_segments(
    source: Path,
    diarization,
) -> List[tuple[int, int, Optional[str]]]:
    total_duration_ms = int(round(_probe_duration_ms(source)))
    segments: List[tuple[int, int, Optional[str]]] = []

    for start_s, end_s, speaker in _iter_diarization_turns(diarization):
        start_ms = int(round(max(0.0, start_s) * 1000))
        end_ms = int(round(min(end_s, total_duration_ms / 1000.0) * 1000))

        if end_ms - start_ms < MIN_SENTENCE_DURATION_MS:
            continue

        for sub_start, sub_end in _segment_turn(start_ms, end_ms):
            safe_start, safe_end = _clamp(s=sub_start, e=sub_end, limit=total_duration_ms)
            if safe_end - safe_start < MIN_SENTENCE_DURATION_MS:
                continue

            segments.append((safe_start, safe_end, speaker))

    return segments


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
        audio, sample_rate = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception:
        return False

    if audio.size < int(0.4 * TARGET_SAMPLE_RATE):
        return False

    try:
        f0, _, _ = librosa.pyin(
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

def _build_sentence_chunk(
    source: Path,
    output_dir: Path,
    index: int,
    start_ms: int,
    end_ms: int,
    speaker: Optional[str],
) -> Tuple[int, SentenceChunk]:
    target_path = output_dir / f"sentence_{index:04d}.wav"
    _extract_segment(source, start_ms, end_ms, target_path)
    is_question = _estimate_question(target_path)

    chunk = SentenceChunk(
        path=target_path,
        start_ms=start_ms,
        end_ms=end_ms,
        speaker=speaker,
        is_question=is_question,
    )

    return index, chunk


def _suggest_sentence_workers(total_segments: int) -> int:
    cpu_count = os.cpu_count() or 1
    if total_segments <= 1 or cpu_count <= 1:
        return 1

    # Keep some headroom for ffmpeg/librosa workloads but avoid oversubscription.
    upper_bound = 4 if torch.cuda.is_available() else 6
    headroom = max(1, cpu_count - 1)
    return max(1, min(upper_bound, total_segments, headroom))


def _probe_duration_ms(source: Path) -> float:
    info = sf.info(str(source))
    if not info.duration:
        return 0.0
    return info.duration * 1000.0

from __future__ import annotations

"""Sentence-level speech segmentation utilities with speaker diarization support."""
from . import torchaudio_compat  # ensure torchaudio API shims are applied early

import logging
import math
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import ffmpeg
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from .audio_chunker import AudioChunk, chunk_audio_file

DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
# Batch sizes for parallel processing during diarization (GPU only)
# Higher values use more GPU memory but process faster
# Reduce if you encounter OOM errors
DIARIZATION_SEGMENTATION_BATCH_SIZE = 32
DIARIZATION_EMBEDDING_BATCH_SIZE = 32
TARGET_SAMPLE_RATE = 16_000
MAX_SENTENCE_DURATION_MS = 20_000
MIN_SENTENCE_DURATION_MS = 1_200
SEGMENT_PADDING_MS = 250  # Padding to prevent mid-word cuts for Whisper timestamp prediction
QUESTION_PITCH_DELTA = 18.0
QUESTION_MIN_FRAMES = 5

_diarization_pipeline: Optional[Any] = None
_diarization_pipeline_loading = False
_diarization_lock = threading.Lock()

logger = logging.getLogger(__name__)


class ProgressAdapter:
    """Translate pyannote's progress hooks into application-specific progress."""

    def __init__(
        self,
        callback: Optional[Callable[[float, Optional[Dict[str, Any]]], None]] = None,
    ):
        self.callback = callback
        # Model loading is fast, so we don't allocate progress time to it.
        # Diarization inference is all that counts.
        self.model_load_fraction = 0.0
        self.inference_fraction = 1.0
        self.chunking_total = 0

    def emit_diarization(self, value: float, meta: Optional[Dict[str, Any]] = None) -> None:
        """Emit progress for the diarization stage."""
        if not self.callback:
            return

        # Model loading doesn't emit progress, so we only handle diarization inference here
        details = {"phase": "diarization", "diarization_fraction": value}
        details.update(meta or {})
        self.callback(value, details)

    def emit_chunking_reset(self, total: int) -> None:
        """Initialize chunking progress tracking."""
        self.chunking_total = total
        if self.callback:
            details = {"phase": "chunking", "total": total, "completed": 0}
            self.callback(0.0, details)

    def emit_chunking_progress(self, completed: int) -> None:
        """Emit progress for audio chunking."""
        if not self.callback or self.chunking_total == 0:
            return
        
        fraction = completed / self.chunking_total
        details = {"phase": "chunking", "total": self.chunking_total, "completed": completed}
        self.callback(fraction, details)


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
    audio_file: Path,
    output_dir: Path,
    progress: Optional[Callable[[float, Optional[Dict[str, Any]]], None]] = None,
) -> List[SentenceChunk]:
    """
    Run diarization and VAD to prepare sentence-level audio chunks.
    
    Args:
        audio_file: Path to the input audio file.
        output_dir: Directory to store temporary chunk files.
        progress: Optional callback for progress updates.
        
    Returns:
        A list of SentenceChunk objects.
    """
    if not progress:
        progress_adapter = None
    else:
        progress_adapter = ProgressAdapter(progress)

    # Diarization is mandatory: always attempt diarization and fail loudly
    try:
        diarization = _run_diarization(
            audio_file,
            progress=progress_adapter.emit_diarization if progress_adapter else None,
        )
    except Exception as exc:
        logger.error("Speaker diarization failed: %s", exc)
        raise RuntimeError("Diarization failed") from exc

    # Log GPU memory before diarization cleanup
    if torch.cuda.is_available():
        try:
            allocated_before = torch.cuda.memory_allocated(0) / 1024**3
            reserved_before = torch.cuda.memory_reserved(0) / 1024**3
            free_before = torch.cuda.mem_get_info(0)[0] / 1024**3
            logger.info(f"GPU before diarization cleanup - Allocated: {allocated_before:.2f}GB, Reserved: {reserved_before:.2f}GB, Free: {free_before:.2f}GB")
        except Exception:
            pass

    # CRITICAL: Unload diarization from GPU before loading VAD
    _offload_diarization_pipeline()

    # Log GPU memory after diarization cleanup
    if torch.cuda.is_available():
        try:
            allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            reserved_after = torch.cuda.memory_reserved(0) / 1024**3
            free_after = torch.cuda.mem_get_info(0)[0] / 1024**3
            freed = free_after - free_before
            logger.info(f"GPU after diarization cleanup - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB, Free: {free_after:.2f}GB (freed {freed:.2f}GB)")
        except Exception:
            pass

    logger.info("Diarization complete, GPU memory freed for VAD")

    segments_to_extract = _collect_sentence_segments(audio_file, diarization)

    if not segments_to_extract:
        logger.error("Diarization completed but produced no chunks for %s", audio_file)
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
                audio_file,
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
                    audio_file,
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
    
    # Log GPU memory before VAD cleanup
    if torch.cuda.is_available():
        try:
            allocated_before = torch.cuda.memory_allocated(0) / 1024**3
            reserved_before = torch.cuda.memory_reserved(0) / 1024**3
            free_before = torch.cuda.mem_get_info(0)[0] / 1024**3
            logger.info(f"GPU before VAD cleanup - Allocated: {allocated_before:.2f}GB, Reserved: {reserved_before:.2f}GB, Free: {free_before:.2f}GB")
        except Exception:
            pass
    
    # CRITICAL: Clear VAD from GPU now that sentence segmentation is complete
    # This frees GPU memory for Stanza (punctuation)
    from .vad import clear_vad_from_memory
    clear_vad_from_memory()
    
    # Log GPU memory after VAD cleanup
    if torch.cuda.is_available():
        try:
            allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            reserved_after = torch.cuda.memory_reserved(0) / 1024**3
            free_after = torch.cuda.mem_get_info(0)[0] / 1024**3
            freed = free_after - free_before
            logger.info(f"GPU after VAD cleanup - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB, Free: {free_after:.2f}GB (freed {freed:.2f}GB)")
        except Exception:
            pass
    
    logger.info("VAD processing complete, GPU memory freed for punctuation")
    
    return prepared_chunks


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


def _run_diarization(
    audio_file: Path,
    progress: Optional[Callable[[float, Optional[Dict[str, Any]]], None]] = None,
) -> Any:
    """Run speaker diarization on the given audio file."""
    pipeline = _load_diarization_pipeline(progress=progress)

    logger.info("Running speaker diarization on %s", audio_file)
    
    # Wrap the pipeline's progress hook
    if progress:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        
        # Create a custom hook to track progress (without console output)
        class TrackingHook(ProgressHook):
            def __init__(self, progress_callback):
                super().__init__()
                self.progress_callback = progress_callback
                
            def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
                # ONLY report progress for the embeddings step - that's the actual diarization work
                # Ignore segmentation and speaker_counting as they're preprocessing
                if step_name == 'embeddings' and completed is not None and total is not None and total > 0:
                    # Report embeddings progress directly as overall diarization progress
                    progress = completed / total
                    self.progress_callback(progress, {"phase": "diarization", "step": "embeddings"})
        
        with TrackingHook(progress) as hook:
            diarization = pipeline(str(audio_file), hook=hook)
            progress(1.0, {"phase": "diarization"})
    else:
        diarization = pipeline(str(audio_file))

    logger.info("Diarization complete")
    return diarization


def _load_diarization_pipeline(
    progress: Optional[Callable[[float, Optional[Dict[str, Any]]], None]] = None,
) -> Any:
    """Load the pyannote diarization pipeline with progress."""
    global _diarization_pipeline, _diarization_pipeline_loading
    with _diarization_lock:
        if _diarization_pipeline:
            # Model is already loaded, return immediately without progress updates
            # Model loading should not count towards progress at all
            return _diarization_pipeline
        if _diarization_pipeline_loading:
            # Another thread is loading, wait and check again
            while _diarization_pipeline_loading:
                time.sleep(0.5)
            if _diarization_pipeline:
                return _diarization_pipeline

        _diarization_pipeline_loading = True

    try:
        logger.info("Loading diarization pipeline: %s", DIARIZATION_MODEL_ID)

        hf_token = _get_hf_token()
        if not hf_token:
            raise RuntimeError("Hugging Face token not found for diarization")

        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL_ID,
            token=hf_token,
        )

        if torch.cuda.is_available():
            logger.info("Moving diarization pipeline to GPU")
            pipeline = pipeline.to(torch.device("cuda"))
            
            # Enable batch processing for faster diarization on GPU
            # Segmentation processes audio chunks in parallel
            # Default is 1 which only uses one core - increase for better GPU utilization
            pipeline.segmentation_batch_size = DIARIZATION_SEGMENTATION_BATCH_SIZE
            pipeline.embedding_batch_size = DIARIZATION_EMBEDDING_BATCH_SIZE
            logger.info(
                f"Configured diarization with segmentation_batch_size={DIARIZATION_SEGMENTATION_BATCH_SIZE}, "
                f"embedding_batch_size={DIARIZATION_EMBEDDING_BATCH_SIZE} for parallel processing"
            )

        with _diarization_lock:
            _diarization_pipeline = pipeline
            return _diarization_pipeline
    finally:
        with _diarization_lock:
            _diarization_pipeline_loading = False


def _offload_diarization_pipeline() -> None:
    """Move the diarization pipeline to CPU and clear GPU cache."""
    global _diarization_pipeline
    with _diarization_lock:
        if not _diarization_pipeline:
            return

        logger.info("Offloading diarization pipeline from GPU to CPU")
        try:
            _diarization_pipeline = _diarization_pipeline.to(torch.device("cpu"))
        except Exception as exc:
            logger.warning("Could not move diarization pipeline to CPU: %s", exc)
        
        _diarization_pipeline = None

        # Force garbage collection and clear cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _get_hf_token() -> Optional[str]:
    """Retrieve the Hugging Face token from the environment or a file."""
    import os
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    
    # Check for a file named 'hf_token' in the root directory
    try:
        from .. import main
        root_dir = Path(main.__file__).parent.parent
        token_file = root_dir / "hf_token"
        if token_file.exists():
            return token_file.read_text().strip()
    except Exception:
        pass
        
    return None


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

        # Some pyannote versions return {'annotated': Annotation_obj}
        if "annotated" in diarization:
            for turn, _, speaker in diarization["annotated"].itertracks(yield_label=True):
                yield float(turn.start), float(turn.end), speaker
            return

    raise TypeError(f"Unsupported diarization type: {type(diarization)}")


def _collect_sentence_segments(
    source: Path,
    diarization,
) -> List[tuple[int, int, Optional[str]]]:
    """
    Collect sentence segments from diarization, refined with Silero VAD.
    
    This function:
    1. Loads the full audio once (optimization!)
    2. Extracts speaker turns from diarization
    3. Runs Silero VAD on each turn to find actual speech
    4. Merges short adjacent segments within the same speaker turn
    5. Enforces MIN/MAX_SENTENCE_DURATION_MS constraints
    """
    from .vad import detect_speech_segments, merge_adjacent_segments, SpeechSegment
    
    total_duration_ms = int(round(_probe_duration_ms(source)))
    segments: List[tuple[int, int, Optional[str]]] = []

    # OPTIMIZATION: Load audio once instead of for every turn
    try:
        audio_data, original_sample_rate = sf.read(str(source))
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # CRITICAL: Resample to 16kHz immediately for VAD compatibility
        # Silero VAD only supports 8kHz and 16kHz (or multiples of 16kHz)
        if original_sample_rate != TARGET_SAMPLE_RATE:
            logger.info("Resampling audio from %dHz to %dHz for VAD", original_sample_rate, TARGET_SAMPLE_RATE)
            # Convert to torch tensor for resampling
            audio_tensor = torch.from_numpy(audio_data).float()
            # Resample using torchaudio
            audio_tensor = AF.resample(audio_tensor, original_sample_rate, TARGET_SAMPLE_RATE)
            # Convert back to numpy
            audio_data = audio_tensor.numpy()
        
        sample_rate = TARGET_SAMPLE_RATE
        logger.info("Loaded audio for VAD refinement: %d samples at %dHz", len(audio_data), sample_rate)
    except Exception as exc:
        logger.warning("Failed to pre-load audio for VAD: %s. Falling back to simple segmentation.", exc)
        audio_data = None
        sample_rate = TARGET_SAMPLE_RATE

    for start_s, end_s, speaker in _iter_diarization_turns(diarization):
        start_ms = int(round(max(0.0, start_s) * 1000))
        end_ms = int(round(min(end_s, total_duration_ms / 1000.0) * 1000))

        if end_ms - start_ms < MIN_SENTENCE_DURATION_MS:
            continue

        # Refine this speaker turn with VAD
        if audio_data is not None:
            try:
                vad_segments = _refine_turn_with_vad_cached(
                    audio_data,
                    sample_rate,
                    start_ms, 
                    end_ms,
                    speaker,
                    total_duration_ms,
                )
                segments.extend(vad_segments)
                continue
            except Exception as exc:
                logger.warning("VAD refinement failed for turn at %d-%d: %s", start_ms, end_ms, exc)
        
        # Fallback to simple segmentation
        for sub_start, sub_end in _segment_turn(start_ms, end_ms):
            safe_start, safe_end = _clamp(s=sub_start, e=sub_end, limit=total_duration_ms)
            if safe_end - safe_start >= MIN_SENTENCE_DURATION_MS:
                segments.append((safe_start, safe_end, speaker))

    return segments


def _refine_turn_with_vad_cached(
    audio_data: np.ndarray,
    sample_rate: int,
    turn_start_ms: int,
    turn_end_ms: int,
    speaker: Optional[str],
    total_duration_ms: int,
) -> List[tuple[int, int, Optional[str]]]:
    """
    Use Silero VAD to refine a speaker turn into precise speech segments (optimized).
    
    This version uses pre-loaded audio data for better performance.
    
    Returns list of (start_ms, end_ms, speaker) tuples.
    """
    from .vad import refine_segment_with_vad_cached, SpeechSegment, merge_adjacent_segments
    
    # Get VAD segments relative to the turn
    relative_segments = refine_segment_with_vad_cached(
        audio_data,
        sample_rate,
        turn_start_ms,
        turn_end_ms,
        min_speech_duration_ms=MIN_SENTENCE_DURATION_MS // 2,  # Be more lenient initially
        min_silence_duration_ms=100,
    )
    
    if not relative_segments:
        # No speech detected, return empty
        return []
    
    # Convert relative offsets to absolute timestamps
    absolute_segments = []
    for rel_start, rel_end in relative_segments:
        abs_start = turn_start_ms + rel_start
        abs_end = turn_start_ms + rel_end
        absolute_segments.append(SpeechSegment(start_ms=abs_start, end_ms=abs_end))
    
    # Merge adjacent segments that are close together
    merged_segments = merge_adjacent_segments(
        absolute_segments,
        max_gap_ms=300,  # Merge if gap is less than 300ms
        max_duration_ms=MAX_SENTENCE_DURATION_MS,
    )
    
    # Convert to final format, enforcing duration constraints
    result = []
    for seg in merged_segments:
        # Clamp to valid range
        start_ms = max(0, min(seg.start_ms, total_duration_ms))
        end_ms = max(start_ms, min(seg.end_ms, total_duration_ms))
        
        duration = end_ms - start_ms
        
        # Skip segments that are too short
        if duration < MIN_SENTENCE_DURATION_MS:
            continue
        
        # Split segments that are too long
        if duration > MAX_SENTENCE_DURATION_MS:
            for sub_start, sub_end in _segment_turn(start_ms, end_ms):
                safe_start, safe_end = _clamp(s=sub_start, e=sub_end, limit=total_duration_ms)
                if safe_end - safe_start >= MIN_SENTENCE_DURATION_MS:
                    result.append((safe_start, safe_end, speaker))
        else:
            result.append((start_ms, end_ms, speaker))
    
    return result


def _refine_turn_with_vad(
    source: Path,
    turn_start_ms: int,
    turn_end_ms: int,
    speaker: Optional[str],
    total_duration_ms: int,
) -> List[tuple[int, int, Optional[str]]]:
    """
    Use Silero VAD to refine a speaker turn into precise speech segments.
    
    DEPRECATED: Use _refine_turn_with_vad_cached for better performance.
    
    Returns list of (start_ms, end_ms, speaker) tuples.
    """
    from .vad import refine_segment_with_vad, SpeechSegment, merge_adjacent_segments
    
    # Get VAD segments relative to the turn
    relative_segments = refine_segment_with_vad(
        source,
        turn_start_ms,
        turn_end_ms,
        min_speech_duration_ms=MIN_SENTENCE_DURATION_MS // 2,  # Be more lenient initially
        min_silence_duration_ms=100,
        sample_rate=TARGET_SAMPLE_RATE,
    )
    
    if not relative_segments:
        # No speech detected, return empty
        return []
    
    # Convert relative offsets to absolute timestamps
    absolute_segments = []
    for rel_start, rel_end in relative_segments:
        abs_start = turn_start_ms + rel_start
        abs_end = turn_start_ms + rel_end
        absolute_segments.append(SpeechSegment(start_ms=abs_start, end_ms=abs_end))
    
    # Merge adjacent segments that are close together
    merged_segments = merge_adjacent_segments(
        absolute_segments,
        max_gap_ms=300,  # Merge if gap is less than 300ms
        max_duration_ms=MAX_SENTENCE_DURATION_MS,
    )
    
    # Convert to final format, enforcing duration constraints
    result = []
    for seg in merged_segments:
        # Clamp to valid range
        start_ms = max(0, min(seg.start_ms, total_duration_ms))
        end_ms = max(start_ms, min(seg.end_ms, total_duration_ms))
        
        duration = end_ms - start_ms
        
        # Skip segments that are too short
        if duration < MIN_SENTENCE_DURATION_MS:
            continue
        
        # Split segments that are too long
        if duration > MAX_SENTENCE_DURATION_MS:
            for sub_start, sub_end in _segment_turn(start_ms, end_ms):
                safe_start, safe_end = _clamp(s=sub_start, e=sub_end, limit=total_duration_ms)
                if safe_end - safe_start >= MIN_SENTENCE_DURATION_MS:
                    result.append((safe_start, safe_end, speaker))
        else:
            result.append((start_ms, end_ms, speaker))
    
    return result


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
    # Add padding to prevent mid-word cuts that cause Whisper timestamp failures
    padded_start_ms = max(0, start_ms - SEGMENT_PADDING_MS)
    padded_end_ms = end_ms + SEGMENT_PADDING_MS  # ffmpeg handles if this exceeds file length
    
    start_s = padded_start_ms / 1000.0
    duration_s = max(0.05, (padded_end_ms - padded_start_ms) / 1000.0)
    
    logger.debug(
        "Extracting segment %d-%dms (padded to %d-%dms, +%dms padding)",
        start_ms, end_ms, padded_start_ms, padded_end_ms, SEGMENT_PADDING_MS
    )

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

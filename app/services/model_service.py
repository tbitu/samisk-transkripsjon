"""Model loading utilities for the Whisper large Northern Sámi model."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Tuple

from ..utils.sentence_segmenter import prepare_sentence_chunks

logger = logging.getLogger(__name__)

MODEL_ID = "NbAiLab/whisper-large-sme"


def _resolve_device() -> int | str:
    """Return CUDA device index when available, otherwise CPU."""
    if torch.cuda.is_available():
        logger.info("CUDA available; using GPU for inference")
        return 0
    logger.warning("CUDA not available; falling back to CPU. Expect slower inference.")
    return "cpu"


def _resolve_dtype(device: int | str) -> torch.dtype:
    """Use float16 on GPU to conserve memory."""
    if isinstance(device, int):
        return torch.float16
    return torch.float32


@lru_cache(maxsize=1)
def get_asr_pipeline() -> Any:
    """Load and cache the speech recognition pipeline."""
    import gc
    
    device = _resolve_device()
    torch_dtype = _resolve_dtype(device)

    # Ensure clean GPU state before loading the large Whisper model
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Log available memory
        try:
            free_memory = torch.cuda.mem_get_info(0)[0] / 1024**3
            total_memory = torch.cuda.mem_get_info(0)[1] / 1024**3
            logger.info(f"Loading Whisper model - GPU free: {free_memory:.2f}GB / {total_memory:.2f}GB")
        except Exception:
            pass

    logger.info("Loading model %s", MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    if isinstance(device, int):
        model = model.to(device)
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        try:
            if torch.backends.cuda.matmul is not None:
                torch.backends.cuda.matmul.allow_tf32 = True
            if torch.backends.cudnn is not None:
                torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Clear deprecated forced_decoder_ids from generation config
    # These are deprecated in favor of task and language parameters
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        if hasattr(model.generation_config, 'forced_decoder_ids'):
            model.generation_config.forced_decoder_ids = None
            logger.debug("Cleared deprecated forced_decoder_ids from generation config")

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=444,  # 448 max_target_positions - 4 for decoder input tokens and safety margin
        chunk_length_s=30,
        batch_size=_default_batch_size(device),
        torch_dtype=torch_dtype,
        device=device if isinstance(device, int) else -1,
    )

    logger.info("Pipeline initialised successfully")
    return pipe


ProgressCallback = Callable[[str, float, Optional[Dict[str, Any]]], None]


def _cleanup_preprocessing_models() -> None:
    """Clear all preprocessing models from GPU memory before transcription.
    
    IMPORTANT: This clears models in reverse order of usage to ensure clean GPU memory:
    1. Diarization (pyannote) - used first, cleared first
    2. VAD (Silero) - used second, cleared second  
    3. Stanza (punctuation) - used last, cleared last
    
    Each model is moved to CPU, cache cleared, and GPU synchronized before moving to next.
    """
    import gc
    import time
    
    logger.info("Clearing preprocessing models from GPU memory in sequence...")
    
    # Step 1: Clear diarization pipeline (pyannote)
    # This was already done in prepare_sentence_chunks, but ensure it's cleared
    try:
        from ..utils.sentence_segmenter import _offload_diarization_pipeline, _load_diarization_pipeline
        _offload_diarization_pipeline()
        logger.info("✓ Diarization pipeline cleared")
        
        # Synchronize after each model and add grace period
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.3)  # Grace period for GPU driver cleanup
    except Exception as exc:
        logger.debug("Could not offload diarization pipeline: %s", exc)
    
    # Step 2: Clear VAD model (Silero)
    # This was already done in prepare_sentence_chunks, but ensure it's cleared
    try:
        from ..utils.vad import clear_vad_from_memory
        clear_vad_from_memory()
        logger.info("✓ VAD model cleared")
        
        # Synchronize after each model and add grace period
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.3)  # Grace period for GPU driver cleanup
    except Exception as exc:
        logger.debug("Could not clear VAD from memory: %s", exc)
    
    # Step 3: Clear Stanza NLP models (used during punctuation)
    # This is critical as Stanza may have been loaded during _format_sentence calls
    try:
        from ..utils.punctuation import clear_stanza_from_memory
        clear_stanza_from_memory()
        logger.info("✓ Stanza models cleared")
        
        # Final synchronization and grace period
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.3)  # Grace period for GPU driver cleanup
    except Exception as exc:
        logger.debug("Could not clear Stanza from memory: %s", exc)
    
    # Final cleanup: Force garbage collection and clear CUDA cache
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Final grace period before Whisper loads
        time.sleep(0.5)
        
        # Log memory status
        try:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            free_memory = torch.cuda.mem_get_info(0)[0] / 1024**3
            total_memory = torch.cuda.mem_get_info(0)[1] / 1024**3
            logger.info(
                f"GPU memory after cleanup - Allocated: {allocated:.2f}GB, "
                f"Reserved: {reserved:.2f}GB, Free: {free_memory:.2f}GB / {total_memory:.2f}GB"
            )
        except Exception:
            pass
        
        logger.info("All preprocessing models cleared, GPU ready for Whisper transcription")


def _reconcile_chunks_with_whisper_timestamps(
    chunk: Any,
    result: Dict[str, Any],
    formatted_text: str,
) -> List[Dict[str, Any]]:
    """
    Split a chunk into multiple sub-chunks if Whisper detected multiple sentences.
    
    Uses word-level timestamps from Whisper to determine sentence boundaries.
    
    Args:
        chunk: The original SentenceChunk
        result: Whisper result with potential word timestamps
        formatted_text: The formatted text with punctuation
        
    Returns:
        List of sub-chunks with refined boundaries
    """
    from ..utils.punctuation import split_into_sentences
    
    # Try to split into sentences
    sentences = split_into_sentences(formatted_text, lang="sme")
    
    # If only one sentence or no word timestamps, return as-is
    chunks_data = result.get("chunks", [])
    if len(sentences) <= 1 or not chunks_data:
        return [{
            "text": formatted_text,
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "speaker": chunk.speaker,
            "is_question": chunk.is_question,
        }]
    
    # Build word timing map
    words_with_times = []
    for word_chunk in chunks_data:
        if isinstance(word_chunk, dict) and "timestamp" in word_chunk:
            word_text = word_chunk.get("text", "").strip()
            timestamp = word_chunk.get("timestamp", [0.0, 0.0])
            if word_text and len(timestamp) == 2:
                words_with_times.append({
                    "text": word_text,
                    "start": timestamp[0],
                    "end": timestamp[1],
                })
    
    if not words_with_times:
        return [{
            "text": formatted_text,
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "speaker": chunk.speaker,
            "is_question": chunk.is_question,
        }]
    
    # Try to align sentences with word timestamps
    result_chunks = []
    word_idx = 0
    
    for sent in sentences:
        sent_words = sent.lower().split()
        if not sent_words:
            continue
        
        # Find matching words in the timestamp list
        sent_start_time = None
        sent_end_time = None
        matches = 0
        
        while word_idx < len(words_with_times) and matches < len(sent_words):
            word_data = words_with_times[word_idx]
            word_text_clean = re.sub(r'[^\w\s]', '', word_data["text"].lower())
            
            if sent_start_time is None:
                sent_start_time = word_data["start"]
            
            sent_end_time = word_data["end"]
            matches += 1
            word_idx += 1
        
        if sent_start_time is not None and sent_end_time is not None:
            # Convert to absolute milliseconds
            abs_start_ms = chunk.start_ms + int(sent_start_time * 1000)
            abs_end_ms = chunk.start_ms + int(sent_end_time * 1000)
            
            result_chunks.append({
                "text": sent,
                "start_ms": abs_start_ms,
                "end_ms": abs_end_ms,
                "speaker": chunk.speaker,
                "is_question": sent.rstrip().endswith("?"),
            })
    
    # If we couldn't split properly, return original
    if not result_chunks:
        return [{
            "text": formatted_text,
            "start_ms": chunk.start_ms,
            "end_ms": chunk.end_ms,
            "speaker": chunk.speaker,
            "is_question": chunk.is_question,
        }]
    
    return result_chunks


def transcribe(
    file_path: str | Path,
    *,
    temperature: float = 0.0,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run transcription on the given audio file."""

    combined_blocks: List[Dict[str, Any]] = []
    combined_segments: List[Dict[str, Any]] = []

    def _emit(stage: str, fraction: float, info: Optional[Dict[str, Any]] = None) -> None:
        if not progress:
            return
        try:
            progress(stage, fraction, info or {})
        except Exception:  # pragma: no cover - progress callbacks are best effort
            logger.debug("Progress callback failed", exc_info=True)

    with TemporaryDirectory(prefix="whisper_sentences_") as tmpdir:
        diarization_progress: Optional[Callable[[float, Optional[Dict[str, Any]]], None]] = None

        if progress:
            _emit("diarization", 0.0, {"stage": "start"})

            def diarization_progress(value: float, meta: Optional[Dict[str, Any]] = None) -> None:
                _emit("diarization", value, meta)
        else:
            diarization_progress = None

        sentence_chunks = prepare_sentence_chunks(
            Path(file_path),
            Path(tmpdir),
            progress=diarization_progress,
        )

        # Log GPU memory before cleanup
        if torch.cuda.is_available():
            try:
                allocated_before = torch.cuda.memory_allocated(0) / 1024**3
                reserved_before = torch.cuda.memory_reserved(0) / 1024**3
                free_before = torch.cuda.mem_get_info(0)[0] / 1024**3
                logger.info(f"GPU before preprocessing cleanup - Allocated: {allocated_before:.2f}GB, Reserved: {reserved_before:.2f}GB, Free: {free_before:.2f}GB")
            except Exception:
                pass

        # CRITICAL: Free GPU memory from diarization, VAD, and Stanza before loading Whisper
        _cleanup_preprocessing_models()

        # Log GPU memory after cleanup, before Whisper loads
        if torch.cuda.is_available():
            try:
                allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                reserved_after = torch.cuda.memory_reserved(0) / 1024**3
                free_after = torch.cuda.mem_get_info(0)[0] / 1024**3
                freed = free_after - free_before
                logger.info(f"GPU after preprocessing cleanup - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB, Free: {free_after:.2f}GB (freed {freed:.2f}GB)")
            except Exception:
                pass

        # NOW load the Whisper pipeline after cleanup
        logger.info("Loading Whisper model for transcription...")
        pipeline_ = get_asr_pipeline()
        
        # Log GPU memory after Whisper loads
        if torch.cuda.is_available():
            try:
                allocated_whisper = torch.cuda.memory_allocated(0) / 1024**3
                reserved_whisper = torch.cuda.memory_reserved(0) / 1024**3
                free_whisper = torch.cuda.mem_get_info(0)[0] / 1024**3
                logger.info(f"GPU after Whisper loaded - Allocated: {allocated_whisper:.2f}GB, Reserved: {reserved_whisper:.2f}GB, Free: {free_whisper:.2f}GB")
            except Exception:
                pass

        total_chunks = len(sentence_chunks)

        if total_chunks == 0:
            logger.warning("No voice activity detected in %s", file_path)
            _emit("diarization", 1.0, {"chunks": 0})
            _emit("transcription", 1.0, {"chunks": 0})
            return {"text": "", "chunks": []}

        logger.info("Processing %d sentence chunks", total_chunks)
        _emit("diarization", 1.0, {"chunks": total_chunks})
        _emit("transcription", 0.0, {"total": total_chunks})

        inputs = [str(chunk.path) for chunk in sentence_chunks]
        pipeline_device = getattr(pipeline_, "device", None)
        batch_size = min(len(inputs), _default_batch_size(pipeline_device))
        logger.info(f"Using batch_size={batch_size} for transcription (processing {batch_size} chunk(s) at a time)")

        def _transcription_progress(processed: int, total: int) -> None:
            if total <= 0:
                fraction = 1.0
            else:
                fraction = processed / total
            _emit("transcription", fraction, {"processed": processed, "total": total})

        results = _transcribe_in_batches(
            pipeline_,
            inputs,
            batch_size=batch_size,
            temperature=temperature,
            progress=_transcription_progress if progress else None,
        )

        _emit("transcription", 1.0, {"processed": total_chunks, "total": total_chunks})

        # CRITICAL: Clear Whisper from GPU before loading Stanza for punctuation
        # This prevents OOM errors when formatting sentences
        logger.info("Transcription complete, clearing Whisper from GPU for punctuation...")
        
        # Log GPU memory before Whisper cleanup
        if torch.cuda.is_available():
            try:
                allocated_before = torch.cuda.memory_allocated(0) / 1024**3
                reserved_before = torch.cuda.memory_reserved(0) / 1024**3
                free_before = torch.cuda.mem_get_info(0)[0] / 1024**3
                logger.info(f"GPU before Whisper cleanup - Allocated: {allocated_before:.2f}GB, Reserved: {reserved_before:.2f}GB, Free: {free_before:.2f}GB")
            except Exception:
                pass
        
        del pipeline_
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import time
            time.sleep(0.3)  # Grace period for GPU driver
            
            # Log GPU memory after Whisper cleanup
            try:
                allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                reserved_after = torch.cuda.memory_reserved(0) / 1024**3
                free_after = torch.cuda.mem_get_info(0)[0] / 1024**3
                freed = free_after - free_before
                logger.info(f"GPU after Whisper cleanup - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB, Free: {free_after:.2f}GB (freed {freed:.2f}GB)")
            except Exception:
                pass
        
        logger.info("✓ Whisper cleared, GPU freed for Stanza punctuation")

        # Now process results with Stanza (which will be loaded on-demand)
        _emit("punctuation", 0.0, {"total": total_chunks})
        
        for idx, (chunk, result) in enumerate(zip(sentence_chunks, results), 1):
            raw_text = (result or {}).get("text", "")
            formatted_text, is_question = _format_sentence(raw_text, chunk.is_question)
            if not formatted_text:
                continue
            
            # Report punctuation progress
            if idx % 5 == 0 or idx == total_chunks:  # Report every 5 chunks or at the end
                _emit("punctuation", idx / total_chunks, {"processed": idx, "total": total_chunks})

            # Reconcile chunks using Whisper word timestamps
            # This may split one chunk into multiple sentences
            sub_chunks = _reconcile_chunks_with_whisper_timestamps(
                chunk, result, formatted_text
            )
            
            for sub_chunk in sub_chunks:
                sub_text = sub_chunk["text"]
                sub_speaker = (sub_chunk["speaker"] or "unknown").strip() or "unknown"
                sub_is_question = sub_chunk["is_question"]
                
                # Build combined blocks (for the main text output)
                if combined_blocks and combined_blocks[-1]["speaker"] == sub_speaker:
                    combined_blocks[-1]["texts"].append(sub_text)
                else:
                    combined_blocks.append({"speaker": sub_speaker, "texts": [sub_text]})
                
                # Extract word-level timestamps if available
                word_timestamps = []
                chunks_data = (result or {}).get("chunks", [])
                if chunks_data:
                    for word_chunk in chunks_data:
                        if isinstance(word_chunk, dict):
                            word_timestamps.append({
                                "text": word_chunk.get("text", ""),
                                "timestamp": word_chunk.get("timestamp", [0.0, 0.0]),
                            })
                
                segment_info = {
                    "timestamp": (
                        sub_chunk["start_ms"] / 1000.0,
                        sub_chunk["end_ms"] / 1000.0,
                    ),
                    "text": sub_text,
                    "speaker": sub_speaker,
                    "question": sub_is_question,
                }
                
                # Add word timestamps if available
                if word_timestamps:
                    segment_info["words"] = word_timestamps
                
                combined_segments.append(segment_info)

    combined_text = "\n\n".join(
        f"{block['speaker']}: {' '.join(block['texts'])}"
        for block in combined_blocks
    )

    response: Dict[str, Any] = {"text": combined_text}

    if combined_segments:
        response["chunks"] = combined_segments

    return response


def _default_batch_size(device: Any) -> int:
    # Use batch size of 1 to be conservative with GPU memory.
    return 1


def _transcribe_in_batches(
    pipeline_: Any,
    inputs: List[str],
    *,
    batch_size: int,
    temperature: float,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    if not inputs:
        if progress:
            progress(0, 0)
        return []

    total = len(inputs)
    processed = 0
    outputs: List[Dict[str, Any]] = []

    for start in range(0, total, batch_size):
        batch_inputs = inputs[start : start + batch_size]
        logger.info(f"Transcribing batch {start//batch_size + 1}/{(total + batch_size - 1)//batch_size}: processing {len(batch_inputs)} chunk(s) (chunks {start+1}-{start+len(batch_inputs)} of {total})")
        try:
            batch_outputs = pipeline_(
                batch_inputs,
                batch_size=len(batch_inputs),
                return_timestamps=True,  # Enable timestamps (will return word-level)
                generate_kwargs={
                    "task": "transcribe",
                    "temperature": temperature,
                },
            )
        except (ValueError, RuntimeError) as exc:
            message = str(exc)
            if "Whisper did not predict an ending timestamp" in message or "timestamp" in message.lower():
                logger.warning(
                    "Timestamp prediction failed for batch starting at chunk %d/%d: %s. "
                    "Falling back to text-only transcription.",
                    start + 1, total, message
                )
                # Fallback to no timestamps for problematic chunks
                for item in batch_inputs:
                    single_output = pipeline_(
                        [item],
                        batch_size=1,
                        return_timestamps=False,
                        generate_kwargs={
                            "task": "transcribe",
                            "temperature": temperature,
                        },
                    )
                    if isinstance(single_output, dict):
                        batch_single = [single_output]
                    else:
                        batch_single = list(single_output)

                    outputs.extend(batch_single)
                    processed += len(batch_single)
                    if progress:
                        progress(processed, total)
                continue
            raise

        if isinstance(batch_outputs, dict):
            batch_list = [batch_outputs]
        else:
            batch_list = list(batch_outputs)

        outputs.extend(batch_list)
        processed += len(batch_list)
        if progress:
            progress(processed, total)

    if progress:
        progress(total, total)

    return outputs


QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "who",
    "where",
    "when",
    "which",
    "whose",
    "kan",
    "hvor",
    "kor",
    "man",
    "mii",
    "gosa",
    "goas",
    "galle",
    "ma",
    "mon",
    "lea",
    "leago",
    "galga",
    "go",
}


def _format_sentence(raw_text: str, audio_question: bool) -> tuple[str, bool]:
    """
    Format a transcribed sentence with proper punctuation using Stanza NLP.
    
    Args:
        raw_text: Raw transcription text
        audio_question: Whether audio analysis (pitch) suggests a question
        
    Returns:
        Tuple of (formatted_text, is_question)
    """
    from ..utils.punctuation import restore_punctuation
    
    text = raw_text.strip()
    if not text:
        return "", False

    # Normalize spacing first
    text = _normalise_spacing(text)
    
    # Use Stanza to restore proper punctuation
    try:
        punctuated = restore_punctuation(
            text,
            lang="sme",  # Northern Sámi (falls back to Norwegian in implementation)
            audio_is_question=audio_question,
        )
        return punctuated.text, punctuated.is_question
        
    except Exception as exc:
        logger.debug("Punctuation restoration failed, using fallback: %s", exc)
        # Fallback to simple formatting
        return _format_sentence_fallback(text, audio_question)


def _format_sentence_fallback(text: str, audio_question: bool) -> tuple[str, bool]:
    """Fallback formatting when Stanza is unavailable."""
    question = audio_question or _looks_like_question(text)

    first_char = text[0]
    if first_char.isalpha() and not first_char.isupper():
        text = first_char.upper() + text[1:]

    if text[-1] in ".!?":
        if text.endswith("?"):
            question = True
        return text, question

    if question:
        text = f"{text}?"
    else:
        text = f"{text}."

    return text, question


def _looks_like_question(text: str) -> bool:
    lowered = text.lower()
    if "?" in lowered:
        return True

    tokens = lowered.split()
    if not tokens:
        return False

    first = tokens[0].strip("¿¡")
    if first in QUESTION_WORDS:
        return True

    return lowered.startswith("is ") or lowered.startswith("are ") or lowered.startswith("do ")


def _normalise_spacing(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text)
    collapsed = re.sub(r"\s+([,.!?])", r"\1", collapsed)
    return collapsed.strip()

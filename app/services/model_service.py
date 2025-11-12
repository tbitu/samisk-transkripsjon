"""Model loading utilities for the Whisper large Northern Sámi model."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
    device = _resolve_device()
    torch_dtype = _resolve_dtype(device)

    logger.info("Loading model %s", MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    if isinstance(device, int):
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=400,
        chunk_length_s=15,
        batch_size=_default_batch_size(device),
        torch_dtype=torch_dtype,
        device=device if isinstance(device, int) else -1,
    )

    logger.info("Pipeline initialised successfully")
    return pipe


def transcribe(file_path: str | Path, *, temperature: float = 0.0) -> Dict[str, Any]:
    """Run transcription on the given audio file."""

    pipeline_ = get_asr_pipeline()
    combined_blocks: List[Dict[str, Any]] = []
    combined_segments: List[Dict[str, Any]] = []

    with TemporaryDirectory(prefix="whisper_sentences_") as tmpdir:
        sentence_chunks = prepare_sentence_chunks(Path(file_path), Path(tmpdir))
        if not sentence_chunks:
            logger.warning("No voice activity detected in %s", file_path)
            return {"text": "", "chunks": []}

        logger.info("Processing %d sentence chunks", len(sentence_chunks))

        inputs = [str(chunk.path) for chunk in sentence_chunks]
        pipeline_device = getattr(pipeline_, "device", None)
        batch_size = min(len(inputs), _default_batch_size(pipeline_device))
        results = _batched_transcribe(
            pipeline_,
            inputs,
            batch_size=batch_size,
            temperature=temperature,
        )

        for chunk, result in zip(sentence_chunks, results):
            raw_text = (result or {}).get("text", "")
            formatted_text, is_question = _format_sentence(raw_text, chunk.is_question)
            if not formatted_text:
                continue

            speaker_label = (chunk.speaker or "unknown").strip() or "unknown"
            if combined_blocks and combined_blocks[-1]["speaker"] == speaker_label:
                combined_blocks[-1]["texts"].append(formatted_text)
            else:
                combined_blocks.append({"speaker": speaker_label, "texts": [formatted_text]})
            combined_segments.append(
                {
                    "timestamp": (
                        chunk.start_ms / 1000.0,
                        chunk.end_ms / 1000.0,
                    ),
                    "text": formatted_text,
                    "speaker": chunk.speaker or "unknown",
                    "question": is_question,
                }
            )

    combined_text = "\n\n".join(
        f"{block['speaker']}: {' '.join(block['texts'])}"
        for block in combined_blocks
    )

    response: Dict[str, Any] = {"text": combined_text}

    if combined_segments:
        response["chunks"] = combined_segments

    return response


def _default_batch_size(device: Any) -> int:
    if isinstance(device, torch.device):
        return 4 if device.type == "cuda" else 1
    if isinstance(device, int):
        return 4
    return 1


def _batched_transcribe(
    pipeline_: Any,
    inputs: List[str],
    *,
    batch_size: int,
    temperature: float,
) -> List[Dict[str, Any]]:
    if not inputs:
        return []

    try:
        outputs = pipeline_(
            inputs,
            batch_size=batch_size,
            return_timestamps=True,
            generate_kwargs={"temperature": temperature},
        )
    except ValueError as exc:
        message = str(exc)
        if "Whisper did not predict an ending timestamp" in message:
            if len(inputs) == 1:
                fallback = pipeline_(
                    inputs,
                    batch_size=1,
                    return_timestamps=False,
                    generate_kwargs={"temperature": temperature},
                )
                return fallback if isinstance(fallback, list) else [fallback]

            outputs: List[Dict[str, Any]] = []
            for item in inputs:
                outputs.extend(
                    _batched_transcribe(
                        pipeline_, [item], batch_size=1, temperature=temperature
                    )
                )
            return outputs
        raise

    if isinstance(outputs, dict):
        return [outputs]
    return list(outputs)


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
    text = raw_text.strip()
    if not text:
        return "", False

    text = _normalise_spacing(text)
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

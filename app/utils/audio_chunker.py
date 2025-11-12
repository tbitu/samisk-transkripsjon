"""Utilities for splitting audio files into sentence-preserving chunks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import ffmpeg
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioChunk:
    """Metadata about an extracted audio chunk."""

    path: Path
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


def _ms_to_samples(ms: int, sample_rate: int) -> int:
    return max(0, int(round(ms * sample_rate / 1000)))


def _samples_to_ms(samples: int, sample_rate: int) -> int:
    return int(round(samples * 1000 / sample_rate))


def _load_audio_mono(source: Path, target_sample_rate: int) -> np.ndarray:
    try:
        stdout, _ = (
            ffmpeg.input(str(source))
            .output(
                "pipe:",
                format="s16le",
                ac=1,
                ar=target_sample_rate,
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:  # pragma: no cover - ffmpeg handles reporting
        message = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"Failed to decode audio with ffmpeg: {message}") from exc

    samples = np.frombuffer(stdout, dtype=np.int16)
    return samples


def _compute_dbfs(normalized_samples: np.ndarray) -> float:
    if normalized_samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(normalized_samples * normalized_samples)))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(rms)


def _find_silence_start(
    normalized_samples: np.ndarray,
    sample_rate: int,
    start_ms: int,
    search_end_ms: int,
    min_silence_len_ms: int,
    silence_linear_threshold: float,
) -> Optional[int]:
    if search_end_ms <= start_ms:
        return None

    min_len_samples = max(1, _ms_to_samples(min_silence_len_ms, sample_rate))
    step_samples = max(1, sample_rate // 1000)

    start_sample = _ms_to_samples(start_ms, sample_rate)
    end_sample = _ms_to_samples(search_end_ms, sample_rate)
    limit = end_sample - min_len_samples

    if limit <= start_sample:
        return None

    for cursor in range(start_sample, limit + 1, step_samples):
        window = normalized_samples[cursor:cursor + min_len_samples]
        if window.size < min_len_samples:
            break
        rms = float(np.sqrt(np.mean(window * window)))
        if rms <= silence_linear_threshold:
            return int(round(cursor * 1000 / sample_rate))
    return None


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    sf.write(path, samples.astype(np.int16), sample_rate, subtype="PCM_16")


def chunk_audio_file(
    source: Path,
    output_dir: Path,
    *,
    target_sample_rate: int = 16_000,
    min_chunk_ms: int = 120_000,
    max_chunk_ms: int = 150_000,
    lookahead_ms: int = 30_000,
    min_silence_len_ms: int = 400,
    silence_thresh_db: Optional[int] = None,
    keep_silence_ms: int = 200,
) -> List[AudioChunk]:
    """Split ``source`` into smaller WAV files and return their metadata."""

    if min_chunk_ms >= max_chunk_ms:
        raise ValueError("min_chunk_ms must be smaller than max_chunk_ms")

    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_audio_mono(source, target_sample_rate)
    normalized_samples = samples.astype(np.float32) / 32768.0
    total_duration_ms = _samples_to_ms(samples.size, target_sample_rate)

    if silence_thresh_db is None:
        reference = _compute_dbfs(normalized_samples)
        silence_thresh_db = int((reference if reference != float("-inf") else -48.0) - 16)

    silence_threshold_linear = 10 ** (silence_thresh_db / 20.0)

    chunks: List[AudioChunk] = []
    cursor = 0
    chunk_index = 0

    while cursor < total_duration_ms:
        min_end = min(total_duration_ms, cursor + min_chunk_ms)
        max_end = min(total_duration_ms, cursor + max_chunk_ms)
        search_end = min(total_duration_ms, max_end + lookahead_ms)

        chunk_end = _find_silence_start(
            normalized_samples,
            target_sample_rate,
            min_end,
            search_end,
            min_silence_len_ms,
            silence_threshold_linear,
        )

        if chunk_end is None:
            chunk_end = max_end

        if chunk_end - cursor < min_chunk_ms and chunk_end < total_duration_ms:
            chunk_end = min(total_duration_ms, cursor + min_chunk_ms)

        if chunk_end <= cursor:
            chunk_end = min(total_duration_ms, cursor + max(min_chunk_ms, min_silence_len_ms))

        padded_start = max(0, cursor - keep_silence_ms)
        padded_end = min(total_duration_ms, chunk_end + keep_silence_ms)

        start_sample = _ms_to_samples(padded_start, target_sample_rate)
        end_sample = _ms_to_samples(padded_end, target_sample_rate)

        segment_samples = samples[start_sample:end_sample]
        file_path = output_dir / f"chunk_{chunk_index:04d}.wav"
        _write_wav(file_path, segment_samples, target_sample_rate)

        chunks.append(AudioChunk(path=file_path, start_ms=padded_start, end_ms=padded_end))

        chunk_index += 1
        cursor = chunk_end

    if not chunks:
        file_path = output_dir / "chunk_0000.wav"
        _write_wav(file_path, samples, target_sample_rate)
        chunks.append(AudioChunk(path=file_path, start_ms=0, end_ms=total_duration_ms))

    return chunks

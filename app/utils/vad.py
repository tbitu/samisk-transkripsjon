"""Voice Activity Detection using Silero VAD model."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechSegment:
    """A detected speech segment with start and end times in milliseconds."""
    
    start_ms: int
    end_ms: int
    
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@lru_cache(maxsize=1)
def _load_silero_vad_model() -> Tuple:
    """Load and cache the Silero VAD model and utility functions."""
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        
        # Move model to GPU when available for better performance
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        logger.info("Silero VAD model loaded successfully (%s)", device_info)
        return model, utils
    except Exception as exc:
        logger.error("Failed to load Silero VAD model: %s", exc)
        raise RuntimeError(f"Failed to load Silero VAD model: {exc}") from exc


def detect_speech_segments(
    audio_path: Path,
    *,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    sample_rate: int = 16000,
    threshold: float = 0.5,
) -> List[SpeechSegment]:
    """
    Detect speech segments in an audio file using Silero VAD.
    
    Args:
        audio_path: Path to the audio file (WAV format, mono, 16kHz recommended)
        min_speech_duration_ms: Minimum duration of a speech segment to keep
        min_silence_duration_ms: Minimum silence duration to split segments
        sample_rate: Target sample rate for processing (Silero works best at 16kHz)
        threshold: VAD threshold (0.0-1.0), higher = more aggressive filtering
        
    Returns:
        List of SpeechSegment objects with start_ms and end_ms times
    """
    model, utils = _load_silero_vad_model()
    
    # Extract utility functions
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    try:
        # Load audio - Silero expects torch tensor at 16kHz
        wav = read_audio(str(audio_path), sampling_rate=sample_rate)
        
        # Ensure tensor is on same device as model to avoid device mismatch
        device = next(model.parameters()).device
        wav = wav.to(device)
        
        # Get speech timestamps (in samples)
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False,  # Return samples, we'll convert to ms
        )
        
        # Convert to milliseconds
        segments = []
        for ts in speech_timestamps:
            start_sample = ts['start']
            end_sample = ts['end']
            
            start_ms = int(round(start_sample * 1000 / sample_rate))
            end_ms = int(round(end_sample * 1000 / sample_rate))
            
            segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))
        
        logger.debug(
            "Detected %d speech segments in %s (total duration: %.2fs)",
            len(segments),
            audio_path.name,
            sum(s.duration_ms for s in segments) / 1000.0,
        )
        
        return segments
        
    except Exception as exc:
        logger.error("Failed to detect speech in %s: %s", audio_path, exc)
        raise RuntimeError(f"VAD processing failed: {exc}") from exc


def merge_adjacent_segments(
    segments: List[SpeechSegment],
    max_gap_ms: int = 300,
    max_duration_ms: int = 20_000,
) -> List[SpeechSegment]:
    """
    Merge adjacent speech segments that are close together.
    
    Args:
        segments: List of speech segments sorted by start time
        max_gap_ms: Maximum gap between segments to merge
        max_duration_ms: Maximum duration for merged segments
        
    Returns:
        List of merged speech segments
    """
    if not segments:
        return []
    
    # Sort by start time just to be sure
    sorted_segments = sorted(segments, key=lambda s: s.start_ms)
    
    merged = []
    current = sorted_segments[0]
    
    for next_seg in sorted_segments[1:]:
        gap = next_seg.start_ms - current.end_ms
        merged_duration = next_seg.end_ms - current.start_ms
        
        # Merge if gap is small and result isn't too long
        if gap <= max_gap_ms and merged_duration <= max_duration_ms:
            current = SpeechSegment(
                start_ms=current.start_ms,
                end_ms=next_seg.end_ms,
            )
        else:
            merged.append(current)
            current = next_seg
    
    merged.append(current)
    
    return merged


def detect_speech_in_array(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    *,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    threshold: float = 0.5,
) -> List[SpeechSegment]:
    """
    Detect speech segments directly from a numpy array (optimized, no file I/O).
    
    Args:
        audio_array: Audio data as numpy array (mono, float32 or int16)
        sample_rate: Sample rate of the audio
        min_speech_duration_ms: Minimum duration of a speech segment to keep
        min_silence_duration_ms: Minimum silence duration to split segments
        threshold: VAD threshold (0.0-1.0)
        
    Returns:
        List of SpeechSegment objects with start_ms and end_ms times
    """
    model, utils = _load_silero_vad_model()
    (get_speech_timestamps, _, read_audio, *_) = utils
    
    try:
        # Ensure audio is float32 and normalized to [-1, 1]
        if audio_array.dtype != np.float32:
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            else:
                audio_array = audio_array.astype(np.float32)
        
        # Convert to torch tensor and move to same device as model
        wav = torch.from_numpy(audio_array).float()
        
        # Move tensor to same device as model to avoid device mismatch
        device = next(model.parameters()).device
        wav = wav.to(device)
        
        # Resample if needed
        if sample_rate != 16000:
            # Simple warning - ideally should resample but avoid extra dependency
            logger.warning("VAD works best at 16kHz, got %dHz", sample_rate)
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False,
        )
        
        # Convert to milliseconds
        segments = []
        for ts in speech_timestamps:
            start_sample = ts['start']
            end_sample = ts['end']
            
            start_ms = int(round(start_sample * 1000 / sample_rate))
            end_ms = int(round(end_sample * 1000 / sample_rate))
            
            segments.append(SpeechSegment(start_ms=start_ms, end_ms=end_ms))
        
        return segments
        
    except Exception as exc:
        logger.error("Failed to detect speech in array: %s", exc)
        raise RuntimeError(f"VAD processing failed: {exc}") from exc


def refine_segment_with_vad(
    audio_path: Path,
    start_ms: int,
    end_ms: int,
    *,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    sample_rate: int = 16000,
    threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Refine a time segment by detecting speech within it.
    
    DEPRECATED: This function loads the entire file each time. 
    Use refine_segment_with_vad_cached instead for better performance.
    
    Returns relative offsets within the segment. Caller must add start_ms
    to get absolute timestamps.
    
    Args:
        audio_path: Path to the full audio file
        start_ms: Start of the segment to analyze (in full file time)
        end_ms: End of the segment to analyze (in full file time)
        threshold: VAD threshold (0.0-1.0)
        
    Returns:
        List of (relative_start_ms, relative_end_ms) tuples within the segment
    """
    try:
        data, sr = sf.read(str(audio_path))
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample to target sample rate if needed (VAD requires 16kHz)
        if sr != sample_rate:
            import torchaudio.functional as AF
            import torch
            data_tensor = torch.from_numpy(data).float()
            data_tensor = AF.resample(data_tensor, sr, sample_rate)
            data = data_tensor.numpy()
            sr = sample_rate
        
        # Extract the time range (now at the correct sample rate)
        start_sample = int(start_ms * sr / 1000)
        end_sample = int(end_ms * sr / 1000)
        segment_data = data[start_sample:end_sample]
        
        # Use the optimized in-memory function
        speech_segments = detect_speech_in_array(
            segment_data,
            sample_rate=sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            threshold=threshold,
        )
        
        # Return as relative offsets
        return [(s.start_ms, s.end_ms) for s in speech_segments]
            
    except Exception as exc:
        logger.warning("Failed to refine segment with VAD: %s", exc)
        # Return the full segment as a fallback
        return [(0, end_ms - start_ms)]


def refine_segment_with_vad_cached(
    audio_data: np.ndarray,
    audio_sample_rate: int,
    start_ms: int,
    end_ms: int,
    *,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    threshold: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Refine a time segment by detecting speech (optimized with pre-loaded audio).
    
    This is much faster than refine_segment_with_vad because audio is already loaded.
    
    Args:
        audio_data: Full audio file as numpy array (mono)
        audio_sample_rate: Sample rate of the audio data
        start_ms: Start of the segment to analyze (in full file time)
        end_ms: End of the segment to analyze (in full file time)
        min_speech_duration_ms: Minimum speech duration
        min_silence_duration_ms: Minimum silence duration
        threshold: VAD threshold (0.0-1.0)
        
    Returns:
        List of (relative_start_ms, relative_end_ms) tuples within the segment
    """
    try:
        # Extract the time range from pre-loaded audio
        start_sample = int(start_ms * audio_sample_rate / 1000)
        end_sample = int(end_ms * audio_sample_rate / 1000)
        segment_data = audio_data[start_sample:end_sample]
        
        # Use the optimized in-memory function
        speech_segments = detect_speech_in_array(
            segment_data,
            sample_rate=audio_sample_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            threshold=threshold,
        )
        
        # Return as relative offsets
        return [(s.start_ms, s.end_ms) for s in speech_segments]
            
    except Exception as exc:
        logger.warning("Failed to refine segment with VAD: %s", exc)
        # Return the full segment as a fallback
        return [(0, end_ms - start_ms)]


def clear_vad_from_memory() -> None:
    """Clear Silero VAD model from GPU memory to free up resources."""
    import gc
    import time
    
    try:
        # CRITICAL: Get the cached model FIRST, move to CPU, THEN clear cache
        # This prevents race conditions and ensures proper cleanup
        try:
            model, utils = _load_silero_vad_model()
            
            # Move model to CPU before clearing cache
            if torch.cuda.is_available():
                cpu_device = torch.device("cpu")
                model.to(cpu_device)
                
                # Ensure GPU operations complete
                torch.cuda.synchronize()
                
        except Exception as move_exc:
            logger.debug("Could not move VAD model to CPU: %s", move_exc)
        
        # Now clear the cache
        _load_silero_vad_model.cache_clear()
        
        # Force garbage collection multiple times to ensure cleanup
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache and synchronize
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Add a small grace period for GPU driver to complete cleanup
            time.sleep(0.2)
            
        logger.info("Silero VAD model cleared from GPU memory")
    except Exception as exc:
        logger.debug("Failed to clear VAD from memory: %s", exc)

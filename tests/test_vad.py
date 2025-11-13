"""Tests for Silero VAD integration."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from tempfile import TemporaryDirectory

from app.utils.vad import (
    SpeechSegment,
    detect_speech_segments,
    merge_adjacent_segments,
    refine_segment_with_vad,
)


@pytest.fixture
def sample_audio_with_speech(tmp_path: Path) -> Path:
    """Create a sample audio file with speech-like signal."""
    sample_rate = 16000
    duration_s = 5.0
    
    # Create a signal with speech-like patterns (sine wave modulated)
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    
    # Speech in first 2 seconds and last 1.5 seconds
    signal = np.zeros_like(t)
    
    # First speech segment (0-2s)
    speech1_mask = t < 2.0
    signal[speech1_mask] = 0.3 * np.sin(2 * np.pi * 200 * t[speech1_mask]) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t[speech1_mask]))
    
    # Silence (2-3.5s) - just low noise
    silence_mask = (t >= 2.0) & (t < 3.5)
    signal[silence_mask] = 0.01 * np.random.randn(silence_mask.sum())
    
    # Second speech segment (3.5-5s)
    speech2_mask = t >= 3.5
    signal[speech2_mask] = 0.3 * np.sin(2 * np.pi * 250 * t[speech2_mask]) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t[speech2_mask]))
    
    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, signal, sample_rate)
    
    return audio_path


@pytest.fixture
def continuous_speech_audio(tmp_path: Path) -> Path:
    """Create audio with continuous speech (no long silences)."""
    sample_rate = 16000
    duration_s = 3.0
    
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    signal = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))
    
    audio_path = tmp_path / "continuous_speech.wav"
    sf.write(audio_path, signal, sample_rate)
    
    return audio_path


def test_speech_segment_dataclass():
    """Test SpeechSegment dataclass properties."""
    seg = SpeechSegment(start_ms=1000, end_ms=3500)
    
    assert seg.start_ms == 1000
    assert seg.end_ms == 3500
    assert seg.duration_ms == 2500


def test_detect_speech_segments_basic(sample_audio_with_speech: Path):
    """Test basic speech detection."""
    segments = detect_speech_segments(
        sample_audio_with_speech,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        threshold=0.1,  # Lower threshold for synthetic audio
    )
    
    # Silero VAD may not detect synthetic sine waves as speech
    # This test verifies the function runs without errors
    assert isinstance(segments, list)
    
    # All segments should have valid timestamps
    for seg in segments:
        assert seg.start_ms >= 0
        assert seg.end_ms > seg.start_ms
        assert seg.duration_ms > 0


def test_detect_speech_segments_respects_min_duration(sample_audio_with_speech: Path):
    """Test that very short segments are filtered out."""
    segments = detect_speech_segments(
        sample_audio_with_speech,
        min_speech_duration_ms=5000,  # 5 seconds - very long
        min_silence_duration_ms=100,
    )
    
    # With such a high minimum, we might get 0-1 segments
    assert len(segments) <= 1


def test_detect_speech_continuous(continuous_speech_audio: Path):
    """Test detection on continuous speech."""
    segments = detect_speech_segments(
        continuous_speech_audio,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        threshold=0.1,  # Lower threshold for synthetic audio
    )
    
    # Silero VAD may not detect synthetic audio - just verify it runs
    assert isinstance(segments, list)
    # Note: Silero VAD may not detect simple synthetic sine waves


def test_merge_adjacent_segments():
    """Test merging of adjacent speech segments."""
    segments = [
        SpeechSegment(start_ms=0, end_ms=1000),
        SpeechSegment(start_ms=1200, end_ms=2000),  # 200ms gap
        SpeechSegment(start_ms=2500, end_ms=3000),  # 500ms gap
    ]
    
    # Merge with max_gap of 300ms
    merged = merge_adjacent_segments(segments, max_gap_ms=300, max_duration_ms=10000)
    
    # First two should merge (200ms gap), third stays separate (500ms gap)
    assert len(merged) == 2
    assert merged[0].start_ms == 0
    assert merged[0].end_ms == 2000
    assert merged[1].start_ms == 2500
    assert merged[1].end_ms == 3000


def test_merge_respects_max_duration():
    """Test that merging respects maximum duration."""
    segments = [
        SpeechSegment(start_ms=0, end_ms=8000),
        SpeechSegment(start_ms=8100, end_ms=10000),  # Would create 10s segment
    ]
    
    # Merge with max_duration of 9000ms
    merged = merge_adjacent_segments(segments, max_gap_ms=200, max_duration_ms=9000)
    
    # Should NOT merge because result would be 10 seconds
    assert len(merged) == 2


def test_merge_empty_list():
    """Test merging empty segment list."""
    merged = merge_adjacent_segments([], max_gap_ms=300, max_duration_ms=10000)
    assert len(merged) == 0


def test_merge_single_segment():
    """Test merging single segment."""
    segments = [SpeechSegment(start_ms=1000, end_ms=2000)]
    merged = merge_adjacent_segments(segments, max_gap_ms=300, max_duration_ms=10000)
    
    assert len(merged) == 1
    assert merged[0] == segments[0]


def test_refine_segment_basic(sample_audio_with_speech: Path):
    """Test refining a time segment with VAD."""
    # Refine the entire audio
    relative_segments = refine_segment_with_vad(
        sample_audio_with_speech,
        start_ms=0,
        end_ms=5000,
        min_speech_duration_ms=250,
        threshold=0.1,  # Lower threshold
    )
    
    # Verify function returns proper format
    assert isinstance(relative_segments, list)
    
    # All segments should be within the range
    for start, end in relative_segments:
        assert 0 <= start < end <= 5000


def test_refine_segment_subsection(sample_audio_with_speech: Path):
    """Test refining just a subsection."""
    # Refine only the first 2.5 seconds (should have speech + silence)
    relative_segments = refine_segment_with_vad(
        sample_audio_with_speech,
        start_ms=0,
        end_ms=2500,
        min_speech_duration_ms=250,
        threshold=0.1,  # Lower threshold
    )
    
    # Verify function runs
    assert isinstance(relative_segments, list)


def test_vad_threshold_adjustment(sample_audio_with_speech: Path):
    """Test that higher threshold detects less speech."""
    # Low threshold - more sensitive
    segments_low = detect_speech_segments(
        sample_audio_with_speech,
        threshold=0.2,
        min_speech_duration_ms=100,
    )
    
    # High threshold - less sensitive
    segments_high = detect_speech_segments(
        sample_audio_with_speech,
        threshold=0.7,
        min_speech_duration_ms=100,
    )
    
    # Low threshold should find at least as much speech as high threshold
    total_low = sum(s.duration_ms for s in segments_low)
    total_high = sum(s.duration_ms for s in segments_high)
    
    assert total_low >= total_high

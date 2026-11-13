"""
Unit and integration tests for detector.py:
- Top-half average aggregation logic
- Wav2Vec2 single clip classification (0.0-1.0 scale)
- Real human vs. AI clone separation
- Segment timeline scoring
"""
import pytest
import numpy as np
from detector import DeepfakeDetector
from audio_utils import load_and_resample, cut_segments

def test_calculate_clip_score_aggregation():
    # Empty
    assert DeepfakeDetector.calculate_clip_score([]) == 0.0

    # Single score
    assert DeepfakeDetector.calculate_clip_score([0.75]) == 0.75

    # Even number of segments: [0.1, 0.2, 0.8, 0.9]
    # Top half (2 items) -> [0.9, 0.8] -> mean = 0.85
    scores_even = [0.1, 0.2, 0.8, 0.9]
    assert DeepfakeDetector.calculate_clip_score(scores_even) == 0.85

    # Odd number of segments: [0.1, 0.3, 0.9]
    # ceil(3/2) = 2 items -> [0.9, 0.3] -> mean = 0.60
    scores_odd = [0.1, 0.3, 0.9]
    assert DeepfakeDetector.calculate_clip_score(scores_odd) == 0.60

def test_score_clip_real_vs_clone():
    detector = DeepfakeDetector()

    # Real human test
    real_audio, sr = load_and_resample("samples/sample_real_human.wav")
    real_score = detector.score_clip(real_audio[:16000 * 3], sr=sr)
    assert 0.0 <= real_score <= 1.0
    assert real_score < 0.2, f"Expected real human score < 0.2, got {real_score}"

    # AI clone test
    clone_audio, sr = load_and_resample("samples/sample_ai_clone.wav")
    clone_score = detector.score_clip(clone_audio[:16000 * 3], sr=sr)
    assert 0.0 <= clone_score <= 1.0
    assert clone_score > 0.8, f"Expected AI clone score > 0.8, got {clone_score}"

def test_score_segments_timeline():
    detector = DeepfakeDetector()
    clone_audio, sr = load_and_resample("samples/sample_ai_clone.wav")
    segments = cut_segments(clone_audio[:16000 * 4], sr=sr, segment_duration=2.0, hop_duration=1.0)

    scored = detector.score_segments(segments, sr=sr)
    assert len(scored) == len(segments)
    for seg in scored:
        assert "start" in seg
        assert "end" in seg
        assert "ai_score" in seg
        assert 0.0 <= seg["ai_score"] <= 1.0

    # Calculate aggregate
    clip_score = DeepfakeDetector.calculate_clip_score([s["ai_score"] for s in scored])
    assert clip_score > 0.8

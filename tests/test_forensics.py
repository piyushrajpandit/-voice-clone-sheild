"""
Unit tests for forensics.py:
- F0 Pitch & Jitter extraction
- Spectral centroid & flatness metrics
- Heuristic anomaly rule evaluation & score normalization
"""
import pytest
import numpy as np
from forensics import measure_forensics
from audio_utils import load_and_resample

def test_measure_forensics_real_human():
    audio, sr = load_and_resample("samples/sample_real_human.wav")
    res = measure_forensics(audio[:16000 * 4], sr=sr)

    for key in ["pitch_f0_mean", "pitch_jitter_std", "spectral_centroid", "spectral_flatness", "forensics_score"]:
        assert key in res, f"Missing key: {key}"
        assert not np.isnan(res[key]), f"Value for {key} is NaN"
        assert not np.isinf(res[key]), f"Value for {key} is Inf"

    assert 0.0 <= res["forensics_score"] <= 1.0
    assert res["pitch_f0_mean"] > 60.0, f"Expected voiced F0 > 60Hz, got {res['pitch_f0_mean']}"

def test_measure_forensics_ai_clone():
    audio, sr = load_and_resample("samples/sample_ai_clone.wav")
    res = measure_forensics(audio[:16000 * 4], sr=sr)

    assert 0.0 <= res["forensics_score"] <= 1.0
    assert res["spectral_centroid"] > 0.0
    assert res["spectral_flatness"] >= 0.0

def test_measure_forensics_empty():
    empty = np.array([], dtype=np.float32)
    res = measure_forensics(empty)
    assert res["forensics_score"] == 0.0
    assert res["pitch_f0_mean"] == 0.0

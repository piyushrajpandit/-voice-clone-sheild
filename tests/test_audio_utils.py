"""
Unit tests for audio_utils.py:
- 16 kHz mono WAV conversion (from file and bytes)
- Multi-channel downmix & resampling
- Peak normalization and DC offset removal
- Windowed segment slicing (timestamps & edge cases)
"""
import io
import numpy as np
import soundfile as sf
import pytest
from audio_utils import load_and_resample, cut_segments

def test_load_and_resample_from_path():
    audio, sr = load_and_resample("samples/sample_real_human.wav", target_sr=16000)
    assert sr == 16000
    assert audio.ndim == 1, "Audio must be 1D mono"
    assert audio.dtype == np.float32, "Audio dtype must be float32"
    assert np.max(np.abs(audio)) <= 0.9501, "Audio peak must be normalized <= 0.95"
    assert len(audio) > 16000 * 0.5, "Audio must contain at least 0.5s of speech"

def test_load_and_resample_from_bytes():
    with open("samples/sample_ai_clone.wav", "rb") as f:
        raw_bytes = f.read()

    audio, sr = load_and_resample(raw_bytes, target_sr=16000)
    assert sr == 16000
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 0.9501

def test_load_and_resample_stereo_44k():
    # Create in-memory 44.1kHz stereo audio
    duration_sec = 2.0
    sr_orig = 44100
    t = np.linspace(0, duration_sec, int(sr_orig * duration_sec), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([left, right]).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, stereo, sr_orig, format="WAV")
    wav_bytes = buf.getvalue()

    audio, sr = load_and_resample(wav_bytes, target_sr=16000)
    assert sr == 16000
    assert audio.ndim == 1
    # Check length is close to 2.0s * 16000
    expected_samples = int(duration_sec * 16000)
    assert abs(len(audio) - expected_samples) < 16000 * 0.1

def test_cut_segments_normal():
    # 5.0 seconds audio at 16kHz
    sr = 16000
    audio = np.random.uniform(-0.5, 0.5, 5 * sr).astype(np.float32)
    segments = cut_segments(audio, sr=sr, segment_duration=2.0, hop_duration=1.0)

    assert len(segments) == 4
    # Check segment 0: 0.0 -> 2.0
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 2.0
    assert len(segments[0]["audio"]) == 2 * sr

    # Check segment 1: 1.0 -> 3.0
    assert segments[1]["start"] == 1.0
    assert segments[1]["end"] == 3.0

    # Check segment 3: 3.0 -> 5.0
    assert segments[3]["start"] == 3.0
    assert segments[3]["end"] == 5.0

def test_cut_segments_short_audio():
    # 1.2 seconds audio (< 2.0s)
    sr = 16000
    audio = np.random.uniform(-0.5, 0.5, int(1.2 * sr)).astype(np.float32)
    segments = cut_segments(audio, sr=sr, segment_duration=2.0, hop_duration=1.0)

    assert len(segments) == 1
    assert segments[0]["start"] == 0.0
    assert segments[0]["end"] == 1.2
    assert len(segments[0]["audio"]) == int(1.2 * sr)

"""
Unit tests for Mel-spectrogram PNG generation in audio_utils.py.
"""
import os
import pytest
from audio_utils import load_and_resample, generate_spectrogram

def test_generate_spectrogram_png():
    audio, sr = load_and_resample("samples/sample_ai_clone.wav")
    test_out = "outputs/test_spec_output.png"

    # Remove if exists from earlier run
    if os.path.exists(test_out):
        os.remove(test_out)

    result_path = generate_spectrogram(audio[:16000 * 3], sr=sr, output_path=test_out)

    assert result_path == test_out
    assert os.path.exists(test_out), "Spectrogram PNG must be written to disk"
    file_size = os.path.getsize(test_out)
    assert file_size > 5000, f"Expected PNG size > 5000 bytes, got {file_size}"

    # Verify PNG magic bytes
    with open(test_out, "rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", "File must have valid PNG magic bytes"

    # Cleanup
    if os.path.exists(test_out):
        os.remove(test_out)

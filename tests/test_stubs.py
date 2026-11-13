"""
Verify repository setup, module imports, and file stubs for Member 1.
"""
import os
import pytest

def test_directory_layout():
    assert os.path.exists("legacy/app_original.py"), "Legacy app must exist in legacy/"
    assert os.path.exists("outputs/.gitkeep"), "outputs/ directory must exist"
    assert os.path.exists("samples"), "samples/ directory must exist"
    assert os.path.exists("requirements.txt"), "requirements.txt must exist"

def test_module_stubs_importable():
    import audio_utils
    import detector
    import forensics
    import pipeline

    assert hasattr(audio_utils, "load_and_resample")
    assert hasattr(audio_utils, "cut_segments")
    assert hasattr(audio_utils, "generate_spectrogram")

    assert hasattr(detector, "DeepfakeDetector")
    assert hasattr(forensics, "measure_forensics")
    assert hasattr(pipeline, "analyze_voice")

def test_sample_audio_files_exist():
    assert os.path.exists("samples/sample_real_human.wav"), "sample_real_human.wav must exist"
    assert os.path.exists("samples/sample_ai_clone.wav"), "sample_ai_clone.wav must exist"

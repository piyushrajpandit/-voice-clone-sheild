"""
pipeline.py - Main analyze_voice orchestration pipeline for Member 1.
Fulfills Section 5.2 schema specification and under 5-second latency requirement.
"""
from typing import Dict, Any, Union, Optional
import time
import uuid
import os
import numpy as np

from audio_utils import load_and_resample, cut_segments, generate_spectrogram
from detector import DeepfakeDetector
from forensics import measure_forensics

# Global detector instance for fast reuse (avoids reloading model weights per call)
_GLOBAL_DETECTOR: Optional[DeepfakeDetector] = None

def get_detector() -> DeepfakeDetector:
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = DeepfakeDetector()
    return _GLOBAL_DETECTOR

def analyze_voice(
    audio_input: Union[str, bytes],
    clip_id: Optional[str] = None,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Complete Voice Detection Pipeline (Section 5.2):
    1. Resample and normalize to 16 kHz mono WAV array.
    2. Slice into sliding window segments (2.0s duration, 1.0s hop).
    3. Score every segment using Wav2Vec2 transformer and aggregate top-half ai_score.
    4. Render and save forensic Mel-spectrogram PNG to outputs/.
    5. Extract acoustic forensics (F0 pitch steadiness, spectral centroid, noise flatness).
    6. Return unified JSON schema with latency profiling.

    Args:
        audio_input: File path (str) or raw audio bytes.
        clip_id: Optional identifier string. Generated if not supplied.
        output_dir: Directory to save generated spectrogram PNG.

    Returns:
        Dict adhering to Section 5.2:
        {
            'clip_id': str,
            'ai_score': float,           # 0.0 - 1.0 (top-half average)
            'segments': list,            # [{'start': 0.0, 'end': 2.0, 'ai_score': float}, ...]
            'spectrogram_path': str,     # path to PNG in outputs/
            'forensics': dict,           # F0, centroid, flatness, forensics_score
            'processing_time_sec': float # Latency benchmark
            'duration_sec': float        # Audio duration in seconds
        }
    """
    start_time = time.time()

    if not clip_id:
        clip_id = f"clip_{uuid.uuid4().hex[:8]}"

    # 1. 16kHz Mono Loading & Resampling
    audio_array, sr = load_and_resample(audio_input, target_sr=16000)
    duration_sec = round(float(len(audio_array)) / sr, 2)

    # 2. Windowed Segmentation
    segments = cut_segments(audio_array, sr=sr, segment_duration=2.0, hop_duration=1.0)

    # 3. Neural Deepfake Inference & Aggregation
    detector = get_detector()
    scored_segments = detector.score_segments(segments, sr=sr)
    segment_ai_scores = [s["ai_score"] for s in scored_segments]
    clip_ai_score = DeepfakeDetector.calculate_clip_score(segment_ai_scores)

    # 4. Forensic Spectrogram PNG Export
    spec_filename = f"{clip_id}_spectrogram.png"
    spec_path = os.path.join(output_dir, spec_filename)
    generate_spectrogram(audio_array, sr=sr, output_path=spec_path)

    # 5. Acoustic Physics-Based Forensics
    forensics = measure_forensics(audio_array, sr=sr)

    # 6. Latency Profiling
    elapsed = round(float(time.time() - start_time), 3)

    return {
        "clip_id": clip_id,
        "ai_score": clip_ai_score,
        "segments": scored_segments,
        "spectrogram_path": spec_path,
        "forensics": forensics,
        "processing_time_sec": elapsed,
        "duration_sec": duration_sec
    }

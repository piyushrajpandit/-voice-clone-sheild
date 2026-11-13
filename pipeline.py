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

def _calibrate_score(neural_score: float, forensics_score: float) -> float:
    """
    Adaptive blending of neural Wav2Vec2 score with enhanced acoustic forensics.

    v2 Calibration Strategy — solves two opposing failure modes:
    ─────────────────────────────────────────────────────────────
    FAILURE A (False Positive — was fixed in v1):
      Raw neural model over-fires on real mic audio because browser OS processing
      (AGC, echo cancel) makes real voices look like TTS to a studio-trained model.
      Fix: blend forensics (less mic-biased) to pull score down.

    FAILURE B (False Negative — fixed here in v2):
      Clean TTS voices (ChatGPT, ElevenLabs) score low on forensics because they
      have clean spectrum and steady pitch — the v1 forensics saw "clean = human".
      Fix: forensics v2 now penalizes *over-regularity* (too-steady RMS and ZCR,
      ultra-flat pitch) which are TTS signatures. Forensics score is now HIGH for
      TTS, pulling calibrated score UP.

    Adaptive blending:
      • Base blend: 60% neural + 40% forensics.
      • Boost: if forensics > 0.55 (strong TTS signal), scale forensics weight up.
        This catches TTS voices where neural score is borderline (0.5–0.7).
      • Cap: final score clamped to [0.0, 1.0].

    Decision threshold: 0.60 (applied in server.py renderResults).
    """
    # Adaptive forensics weight: boost only when forensics is very confident (> 0.65)
    # This avoids over-penalizing real voices where forensics returns a moderate score
    if forensics_score > 0.65:
        # Forensics is confidently detecting TTS signatures — increase its influence
        forensics_weight = 0.40 + (forensics_score - 0.65) * 0.70   # up to ~0.65
        neural_weight    = 1.0 - forensics_weight
    else:
        # Normal blending — forensics tempers neural false positives on real mic audio
        forensics_weight = 0.40
        neural_weight    = 0.60

    blended = (neural_weight * neural_score) + (forensics_weight * forensics_score)
    return round(float(np.clip(blended, 0.0, 1.0)), 4)

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
    6. Compute calibrated_score blending neural + forensics for mic-quality robustness.
    7. Return unified JSON schema with latency profiling.

    Returns:
        Dict adhering to Section 5.2 plus calibrated_score:
        {
            'clip_id': str,
            'ai_score': float,            # Raw neural-only score (0.0 - 1.0)
            'calibrated_score': float,    # Blended neural (70%) + forensics (30%)
            'segments': list,             # [{'start': 0.0, 'end': 2.0, 'ai_score': float}, ...]
            'spectrogram_path': str,      # path to PNG in outputs/
            'forensics': dict,            # F0, centroid, flatness, forensics_score
            'processing_time_sec': float, # Latency benchmark
            'duration_sec': float         # Audio duration in seconds
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

    # 6. Calibrated blended score (reduces false positives on mic-quality audio)
    calibrated_score = _calibrate_score(clip_ai_score, forensics["forensics_score"])

    # 7. Latency Profiling
    elapsed = round(float(time.time() - start_time), 3)

    return {
        "clip_id": clip_id,
        "ai_score": clip_ai_score,
        "calibrated_score": calibrated_score,
        "segments": scored_segments,
        "spectrogram_path": spec_path,
        "forensics": forensics,
        "processing_time_sec": elapsed,
        "duration_sec": duration_sec
    }

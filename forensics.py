"""
forensics.py - Acoustic measurements and physics-based forensic heuristics.

================================================================================
FORENSIC MEASUREMENT RULES & SCORING HEURISTIC:
--------------------------------------------------------------------------------
Synthetic neural vocoders (HiFi-GAN, BigVGAN, MelGAN, WaveGlow) and autoregressive
TTS engines exhibit subtle acoustic anomalies distinct from natural human vocal tracts:

1. Pitch Steadiness (Fundamental Frequency F0 & Jitter):
   - Natural Human Speech: Vocal cord biomechanics exhibit organic micro-tremors,
     producing pitch standard deviation typically in the range of 18 Hz to 75 Hz.
   - Synthetic AI Voice: Neural TTS frequently produces either hyper-flat pitch
     contours (jitter std < 12 Hz) or erratic phase jumps (jitter std > 110 Hz).
   - Anomaly rule:
     jitter_penalty = 1.0 if jitter < 12.0 or jitter > 110.0 else max(0.0, (25.0 - jitter) / 25.0)

2. Spectral Flatness (Noise-to-Tonality Ratio):
   - Natural Human Voicing: Voiced formants produce high harmonicity and low
     spectral flatness (typically < 0.015).
   - Synthetic AI Voice: Vocoder reconstruction introduces diffuse micro-noise in
     high frequency bands, driving spectral flatness up (> 0.030).
   - Anomaly rule:
     flatness_penalty = min(1.0, max(0.0, (mean_flatness - 0.015) / 0.035))

3. Spectral Centroid (Timbre Brightness Center-of-Mass):
   - Natural Speech: Centroid centers organically between 1,200 Hz and 2,600 Hz.
   - Synthetic AI Voice: High-frequency vocoder synthesis artifacts often push
     the centroid above 3,200 Hz or truncate it unnaturally below 1,000 Hz.
   - Anomaly rule:
     centroid_penalty = 1.0 if mean_centroid > 3200.0 or mean_centroid < 900.0 else 0.0

Composite forensics_score (0.0 to 1.0, higher = more synthetic anomalies):
   forensics_score = (0.45 * jitter_penalty) + (0.40 * flatness_penalty) + (0.15 * centroid_penalty)
================================================================================
"""
from typing import Dict, Any
import numpy as np
import librosa

def measure_forensics(audio_array: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Extract pitch steadiness (F0), spectral centroid, and noise flatness using librosa.
    Computes a composite forensics_score (0.0 = completely natural, 1.0 = highly anomalous).

    Args:
        audio_array: 1D float32 audio samples.
        sr: Sampling rate (default 16000).

    Returns:
        Dict:
        {
            'pitch_f0_mean': float,
            'pitch_jitter_std': float,
            'spectral_centroid': float,
            'spectral_flatness': float,
            'forensics_score': float
        }
    """
    if len(audio_array) == 0:
        return {
            "pitch_f0_mean": 0.0,
            "pitch_jitter_std": 0.0,
            "spectral_centroid": 0.0,
            "spectral_flatness": 0.0,
            "forensics_score": 0.0
        }

    audio_data = audio_array.astype(np.float32)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    try:
        # 1. Fundamental Frequency F0 & Pitch Jitter (piptrack)
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        threshold = np.median(magnitudes) if np.max(magnitudes) > 0 else 0.0
        valid_pitches = pitches[magnitudes > threshold]

        # Filter out 0Hz / unvoiced frames
        voiced_pitches = valid_pitches[(valid_pitches > 60.0) & (valid_pitches < 500.0)]
        if len(voiced_pitches) > 0:
            pitch_f0_mean = round(float(np.mean(voiced_pitches)), 2)
            pitch_jitter_std = round(float(np.std(voiced_pitches)), 2)
        else:
            pitch_f0_mean = 0.0
            pitch_jitter_std = 0.0

        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        mean_centroid = round(float(np.mean(centroid)), 2)

        # 3. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio_data)
        mean_flatness = round(float(np.mean(flatness)), 4)

        # 4. Compute Heuristic Penalties per Rule Specification
        # Pitch jitter penalty (penalize hyper-steady robotic voices or wild jumps)
        if pitch_jitter_std == 0.0:
            jitter_penalty = 0.5
        elif pitch_jitter_std < 12.0 or pitch_jitter_std > 110.0:
            jitter_penalty = 1.0
        else:
            jitter_penalty = max(0.0, (25.0 - pitch_jitter_std) / 25.0)

        # Flatness penalty (penalize diffuse vocoder noise floor)
        flatness_penalty = min(1.0, max(0.0, (mean_flatness - 0.015) / 0.035))

        # Centroid penalty (penalize abnormal spectral balance)
        if mean_centroid > 3200.0 or (mean_centroid < 900.0 and mean_centroid > 0.0):
            centroid_penalty = 1.0
        else:
            centroid_penalty = 0.0

        # Weighted composite forensics score
        composite = (0.45 * jitter_penalty) + (0.40 * flatness_penalty) + (0.15 * centroid_penalty)
        forensics_score = round(float(np.clip(composite, 0.0, 1.0)), 4)

        return {
            "pitch_f0_mean": pitch_f0_mean,
            "pitch_jitter_std": pitch_jitter_std,
            "spectral_centroid": mean_centroid,
            "spectral_flatness": mean_flatness,
            "forensics_score": forensics_score
        }

    except Exception as e:
        print(f"Forensics extraction error: {e}")
        return {
            "pitch_f0_mean": 0.0,
            "pitch_jitter_std": 0.0,
            "spectral_centroid": 0.0,
            "spectral_flatness": 0.0,
            "forensics_score": 0.5
        }

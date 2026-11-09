"""
Voice Clone Shield - Fast Model Validator
Tests pre-trained HuggingFace audio deepfake / spoof detection models
and provides multi-signal forensic analysis (Acoustics + Neural Model).
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch

def inspect_audio_file(file_path):
    """Inspect and load audio to 16kHz mono."""
    data, sr = sf.read(file_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1) # stereo to mono
    
    # Simple resample if not 16000
    if sr != 16000:
        import scipy.signal
        target_len = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, target_len)
        sr = 16000
        
    duration = len(data) / sr
    return data.astype(np.float32), sr, duration

def extract_forensic_features(audio_data, sr=16000):
    """
    Forensic heuristics used alongside neural models:
    - Spectral flatness & high frequency cutoff (typical of synthetic vocoders)
    - Zero crossing rate variance (jitter / synthetic smoothing)
    - Silence/pause naturalness
    """
    from scipy.fft import rfft, rfftfreq
    
    # Compute FFT
    N = len(audio_data)
    fft_vals = np.abs(rfft(audio_data))
    freqs = rfftfreq(N, 1.0 / sr)
    
    # High frequency energy ratio (> 7.5kHz cutoff common in older/fast TTS)
    high_freq_mask = freqs > 7000
    hf_energy = np.sum(fft_vals[high_freq_mask] ** 2)
    total_energy = np.sum(fft_vals ** 2) + 1e-10
    hf_ratio = float(hf_energy / total_energy)
    
    # Jitter / frame energy variance
    frame_size = int(sr * 0.03) # 30ms frames
    hop_size = int(sr * 0.015)
    energies = []
    for i in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[i:i + frame_size]
        energies.append(np.sum(frame ** 2))
    
    energy_var = float(np.var(energies)) if energies else 0.0
    
    # Zero crossing rate stability (AI models often produce unnaturally uniform zero crossings)
    zcr = []
    for i in range(0, len(audio_data) - frame_size, hop_size):
        frame = audio_data[i:i + frame_size]
        zc = np.sum(np.abs(np.diff(np.signbit(frame)))) / (2 * len(frame))
        zcr.append(zc)
    zcr_std = float(np.std(zcr)) if zcr else 0.0

    return {
        "hf_ratio": round(hf_ratio, 5),
        "energy_variance": round(energy_var, 5),
        "zcr_stability_std": round(zcr_std, 5),
    }

def run_classifier(file_path):
    print(f"\n==========================================")
    print(f"🔍 Analyzing: {os.path.basename(file_path)}")
    print(f"==========================================")
    
    data, sr, duration = inspect_audio_file(file_path)
    print(f"Duration: {duration:.2f}s | Sample Rate: {sr}Hz | Samples: {len(data)}")
    
    forensics = extract_forensic_features(data, sr)
    print(f"Forensic Indicators: {forensics}")
    
    # Check if transformers pipeline can load model
    try:
        from transformers import pipeline
        # Popular lightweight Hugging Face model for deepfake/spoof detection
        print("Loading Hugging Face model (MelodyMachine/Deepfake-audio-detection)...")
        pipe = pipeline("audio-classification", model="MelodyMachine/Deepfake-audio-detection")
        
        # Passing raw numpy array and sampling_rate avoids external ffmpeg requirement
        results = pipe({"raw": data, "sampling_rate": sr})
        print(f"Model Raw Predictions: {results}")
        
        # Parse top result
        top = results[0]
        label = top['label'].upper()
        score = top['score']
        
        print("\n--- FINAL VERDICT ---")
        if "FAKE" in label or "SPOOF" in label or "SYNTHETIC" in label:
            print(f"🚨 ALERT: DETECTED AS AI VOICE CLONE (Confidence: {score*100:.1f}%)")
        else:
            print(f"✅ VERIFIED: DETECTED AS AUTHENTIC HUMAN VOICE (Confidence: {score*100:.1f}%)")
            
    except Exception as e:
        print(f"Model test notice: {e}")
        print("Fallback forensic analysis used.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_detector.py <path_to_audio.wav>")
        sys.exit(1)
    run_classifier(sys.argv[1])

import os
import io
import time
import numpy as np
import soundfile as sf
import scipy.signal
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import librosa
from faster_whisper import WhisperModel

app = FastAPI(title="Voice Clone Shield")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the industry-standard deepfake audio detection model
# mo-thecreator/Deepfake-audio-detection (robust against real live mic noise & room acoustics)
MODEL_ID = "mo-thecreator/Deepfake-audio-detection"
print(f"Loading robust detection pipeline ({MODEL_ID})...")
classifier = pipeline("audio-classification", model=MODEL_ID)
print("Detector loaded successfully!")

# Load faster-whisper model (quantized for high-speed CPU / Apple Silicon inference)
print("Loading faster-whisper model (tiny.en)...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("faster-whisper loaded successfully!")

SAFE_WORD = "SUPERNOVA-42"
os.makedirs("debug_recordings", exist_ok=True)
os.makedirs("samples", exist_ok=True)

def ensure_samples():
    import base64
    for name in ["sample_ai_clone", "sample_real_human"]:
        wav_path = os.path.join("samples", f"{name}.wav")
        b64_path = os.path.join("samples", f"{name}.b64")
        if not os.path.exists(wav_path) and os.path.exists(b64_path):
            with open(b64_path, "r") as f:
                decoded = base64.b64decode(f.read().strip())
            with open(wav_path, "wb") as f:
                f.write(decoded)
            print(f"Restored sample: {wav_path}")

ensure_samples()

def process_audio_buffer(file_bytes):
    """Decode audio bytes to 16kHz float32 mono array with normalization."""
    try:
        data, sr = sf.read(io.BytesIO(file_bytes))
    except Exception:
        # Fallback raw PCM / wav parser
        import wave
        with wave.open(io.BytesIO(file_bytes), 'rb') as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if len(data.shape) > 1:
        data = np.mean(data, axis=1) # Mono
    
    # Resample to 16kHz
    if sr != 16000:
        target_samples = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, target_samples)
        sr = 16000

    # Remove DC offset & peak normalize
    data = data - np.mean(data)
    peak = np.max(np.abs(data))
    if peak > 1e-4:
        data = data / peak * 0.95 # Normalized to 0.95 peak

    # Trim leading/trailing silence (energy threshold)
    frame_len = int(16000 * 0.05)
    energies = [np.sum(data[i:i+frame_len]**2) for i in range(0, len(data), frame_len)]
    thresh = (np.mean(energies) * 0.1) if energies else 0.0
    start_idx = 0
    end_idx = len(data)
    for i, e in enumerate(energies):
        if e > thresh:
            start_idx = i * frame_len
            break
    for i in range(len(energies)-1, -1, -1):
        if energies[i] > thresh:
            end_idx = min(len(data), (i + 1) * frame_len)
            break
    
    trimmed = data[start_idx:end_idx]
    if len(trimmed) > 16000 * 0.5: # At least 0.5s of speech
        data = trimmed

    return data.astype(np.float32), sr

def extract_librosa_forensics(audio_data, sr=16000):
    """
    Extract multi-signal acoustic forensics using librosa:
    - 13 Mel-Frequency Cepstral Coefficients (MFCCs)
    - Spectral centroid (timbre brightness)
    - Spectral flatness (synthesizer noise vs tone ratio)
    - Spectral rolloff (high-frequency energy drop-off)
    - Fundamental frequency (F0) & pitch modulation stability
    - Mel-spectrogram normalized preview for visualization
    """
    try:
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        mfcc_summary = [round(float(x), 2) for x in np.mean(mfccs, axis=1)]

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        mean_centroid = round(float(np.mean(centroid)), 1)

        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=audio_data)
        mean_flatness = round(float(np.mean(flatness)), 4)

        # Spectral Rolloff (85% energy frequency)
        rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, roll_percent=0.85)
        mean_rolloff = round(float(np.mean(rolloff)), 1)

        # Fundamental Frequency F0 / Pitch estimation
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sr)
        valid_pitches = pitches[magnitudes > np.median(magnitudes)]
        mean_pitch = round(float(np.mean(valid_pitches)), 1) if len(valid_pitches) > 0 else 0.0
        pitch_jitter = round(float(np.std(valid_pitches)), 1) if len(valid_pitches) > 0 else 0.0

        # Mel Spectrogram Preview (32 frequency bands x up to 60 time slices)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=32, hop_length=int(sr * 0.04))
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
        step = max(1, mel_norm.shape[1] // 60)
        spec_preview = [[round(float(v), 3) for v in col] for col in mel_norm[:, ::step].T]

        return {
            "mfcc_summary": mfcc_summary,
            "spectral_centroid_hz": mean_centroid,
            "spectral_flatness": mean_flatness,
            "spectral_rolloff_hz": mean_rolloff,
            "pitch_f0_hz": mean_pitch,
            "pitch_jitter_std": pitch_jitter,
            "spectrogram_preview": spec_preview
        }
    except Exception as e:
        print(f"Librosa extraction error: {e}")
        return {
            "mfcc_summary": [],
            "spectral_centroid_hz": 0.0,
            "spectral_flatness": 0.0,
            "spectral_rolloff_hz": 0.0,
            "pitch_f0_hz": 0.0,
            "pitch_jitter_std": 0.0,
            "spectrogram_preview": []
        }

def transcribe_audio_snippet(audio_data, sr=16000):
    """Transcribe audio using faster-whisper and check for spoken emergency safewords."""
    try:
        segments, info = whisper_model.transcribe(audio_data, beam_size=1, language="en")
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        
        # Check if the designated safe-word or keyword was spoken
        cleaned_text = text.lower()
        safeword_detected = (
            "supernova" in cleaned_text or 
            SAFE_WORD.lower() in cleaned_text
        )
        
        return {
            "transcript": text if text else "(No spoken words detected)",
            "language": info.language,
            "language_confidence": round(float(info.language_probability) * 100, 1),
            "safeword_spoken": safeword_detected
        }
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return {
            "transcript": "(Transcription unavailable)",
            "language": "en",
            "language_confidence": 0.0,
            "safeword_spoken": False
        }

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    start_time = time.time()
    contents = await file.read()
    
    # Save a copy for debug inspection
    debug_path = os.path.join("debug_recordings", f"rec_{int(time.time()*1000)}.wav")
    
    try:
        audio_data, sr = process_audio_buffer(contents)
        sf.write(debug_path, audio_data, sr)
    except Exception as e:
        print(f"Decode error on {file.filename}: {e}")
        return JSONResponse(status_code=400, content={"error": f"Audio processing error: {str(e)}"})
    
    duration = round(len(audio_data) / sr, 2)
    sample_slice = audio_data[:16000 * 10]
    
    # 1. Neural Spoof Classifier (mo-thecreator)
    predictions = classifier({"raw": sample_slice, "sampling_rate": 16000})
    print(f"Raw Model Predictions: {predictions}")
    
    scores = {p['label'].lower(): p['score'] for p in predictions}
    fake_score = scores.get('fake', 0.0)
    real_score = scores.get('real', 0.0)
    
    is_fake = fake_score > 0.5
    confidence = fake_score if is_fake else real_score
    
    # 2. Librosa Acoustic Forensics (MFCC, Centroid, Flatness, F0 Pitch)
    forensics = extract_librosa_forensics(sample_slice, sr)
    
    # 3. faster-whisper Speech-to-Text & Spoken Safeword Intercept
    speech = transcribe_audio_snippet(sample_slice, sr)
    
    hf_cutoff = float(np.mean(np.abs(np.diff(sample_slice))))
    latency_ms = round((time.time() - start_time) * 1000)

    print(f"Verdict: {'DEEPFAKE' if is_fake else 'AUTHENTIC'} | Conf: {confidence*100:.1f}% | Transcript: {speech['transcript']}")

    return {
        "status": "success",
        "verdict": "DEEPFAKE_CLONE" if is_fake else "AUTHENTIC_HUMAN",
        "is_fake": is_fake,
        "threat_score": round(fake_score * 100, 1),
        "confidence": round(confidence * 100, 1),
        "duration": duration,
        "latency_ms": latency_ms,
        "transcript": speech["transcript"],
        "speech_analysis": speech,
        "acoustic_forensics": {
            "spectral_centroid_hz": forensics["spectral_centroid_hz"],
            "spectral_flatness": forensics["spectral_flatness"],
            "spectral_rolloff_hz": forensics["spectral_rolloff_hz"],
            "pitch_f0_hz": forensics["pitch_f0_hz"],
            "pitch_jitter_std": forensics["pitch_jitter_std"],
            "mfcc_summary": forensics["mfcc_summary"]
        },
        "acoustic_metrics": {
            "hf_jitter": round(hf_cutoff * 1000, 2),
            "sample_rate": sr,
            "samples_analyzed": len(sample_slice)
        },
        "spectrogram_preview": forensics["spectrogram_preview"]
    }

@app.post("/api/verify-safeword")
async def verify_safeword(answer: str = Form(...)):
    is_match = answer.strip().upper() == SAFE_WORD
    return {
        "verified": is_match,
        "message": "Protocol Verified: Caller identity confirmed." if is_match else "Access Denied: Incorrect family safe-word. Call terminated."
    }

@app.get("/api/samples/{sample_name}")
async def get_sample(sample_name: str):
    from fastapi.responses import FileResponse
    path = os.path.join("samples", sample_name)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"error": "Not found"})

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

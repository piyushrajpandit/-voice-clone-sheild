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

def compute_spectrogram_preview(audio_data, sr=16000):
    """Compute spectrogram frequencies and time slices for frontend visualization."""
    from scipy.fft import rfft
    frame_size = int(sr * 0.04) # 40ms
    hop_size = int(sr * 0.02)
    spec = []
    
    total_frames = max(1, (len(audio_data) - frame_size) // hop_size)
    step = max(1, total_frames // 60)
    
    for i in range(0, len(audio_data) - frame_size, hop_size * step):
        frame = audio_data[i:i + frame_size] * np.hanning(frame_size)
        mags = np.abs(rfft(frame))[:32]
        spec.append([round(float(m), 3) for m in mags])
        
    return spec

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
    
    # Run through mo-thecreator model
    predictions = classifier({"raw": sample_slice, "sampling_rate": 16000})
    print(f"Raw Model Predictions: {predictions}")
    
    # mo-thecreator outputs labels 'real' and 'fake'
    scores = {p['label'].lower(): p['score'] for p in predictions}
    fake_score = scores.get('fake', 0.0)
    real_score = scores.get('real', 0.0)
    
    is_fake = fake_score > 0.5
    confidence = fake_score if is_fake else real_score
    
    hf_cutoff = float(np.mean(np.abs(np.diff(sample_slice))))
    spectrogram = compute_spectrogram_preview(sample_slice, sr)
    latency_ms = round((time.time() - start_time) * 1000)

    print(f"Verdict: {'DEEPFAKE' if is_fake else 'AUTHENTIC'} | Confidence: {confidence*100:.1f}% | Duration: {duration}s")

    return {
        "status": "success",
        "verdict": "DEEPFAKE_CLONE" if is_fake else "AUTHENTIC_HUMAN",
        "is_fake": is_fake,
        "threat_score": round(fake_score * 100, 1),
        "confidence": round(confidence * 100, 1),
        "duration": duration,
        "latency_ms": latency_ms,
        "acoustic_metrics": {
            "hf_jitter": round(hf_cutoff * 1000, 2),
            "sample_rate": sr,
            "samples_analyzed": len(sample_slice)
        },
        "spectrogram_preview": spectrogram
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

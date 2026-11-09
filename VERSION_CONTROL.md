# 🛡️ Voice Clone Shield — VERSION CONTROL v1.0
> **Created:** 2026-11-10 | **Purpose:** Snapshot of exact working state before Live Call Monitor feature is added.
> If anything breaks after new changes, follow the RESTORE STEPS section to get back here instantly.

---

## 📁 Project Location
```
/Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield/
```

---

## 📂 File Structure (Complete)

```
voice-clone-shield/
├── app.py                        ← FastAPI backend (193 lines, main server)
├── requirements.txt              ← Python dependencies (12 lines)
├── Dockerfile                    ← Container build config (Render/HuggingFace)
├── render.yaml                   ← Render.com deployment blueprint
├── README.md                     ← HuggingFace Space model card
├── .gitignore                    ← Git ignored files
├── .dockerignore                 ← Docker ignored files
├── static/
│   └── index.html                ← Full frontend UI (1069 lines)
├── samples/
│   ├── sample_ai_clone.wav       ← Fake/deepfake audio sample
│   ├── sample_ai_clone.b64       ← Base64 backup of fake sample
│   ├── sample_real_human.wav     ← Real human voice sample
│   ├── sample_real_human.b64     ← Base64 backup of real sample
│   └── sample_ai_synth.wav       ← Synthetic voice sample
├── venv/                         ← Python virtual environment (do NOT delete)
├── debug_recordings/             ← Auto-saved incoming audio (for debugging)
└── VERSION_CONTROL.md            ← This file
```

---

## ⚙️ How the System Works Right Now (v1.0)

### 1. Backend — app.py (FastAPI Server)

**Server starts on:** http://127.0.0.1:8000 (local) or PORT env var for cloud

**Model loaded at startup:**
```
mo-thecreator/Deepfake-audio-detection
```
- Wav2Vec2-based model fine-tuned on the ASVspoof dataset
- Outputs two labels: `real` and `fake` with confidence scores (0.0–1.0)
- Runs entirely on CPU (no GPU needed)
- Model cached at: ~/.cache/huggingface/ after first download

**Audio Processing Pipeline (step-by-step):**
```
1. Browser records audio → sends as WAV file (POST /api/analyze)
2. soundfile.read() decodes the WAV bytes into numpy array
3. If stereo → converted to mono (average both channels)
4. If sample rate ≠ 16000Hz → resampled via scipy.signal.resample()
5. DC offset removed: data = data - mean(data)
6. Peak normalized to 0.95 amplitude
7. Silence trimmed from start/end (energy threshold = 10% of mean frame energy)
8. Must be at least 0.5 seconds long to proceed
9. First 10 seconds fed into the classifier model
10. Scores extracted: fake_score and real_score
11. is_fake = True if fake_score > 0.5
12. Spectrogram computed: 40ms frames, 32 frequency bins for visualization
13. JSON response returned to browser
```

**API Endpoints:**
| Endpoint | Method | Purpose |
|---|---|---|
| / | GET | Serves the frontend HTML |
| /api/analyze | POST | Accepts audio file, returns deepfake verdict |
| /api/verify-safeword | POST | Verifies family safe-word (SUPERNOVA-42) |
| /api/samples/{name} | GET | Serves demo WAV files |
| /static/* | GET | Serves CSS/JS static assets |

**Current Safe Word:** SUPERNOVA-42

---

### 2. Frontend — static/index.html (1069 lines)

**Tech Stack:** Pure HTML + Vanilla CSS + Vanilla JavaScript (zero frameworks, zero npm)

**Design Colors:**
```
Background: #060913 (dark navy)
Accent Cyan: #06b6d4
Accent Blue: #3b82f6
AUTHENTIC color: #10b981 (green)
DEEPFAKE color: #f43f5e (red)
Fonts: Outfit + JetBrains Mono (from Google Fonts)
```

**Two UI Tabs:**
1. Voice Scan — Records microphone audio → sends to /api/analyze
2. Video Intercept — Records mic + screen video → extracts audio for analysis

**Recording Flow (Voice Scan tab, step-by-step):**
```
1. User clicks "Start Recording"
2. Browser requests mic permission via getUserMedia({audio:true})
3. AudioContext created → MediaStreamSource → ScriptProcessor (4096 sample buffer)
4. ScriptProcessor downmixes to mono → collects raw Float32 PCM at 16kHz
5. User clicks "Stop Recording"
6. Float32 array packed into WAV blob in-browser (raw PCM, NOT WebM/Opus)
7. WAV blob POSTed to /api/analyze via FormData fetch()
8. JSON response parsed → verdict card updated with animation
9. Spectrogram drawn on HTML5 canvas element
```

**Why raw PCM WAV (not WebM/Opus)?**
The browser default MediaRecorder produces WebM/Opus compressed audio.
Opus compression distorts acoustic artifacts that the model needs to detect deepfakes.
Using ScriptProcessor + manual WAV encoding preserves the exact waveform the model expects.

**Verdict Card:**
- Green glowing card → AUTHENTIC HUMAN voice
- Red pulsing card → DEEPFAKE CLONE detected
- Shows: Threat Score %, Confidence %, Duration (seconds), HF Jitter metric, API latency ms

**Safe-Word Modal:**
- Triggered manually by button, or auto-triggers when deepfake detected
- User types in SUPERNOVA-42 → verified against backend /api/verify-safeword
- Shows VERIFIED (green) or DENIED (red) banner

**Demo Samples:**
- "Test Real" and "Test Fake" buttons load pre-recorded WAV from /api/samples/
- These work without a microphone for demonstration

---

### 3. Python Environment

**Virtual environment location:** ./venv/

**Activate command:**
```bash
cd /Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield
source venv/bin/activate
```

**Key packages installed:**
```
torch>=2.2.0 (CPU wheels)
torchaudio>=2.2.0
transformers>=4.38.0  ← HuggingFace model loading
soundfile>=0.12.1     ← WAV/FLAC/OGG decoding
scipy>=1.12.0         ← Resampling + FFT spectrogram
numpy>=1.26.0         ← Array math
fastapi>=0.110.0      ← Web API framework
uvicorn>=0.28.0       ← ASGI server runner
python-multipart      ← Handles file upload in forms
requests>=2.31.0      ← HTTP client
```

---

## 🚀 HOW TO RUN LOCALLY (Exact Commands)

**Step 1 — Start the backend:**
```bash
cd /Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield
./venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

**Step 2 — Open in browser (desktop):**
```
http://127.0.0.1:8000
```

**Step 3 — Open on your phone (Cloudflare public tunnel):**
```bash
# In a NEW terminal:
cloudflared tunnel --url http://127.0.0.1:8000
# → Gives: https://random-words.trycloudflare.com
# → Open that URL on your phone
```

**Step 4 — Verify server is healthy:**
```bash
curl -s http://127.0.0.1:8000/ | head -5
# Should return HTML starting with <!DOCTYPE html>
```

---

## ✅ Known Working / Verified Results (v1.0)

Tested and confirmed correct outputs:
```
sample_real_human.wav → AUTHENTIC HUMAN | Confidence ~100% | Threat ~0%
sample_ai_clone.wav   → DEEPFAKE CLONE  | Confidence ~100% | Threat ~100%
Live mic (real voice) → AUTHENTIC HUMAN | Confidence ~99.9%
```
Server logs show: "Verdict: AUTHENTIC | Confidence: 100.0%" and "Verdict: DEEPFAKE | Confidence: 100.0%"

---

## 🔧 RESTORE STEPS (If Something Breaks After v1.1 Changes)

### Scenario A — Backend crashed / won't start
```bash
cd /Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield
pkill -f uvicorn
./venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

### Scenario B — Venv broken / packages missing
```bash
cd /Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield
rm -rf venv
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### Scenario C — Frontend broken, restore from git
```bash
cd /Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield
git log --oneline          # Find the v1.0 commit hash
git checkout <HASH> -- static/index.html
```

### Scenario D — Backend broken, restore app.py from git
```bash
cd /Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield
git log --oneline
git checkout <HASH> -- app.py
```

### Scenario E — Restore ENTIRE project to v1.0 state
```bash
cd /Users/piyushraj/.gemini/antigravity-ide/scratch/voice-clone-shield
git log --oneline          # Find commit tagged "version 1.0" or last commit before v1.1
git reset --hard <HASH>    # WARNING: This loses all uncommitted changes
```

### Scenario F — Cloudflare Tunnel URL expired
```bash
# Tunnels expire every ~8 hours. Just restart:
cloudflared tunnel --url http://127.0.0.1:8000
# New URL generated automatically — works identically
```

### Scenario G — Model giving wrong verdicts
```
1. Check debug_recordings/ — latest audio chunk saved there automatically
2. Open WAV file in QuickTime or VLC to hear what was actually recorded
3. If audio sounds garbled → browser encoding bug (ScriptProcessor issue)
4. If audio sounds clear but wrong verdict → try restarting the server (model reload)
5. Click "Test Fake" / "Test Real" buttons to confirm demo samples still work
6. If demos work but live mic doesn't → mic permission or sampleRate mismatch
```

---

## 📋 What Gets Added in v1.1 — Live Call Monitor

Changes planned AFTER this version control checkpoint:
- New "Call Monitor" panel in index.html (new tab or overlay)
- Continuous 4-second rolling audio capture (no manual start/stop)
- Auto-POST to /api/analyze every 4 seconds
- Live threat timeline visualization (chart updating in real time)
- Auto-alarm (audio beep + red flash) when 2 consecutive chunks > 70% fake
- Mobile-first responsive layout for phone use

New things that will NOT change from v1.0:
- app.py backend logic (same endpoints, same model)
- Python environment / requirements.txt
- Safe-word system
- Existing Voice Scan and Video Intercept tabs

---

## 🏷️ Version History

| Version | Date | Key Feature |
|---|---|---|
| v1.0 | 2026-11-10 | Manual record → analyze, deepfake detection, safe-word modal, video intercept |
| v1.1 | TBD | Live Call Monitor Mode (continuous auto-scanning for phone call interception) |

---
*Voice Clone Shield — VERSION_CONTROL.md | Last updated: 2026-11-10*

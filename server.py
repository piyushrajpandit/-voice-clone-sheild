"""
server.py - Local testing server for Voice Clone Shield.
Provides a local web dashboard to test audio samples, upload clips, or record via mic.
Runs on http://localhost:5050
"""
import os
import io
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from pipeline import analyze_voice

app = FastAPI(title="Voice Clone Shield - Local Lab")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Clone Shield — Local Test Lab</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #080c14;
            --bg-card: rgba(16, 24, 40, 0.85);
            --bg-subtle: #121c2e;
            --border: rgba(255, 255, 255, 0.1);
            --cyan: #00f2fe;
            --red: #f43f5e;
            --green: #10b981;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-tertiary: #64748b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-page);
            color: var(--text-primary);
            font-family: 'Plus Jakarta Sans', sans-serif;
            padding: 30px 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 960px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 28px;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-icon { font-size: 1.8rem; }
        .brand h1 {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .brand h1 span { color: var(--cyan); }
        .badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            padding: 4px 10px;
            border-radius: 9999px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-weight: 700;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
        }
        .section-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 16px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-family: 'JetBrains Mono', monospace;
        }
        .btn-group {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .btn {
            font-family: 'Plus Jakarta Sans', sans-serif;
            font-size: 0.92rem;
            font-weight: 700;
            padding: 12px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn-sample-human {
            background: rgba(16, 185, 129, 0.15);
            color: var(--green);
            border: 1px solid rgba(16, 185, 129, 0.4);
        }
        .btn-sample-human:hover { background: rgba(16, 185, 129, 0.25); }
        .btn-sample-clone {
            background: rgba(244, 63, 94, 0.15);
            color: var(--red);
            border: 1px solid rgba(244, 63, 94, 0.4);
        }
        .btn-sample-clone:hover { background: rgba(244, 63, 94, 0.25); }
        .btn-mic {
            background: rgba(0, 242, 254, 0.15);
            color: var(--cyan);
            border: 1px solid rgba(0, 242, 254, 0.4);
        }
        .btn-mic:hover { background: rgba(0, 242, 254, 0.25); }
        .btn-mic.recording {
            background: var(--red);
            color: #fff;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        .upload-row {
            display: flex;
            gap: 12px;
            align-items: center;
            border-top: 1px dashed var(--border);
            padding-top: 18px;
        }
        input[type="file"] {
            background: var(--bg-subtle);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 8px;
            color: var(--text-secondary);
            font-size: 0.85rem;
            cursor: pointer;
            flex: 1;
        }
        .btn-submit {
            background: linear-gradient(135deg, var(--cyan), #38bdf8);
            color: #040914;
        }
        #loading {
            display: none;
            text-align: center;
            padding: 40px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--cyan);
            font-weight: 700;
        }
        .spinner {
            display: inline-block;
            width: 28px;
            height: 28px;
            border: 3px solid rgba(0, 242, 254, 0.2);
            border-top-color: var(--cyan);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-bottom: 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Results Card */
        #results-container { display: none; }
        .result-banner {
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .banner-danger {
            background: rgba(244, 63, 94, 0.15);
            border: 1px solid rgba(244, 63, 94, 0.4);
            color: #fecdd3;
        }
        .banner-safe {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #a7f3d0;
        }
        .verdict-title { font-size: 1.3rem; font-weight: 800; }
        .verdict-sub { font-size: 0.85rem; opacity: 0.9; margin-top: 4px; }
        .score-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.8rem;
            font-weight: 800;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }
        .metric-box {
            background: var(--bg-subtle);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
        }
        .metric-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: var(--text-tertiary);
            margin-bottom: 6px;
            text-transform: uppercase;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text-primary);
        }
        .spectrogram-box {
            background: #040711;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
            margin-bottom: 20px;
        }
        .spectrogram-box img {
            width: 100%;
            border-radius: 6px;
            display: block;
        }
        .timeline-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .timeline-table th {
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text-tertiary);
            font-size: 0.75rem;
        }
        .timeline-table td {
            padding: 10px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }
        .tag-fake { color: var(--red); font-weight: 700; }
        .tag-safe { color: var(--green); font-weight: 700; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <span class="brand-icon">🛡️</span>
                <div>
                    <h1>Voice Clone <span>Shield</span></h1>
                    <p style="font-size: 0.8rem; color: var(--text-secondary);">Real-Time Voice Detection Engine • Member 1 Local Lab</p>
                </div>
            </div>
            <div class="badge">● LOCAL ENGINE ONLINE</div>
        </header>

        <!-- Test Controls Card -->
        <div class="card">
            <div class="section-title">Trigger Voice Detection</div>
            
            <div class="btn-group">
                <button class="btn btn-sample-human" onclick="runSample('samples/sample_real_human.wav')">
                    <span>🟢</span> Test Sample: Real Human
                </button>
                <button class="btn btn-sample-clone" onclick="runSample('samples/sample_ai_clone.wav')">
                    <span>🔴</span> Test Sample: AI Clone
                </button>
                <button class="btn btn-mic" id="mic-btn" onclick="toggleMicrophone()">
                    <span id="mic-icon">🎙️</span> <span id="mic-text">Record Microphone</span>
                </button>
            </div>

            <div class="upload-row">
                <input type="file" id="file-input" accept="audio/*">
                <button class="btn btn-submit" onclick="uploadAudioFile()">Analyze Custom File</button>
            </div>
        </div>

        <!-- Loading Spinner -->
        <div id="loading">
            <div class="spinner"></div>
            <div>Executing 16kHz resample, Mel-spectrogram & Wav2Vec2 inference...</div>
        </div>

        <!-- Results Card -->
        <div id="results-container">
            <div class="card">
                <div class="section-title">Detection Result</div>

                <div id="result-banner" class="result-banner">
                    <div>
                        <div class="verdict-title" id="verdict-title">DEEPFAKE DETECTED</div>
                        <div class="verdict-sub" id="verdict-sub">High confidence synthetic vocoder signature</div>
                    </div>
                    <div class="score-pill" id="score-pill">99.8%</div>
                </div>

                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-label">Latency</div>
                        <div class="metric-value" id="val-latency">1.82s</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Duration</div>
                        <div class="metric-value" id="val-duration">3.0s</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">F0 Pitch Mean</div>
                        <div class="metric-value" id="val-pitch">184 Hz</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Spectral Flatness</div>
                        <div class="metric-value" id="val-flatness">0.0084</div>
                    </div>
                </div>

                <div class="section-title" style="margin-top: 10px;">Generated Mel-Spectrogram (128 Bands)</div>
                <div class="spectrogram-box">
                    <img id="spectrogram-img" src="" alt="Mel-Spectrogram">
                </div>

                <div class="section-title">Windowed Segment Timeline</div>
                <table class="timeline-table">
                    <thead>
                        <tr>
                            <th>Time Window</th>
                            <th>Segment Spoof Score</th>
                            <th>Classification</th>
                        </tr>
                    </thead>
                    <tbody id="timeline-body">
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let isRecording = false;
        let audioContext = null;
        let mediaStream = null;
        let recorderNode = null;
        let recordedBuffers = [];

        // Universal 16-bit PCM WAV Encoder (compatible with Safari, Chrome, Firefox)
        function exportWAV(buffers, sampleRate) {
            let totalLength = 0;
            for (let b of buffers) totalLength += b.length;
            let merged = new Float32Array(totalLength);
            let offset = 0;
            for (let b of buffers) {
                merged.set(b, offset);
                offset += b.length;
            }

            const wavBuffer = new ArrayBuffer(44 + merged.length * 2);
            const view = new DataView(wavBuffer);

            function writeString(v, off, str) {
                for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i));
            }

            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + merged.length * 2, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM format
            view.setUint16(22, 1, true); // Mono
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true); // Byte rate
            view.setUint16(32, 2, true); // Block align
            view.setUint16(34, 16, true); // Bits per sample
            writeString(view, 36, 'data');
            view.setUint32(40, merged.length * 2, true);

            let dataOffset = 44;
            for (let i = 0; i < merged.length; i++) {
                let s = Math.max(-1, Math.min(1, merged[i]));
                view.setInt16(dataOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                dataOffset += 2;
            }
            return new Blob([view], { type: 'audio/wav' });
        }

        async function runSample(samplePath) {
            showLoading(true);
            try {
                const res = await fetch('/api/analyze-path?path=' + encodeURIComponent(samplePath));
                if (!res.ok) {
                    const err = await res.text();
                    throw new Error(err);
                }
                const data = await res.json();
                renderResults(data);
            } catch (err) {
                alert('Analysis failed: ' + err.message);
            } finally {
                showLoading(false);
            }
        }

        async function uploadAudioFile() {
            const input = document.getElementById('file-input');
            if (!input.files || input.files.length === 0) {
                alert('Please select an audio file first.');
                return;
            }
            showLoading(true);
            const formData = new FormData();
            formData.append('audio', input.files[0]);

            try {
                const res = await fetch('/api/analyze', { method: 'POST', body: formData });
                if (!res.ok) {
                    const err = await res.text();
                    throw new Error(err);
                }
                const data = await res.json();
                renderResults(data);
            } catch (err) {
                alert('Upload failed: ' + err.message);
            } finally {
                showLoading(false);
            }
        }

        async function toggleMicrophone() {
            const btn = document.getElementById('mic-btn');
            const icon = document.getElementById('mic-icon');
            const text = document.getElementById('mic-text');

            if (!isRecording) {
                try {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    mediaStream = await navigator.mediaDevices.getUserMedia({
                        audio: {
                            noiseSuppression: false,
                            echoCancellation: false,
                            autoGainControl: false,
                            sampleRate: 48000
                        }
                    });
                    const sourceNode = audioContext.createMediaStreamSource(mediaStream);

                    // Buffer size 4096, 1 input channel, 1 output channel
                    recorderNode = audioContext.createScriptProcessor(4096, 1, 1);
                    recordedBuffers = [];

                    recorderNode.onaudioprocess = e => {
                        if (!isRecording) return;
                        const channelData = e.inputBuffer.getChannelData(0);
                        recordedBuffers.push(new Float32Array(channelData));
                    };

                    sourceNode.connect(recorderNode);
                    recorderNode.connect(audioContext.destination);

                    isRecording = true;
                    btn.classList.add('recording');
                    text.innerText = 'Stop & Analyze';
                    icon.innerText = '⏹️';
                } catch (err) {
                    alert('Microphone access denied: ' + err.message);
                }
            } else {
                isRecording = false;
                btn.classList.remove('recording');
                text.innerText = 'Record Microphone';
                icon.innerText = '🎙️';

                if (recorderNode) recorderNode.disconnect();
                if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());

                const currentSR = audioContext.sampleRate;
                const wavBlob = exportWAV(recordedBuffers, currentSR);

                showLoading(true);
                const formData = new FormData();
                formData.append('audio', wavBlob, 'recording.wav');

                try {
                    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
                    if (!res.ok) {
                        const err = await res.text();
                        throw new Error(err);
                    }
                    const data = await res.json();
                    renderResults(data);
                } catch (err) {
                    alert('Mic analysis failed: ' + err.message);
                } finally {
                    showLoading(false);
                }
            }
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
            if (show) {
                document.getElementById('results-container').style.display = 'none';
            }
        }

        function renderResults(data) {
            // Fix 2: Use calibrated_score (blends neural 70% + forensics 30%) with raised threshold 0.75
            // This reduces false positives on real microphone recordings
            const calScore = data.calibrated_score !== undefined ? data.calibrated_score : data.ai_score;
            const isAI = calScore >= 0.65;
            const calPct = (calScore * 100).toFixed(1) + '%';
            const rawPct = (data.ai_score * 100).toFixed(1) + '%';
            
            const banner = document.getElementById('result-banner');
            const title = document.getElementById('verdict-title');
            const sub = document.getElementById('verdict-sub');
            const scorePill = document.getElementById('score-pill');

            // Show calibrated score prominently; raw score shown as secondary info
            scorePill.innerText = calPct;
            scorePill.title = 'Raw neural score: ' + rawPct + '  |  Calibrated (mic-adjusted): ' + calPct;

            if (isAI) {
                banner.className = 'result-banner banner-danger';
                title.innerText = '🚨 SYNTHETIC AI VOICE CLONE';
                sub.innerText = 'Neural + acoustic forensics both flag synthetic vocoder patterns (score ≥ 65%)';
            } else {
                banner.className = 'result-banner banner-safe';
                title.innerText = '✅ AUTHENTIC HUMAN SPEECH';
                sub.innerText = 'Natural pitch jitter, energy variation, and vocal tract resonance verified (score < 65%)';
            }

            // Show both raw and calibrated scores in metrics
            document.getElementById('val-latency').innerText = data.processing_time_sec + 's';
            document.getElementById('val-duration').innerText = (data.duration_sec || 0) + 's';
            document.getElementById('val-pitch').innerText = (data.forensics.pitch_f0_mean || 0) + ' Hz';
            document.getElementById('val-flatness').innerText = data.forensics.spectral_flatness || 0;

            // Inject raw vs calibrated score detail below metrics if element exists
            const rawScoreEl = document.getElementById('val-raw-score');
            if (rawScoreEl) rawScoreEl.innerText = rawPct + ' → ' + calPct + ' (calibrated)';

            const img = document.getElementById('spectrogram-img');
            img.src = '/' + data.spectrogram_path + '?t=' + Date.now();

            const tbody = document.getElementById('timeline-body');
            tbody.innerHTML = '';
            (data.segments || []).forEach(seg => {
                const tr = document.createElement('tr');
                const segPct = (seg.ai_score * 100).toFixed(1) + '%';
                // Segment threshold aligned with main threshold 0.60
                const tagClass = seg.ai_score >= 0.65 ? 'tag-fake' : 'tag-safe';
                const tagLabel = seg.ai_score >= 0.65 ? 'AI CLONE' : 'HUMAN';
                tr.innerHTML = `
                    <td>${seg.start.toFixed(1)}s – ${seg.end.toFixed(1)}s</td>
                    <td class="${tagClass}">${segPct}</td>
                    <td class="${tagClass}">${tagLabel}</td>
                `;
                tbody.appendChild(tr);
            });

            document.getElementById('results-container').style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

@app.post("/api/analyze")
async def analyze_uploaded_audio(audio: UploadFile = File(...)):
    contents = await audio.read()
    try:
        result = analyze_voice(contents, clip_id=f"upload_{int(time.time()*1000)}")
        return JSONResponse(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/api/analyze-path")
def analyze_path(path: str):
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": f"Path not found: {path}"})
    try:
        result = analyze_voice(path, clip_id=f"sample_{int(time.time()*1000)}")
        return JSONResponse(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, log_level="info")

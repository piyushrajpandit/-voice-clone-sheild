---
title: Voice Clone Shield
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time AI voice clone detector & safe-word.
---

# 🛡️ Voice Clone Shield

> **Real-time AI voice clone detector & automated Family Safe-Word challenge protocol for active voice and video call security.**

## 💡 Overview
With consumer generative voice cloning tools (like ElevenLabs, Play.ht, and open-source TTS models), bad actors can clone a loved one's voice with just 5 seconds of social media audio. **Voice Clone Shield** acts as a real-time cybersecurity sentinel that analyzes incoming voice/video audio streams and detects synthetic vocoder artifacts, frequency cutoffs, and acoustic anomalies.

## 🚀 Key Features
- **Acoustic Deepfake Detection**: Powered by fine-tuned Wav2Vec2 transformer models trained on speech anti-spoofing benchmarks.
- **Real-Time Video Call Intercept**: Live webcam & microphone stream intercept with facial track overlay.
- **Studio-Grade 16kHz PCM Capture**: Direct browser Web Audio API capture to avoid lossy WebM compression artifacts.
- **Family Safe-Word Emergency Protocol**: Triggers instant lockdown when spoof risk exceeds safety thresholds, requiring pre-shared secret verification before sensitive actions (e.g. wire transfers) can occur.
- **Live Spectral Density Matrix**: Interactive Mel-spectrogram visualization exposing high-frequency cutoff boundaries.

## 🛠️ Stack
- **Framework**: Python FastAPI
- **Audio Processing**: SciPy, SoundFile, NumPy
- **Neural Model**: `mo-thecreator/Deepfake-audio-detection` (Hugging Face Transformers)
- **Frontend**: Vanilla HTML5, CSS3 Glassmorphism, Web Audio API

---
title: Voice Clone Shield
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-time AI voice clone detector with acoustic forensics.
---

# 🛡️ Voice Clone Shield

> **Real-time AI voice clone detector combining deep neural transformer detection (Wav2Vec2) and physics-based acoustic forensics (pitch jitter, spectral flatness, energy regularity, and zero-crossing dynamics).**

## 🚀 Features
- **Wav2Vec2 Deepfake Audio Detection**: Powered by `mo-thecreator/Deepfake-audio-detection`.
- **Physics-Based Acoustic Forensics**:
  - Voice Activity Detection (VAD) frame filtering.
  - Micro-pitch tremor (F0 jitter) analysis.
  - Spectral flatness & centroid balance.
  - Voiced energy envelope regularity (RMS CV) & ZCR consistency to catch clean TTS voices (ChatGPT, ElevenLabs).
- **Calibrated Scoring Engine**: Mitigates browser microphone false-positives while maintaining high recall on synthetic speech.
- **Interactive Web Interface**: Live browser microphone recording, sample test buttons, spectrogram visualization, and segment timeline breakdown.

## 🛠️ Local Development
```bash
git clone https://github.com/piyushrajpandit/-voice-clone-sheild.git
cd -voice-clone-sheild
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```
Open [http://localhost:5050](http://localhost:5050) in your browser.

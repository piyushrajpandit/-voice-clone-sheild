"""
audio_utils.py - Audio processing utilities for Voice Clone Shield.
Handles conversion to 16 kHz mono WAV, segment cutting, and spectrogram generation.
"""
from typing import List, Dict, Any, Tuple, Union
import io
import os
import numpy as np
import soundfile as sf
import scipy.signal

def load_and_resample(
    audio_input: Union[str, bytes],
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file path or raw bytes, convert to mono float32 array,
    normalize peak amplitude to 0.95, remove DC offset, trim leading/trailing silence,
    and resample to target_sr (default 16,000 Hz).

    Returns:
        Tuple of (audio_array: np.ndarray, sample_rate: int)
    """
    # 1. Read audio data and sample rate
    if isinstance(audio_input, bytes):
        try:
            data, sr = sf.read(io.BytesIO(audio_input), dtype="float32")
        except Exception:
            import wave
            with wave.open(io.BytesIO(audio_input), 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input}")
        data, sr = sf.read(audio_input, dtype="float32")
    else:
        raise TypeError(f"audio_input must be a file path (str) or bytes, got {type(audio_input)}")

    # 2. Convert to mono if multi-channel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 3. Resample to target_sr if necessary
    if sr != target_sr:
        num_target_samples = int(round(len(data) * float(target_sr) / sr))
        data = scipy.signal.resample(data, num_target_samples)
        sr = target_sr

    # 4. Remove DC offset
    data = data - np.mean(data)

    # 5. Peak normalize to 0.95
    peak = np.max(np.abs(data))
    if peak > 1e-4:
        data = (data / peak) * 0.95

    # 6. Energy-based silence trimming (leading & trailing)
    frame_len = int(target_sr * 0.05)  # 50ms frames
    if len(data) > frame_len:
        num_frames = len(data) // frame_len
        energies = [
            float(np.sum(data[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(num_frames)
        ]
        thresh = float(np.mean(energies) * 0.08) if energies else 0.0

        start_idx = 0
        for i, e in enumerate(energies):
            if e > thresh:
                start_idx = i * frame_len
                break

        end_idx = len(data)
        for i in range(len(energies) - 1, -1, -1):
            if energies[i] > thresh:
                end_idx = min(len(data), (i + 1) * frame_len)
                break

        trimmed = data[start_idx:end_idx]
        if len(trimmed) >= int(target_sr * 0.4):
            data = trimmed

    return data.astype(np.float32), sr

def cut_segments(
    audio_array: np.ndarray,
    sr: int = 16000,
    segment_duration: float = 2.0,
    hop_duration: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Slice audio array into overlapping window segments with start and end timestamps.

    Args:
        audio_array: 1D float32 numpy array.
        sr: Sampling rate (default 16000).
        segment_duration: Duration of each slice in seconds (default 2.0s).
        hop_duration: Hop/stride between slice starts in seconds (default 1.0s).

    Returns:
        List of dicts: [{'start': float, 'end': float, 'audio': np.ndarray}]
    """
    total_samples = len(audio_array)
    segment_samples = int(round(segment_duration * sr))
    hop_samples = int(round(hop_duration * sr))

    if total_samples <= segment_samples:
        return [{
            "start": 0.0,
            "end": round(float(total_samples) / sr, 2),
            "audio": audio_array
        }]

    segments = []
    start_sample = 0
    while start_sample < total_samples:
        end_sample = min(start_sample + segment_samples, total_samples)
        chunk = audio_array[start_sample:end_sample]

        if len(chunk) < int(segment_samples * 0.5):
            break

        segments.append({
            "start": round(float(start_sample) / sr, 2),
            "end": round(float(end_sample) / sr, 2),
            "audio": chunk
        })

        if end_sample == total_samples:
            break
        start_sample += hop_samples

    return segments

def generate_spectrogram(
    audio_array: np.ndarray,
    sr: int = 16000,
    output_path: str = "outputs/spectrogram.png"
) -> str:
    """
    Compute Mel-spectrogram with librosa and save a PNG image to output_path.
    Highlights high-frequency vocoder cutoff regions for forensic analysis.

    Returns:
        The string path of the generated PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 1. Compute Mel-spectrogram (128 bands)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sr,
        n_mels=128,
        fmax=sr // 2,
        hop_length=int(sr * 0.01),  # 10ms hop
        n_fft=int(sr * 0.025)       # 25ms window
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 2. Render plot
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=100)
    img = librosa.display.specshow(
        mel_db,
        x_axis="time",
        y_axis="mel",
        sr=sr,
        fmax=sr // 2,
        ax=ax,
        cmap="magma"
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    # Add forensic annotation for high-frequency vocoder cutoff line
    ax.axhline(y=8000, color="cyan", linestyle="--", linewidth=1.2, alpha=0.7, label="Nyquist Limit (8kHz)")
    ax.set_title("Voice Clone Shield: Acoustic Mel-Spectrogram (16kHz)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return output_path

"""
app/phone_audio.py
Member 2 - Phone-quality audio simulation.

Pipeline: 16kHz mono -> downsample to 8kHz (telephone band) ->
compress dynamic range -> add light noise -> upsample back to 16kHz mono.
"""

import numpy as np
import soundfile as sf
from scipy import signal


def _downsample(audio, orig_sr, target_sr):
    num_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples)


def _upsample(audio, orig_sr, target_sr):
    num_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples)


def _compress_dynamic_range(audio, threshold=0.3, ratio=4.0):
    compressed = np.copy(audio)
    above = np.abs(compressed) > threshold
    sign = np.sign(compressed[above])
    excess = np.abs(compressed[above]) - threshold
    compressed[above] = sign * (threshold + excess / ratio)
    return compressed


def _add_light_noise(audio, noise_level=0.003):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


def make_phone_quality_copy(wav_path, output_path=None):
    try:
        audio, sr = sf.read(wav_path)
    except Exception:
        return None

    try:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != 16000:
            audio = _downsample(audio, sr, 16000)
            sr = 16000

        narrow = _downsample(audio, sr, 8000)
        narrow = _compress_dynamic_range(narrow)
        narrow = _add_light_noise(narrow)

        wide = _upsample(narrow, 8000, 16000)
        wide = np.clip(wide, -1.0, 1.0)

        if output_path is None:
            output_path = wav_path.replace(".wav", "_phone.wav")

        sf.write(output_path, wide, 16000, subtype="PCM_16")
        return output_path

    except Exception:
        return None

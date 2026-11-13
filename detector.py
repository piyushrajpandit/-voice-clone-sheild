"""
detector.py - Pre-trained Wav2Vec2 Deepfake Audio Detector wrapper.
Model: mo-thecreator/Deepfake-audio-detection
Normalizes output so 0.0 = Real/Human, 1.0 = AI Cloned.
"""
from typing import List, Dict, Any, Optional
import math
import numpy as np

MODEL_ID = "mo-thecreator/Deepfake-audio-detection"

class DeepfakeDetector:
    def __init__(self, model_id: str = MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._classifier = None

    def _get_classifier(self):
        """Lazy loader for huggingface audio-classification pipeline."""
        if self._classifier is None:
            from transformers import pipeline
            kwargs = {}
            if self.device is not None:
                kwargs["device"] = self.device
            self._classifier = pipeline("audio-classification", model=self.model_id, **kwargs)
        return self._classifier

    def score_clip(self, audio_array: np.ndarray, sr: int = 16000) -> float:
        """
        Score a single 1D audio clip array.

        Returns:
            float in [0.0, 1.0] where 0.0 = completely human, 1.0 = completely AI fake.
        """
        if len(audio_array) == 0:
            return 0.0

        # Ensure float32 1D
        audio_data = audio_array.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        classifier = self._get_classifier()
        preds = classifier({"raw": audio_data, "sampling_rate": sr})

        scores = {p["label"].lower(): float(p["score"]) for p in preds}
        if "fake" in scores:
            ai_score = scores["fake"]
        elif "spoof" in scores:
            ai_score = scores["spoof"]
        elif "real" in scores:
            ai_score = 1.0 - scores["real"]
        else:
            ai_score = 0.5

        return round(float(np.clip(ai_score, 0.0, 1.0)), 4)

    def score_segments(
        self,
        segments: List[Dict[str, Any]],
        sr: int = 16000
    ) -> List[Dict[str, Any]]:
        """
        Score every segment and append 'ai_score'.

        Args:
            segments: List of dicts, each containing 'start', 'end', 'audio'.
            sr: Sampling rate.

        Returns:
            List of dicts: [{'start': float, 'end': float, 'ai_score': float}]
        """
        scored_segments = []
        for seg in segments:
            score = self.score_clip(seg["audio"], sr=sr)
            scored_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "ai_score": score
            })
        return scored_segments

    @staticmethod
    def calculate_clip_score(segment_scores: List[float]) -> float:
        """
        Calculate clip-level ai_score using top-half average:
        sort descending, take the top ceil(N/2) scores, and average them.
        This ensures even a localized 3-second spliced fake strongly raises the score.
        """
        if not segment_scores:
            return 0.0

        sorted_scores = sorted(segment_scores, reverse=True)
        top_k = max(1, math.ceil(len(sorted_scores) / 2))
        top_scores = sorted_scores[:top_k]
        return round(float(np.mean(top_scores)), 4)

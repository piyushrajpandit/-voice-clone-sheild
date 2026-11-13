"""
app/scam.py
Member 2 - Scam language detection module.

SCAM SCORE RULE (plain language):
- We look for known scam phrases in the transcript, grouped into categories:
  urgency, otp_request, money_transfer, secrecy, authority_threat.
- Each category found adds a fixed weight to the score (only once per
  category, even if several phrases in that category match):
      otp_request       -> 0.35   (asking for OTP/codes - most dangerous)
      money_transfer     -> 0.30   (asking to send/transfer money)
      authority_threat    -> 0.25   (claims of account/card/legal action)
      secrecy             -> 0.25   (asking to keep the call secret)
      urgency              -> 0.15   (time-pressure language)
- scam_score = sum of weights for every category matched, capped at 1.0.
- No matches -> scam_score is 0.0.
- Phrase start/end times come from matching Whisper's word-level
  timestamps against each word in the phrase (not just segment-level).
"""

import json
import os
import re
import whisper

_WHISPER_MODEL = None
_SCAM_PHRASES = None

_PHRASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scam_phrases.json")

CATEGORY_WEIGHTS = {
    "otp_request": 0.35,
    "money_transfer": 0.30,
    "authority_threat": 0.25,
    "secrecy": 0.25,
    "urgency": 0.15,
}


def _load_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("small")
    return _WHISPER_MODEL


def _load_phrases():
    global _SCAM_PHRASES
    if _SCAM_PHRASES is None:
        with open(_PHRASES_PATH, "r", encoding="utf-8-sig") as f:
            _SCAM_PHRASES = json.load(f)
    return _SCAM_PHRASES


def _normalize_word(w):
    return re.sub(r"[^\w]", "", w.strip().lower())


def _get_all_words(segments):
    """
    Flattens every segment's word list into one list of
    (normalized_word, start, end) tuples, in order.
    Falls back to segment-level if word timestamps are missing.
    """
    words = []
    for seg in segments:
        seg_words = seg.get("words")
        if seg_words:
            for w in seg_words:
                norm = _normalize_word(w.get("word", ""))
                if norm:
                    words.append((norm, w.get("start", seg.get("start", 0.0)),
                                  w.get("end", seg.get("end", 0.0))))
        else:
            # fallback: treat whole segment as one "word"
            norm = _normalize_word(seg.get("text", ""))
            if norm:
                words.append((norm, seg.get("start", 0.0), seg.get("end", 0.0)))
    return words


def _find_phrase_in_words(phrase, all_words):
    """
    Looks for the sequence of words in `phrase` inside all_words.
    Returns (start, end) as plain floats if found, else None.
    """
    phrase_words = [_normalize_word(w) for w in phrase.split() if _normalize_word(w)]
    if not phrase_words:
        return None

    n = len(phrase_words)
    for i in range(len(all_words) - n + 1):
        window = [all_words[i + j][0] for j in range(n)]
        if window == phrase_words:
            start = all_words[i][1]
            end = all_words[i + n - 1][2]
            return float(start), float(end)
    return None


def analyze_scam(wav_path):
    safe_default = {
        "transcript": "",
        "scam_score": 0.0,
        "flags": [],
        "error": None,
    }

    try:
        model = _load_whisper()
        phrases_by_category = _load_phrases()
    except Exception as e:
        safe_default["error"] = f"Failed to load model or phrases: {e}"
        return safe_default

    try:
        result = model.transcribe(wav_path, word_timestamps=True)
    except Exception as e:
        safe_default["error"] = f"Transcription failed: {e}"
        return safe_default

    transcript = result.get("text", "").strip()
    segments = result.get("segments", [])
    all_words = _get_all_words(segments)

    flags = []
    categories_matched = set()

    for category, phrase_list in phrases_by_category.items():
        for phrase in phrase_list:
            match = _find_phrase_in_words(phrase, all_words)
            if match:
                start, end = match
                flags.append({
                    "phrase": phrase,
                    "category": category,
                    "start": round(start, 2),
                    "end": round(end, 2),
                })
                categories_matched.add(category)

    scam_score = 0.0
    for category in categories_matched:
        scam_score += CATEGORY_WEIGHTS.get(category, 0.0)
    scam_score = float(min(1.0, round(scam_score, 4)))

    return {
        "transcript": transcript,
        "scam_score": scam_score,
        "flags": flags,
        "error": None,
    }

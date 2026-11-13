# app/risk.py  (Member 2)
# Rule: weights are ai 0.40, scam 0.25, match 0.20, challenge 0.15.
# If a signal is missing, its weight is shared equally among the signals
# that exist, so the total is always 1.0.
# The safe-word shares the challenge slot: the HIGHER of challenge_risk and
# safe_word_risk is used. LOW < 0.35, MEDIUM 0.35 to 0.65, HIGH > 0.65.

WEIGHTS = {"ai": 0.40, "scam": 0.25, "match": 0.20, "challenge": 0.15}


def _num(value, default=0.5):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def combine_risk(detect, scam, match=None, challenge=None, safeword=None):
    try:
        detect = detect or {}
        scam = scam or {}
        values = {}
        reasons = []

        # 1. AI voice score
        values["ai"] = _num(detect.get("ai_score"))
        segments = detect.get("segments") or []
        if segments:
            fake = sum(1 for s in segments if _num(s.get("ai_score"), 0.0) > 0.5)
            if fake:
                reasons.append(f"Voice looks AI-made in {fake} of {len(segments)} segments")
        elif values["ai"] > 0.5:
            reasons.append("Voice looks AI-made")

        # 2. Scam language
        values["scam"] = _num(scam.get("scam_score"))
        cats = sorted({f.get("category") for f in (scam.get("flags") or []) if f.get("category")})
        if cats:
            reasons.append("Scam language found: " + ", ".join(cats))

        # 3. Family match (missing if nobody is enrolled)
        if match and match.get("best_match"):
            values["match"] = _num(match.get("match_risk"))
            if values["match"] > 0.5:
                reasons.append("Voice does not match any enrolled family member")

        # 4. Challenge slot (challenge phrase and/or safe-word)
        slot = []
        if challenge:
            slot.append(_num(challenge.get("challenge_risk")))
            if not challenge.get("passed", False):
                reasons.append("Caller failed the challenge phrase")
        if safeword and safeword.get("safe_word_set", False):
            slot.append(_num(safeword.get("safe_word_risk")))
            if not safeword.get("passed", False):
                reasons.append("Caller did not say the family safe-word")
        if slot:
            values["challenge"] = max(slot)

        # Share the weight of missing signals equally
        missing = sum(w for k, w in WEIGHTS.items() if k not in values)
        extra = missing / len(values)
        score = sum(values[k] * (WEIGHTS[k] + extra) for k in values)
        score = round(min(1.0, max(0.0, score)), 3)

        if score < 0.35:
            level = "LOW"
        elif score <= 0.65:
            level = "MEDIUM"
        else:
            level = "HIGH"
        if not reasons:
            reasons.append("No strong warning signs found")
        return {"final_score": score, "level": level, "reasons": reasons}
    except Exception:
        return {"final_score": 0.5, "level": "MEDIUM",
                "reasons": ["Could not calculate risk, please try again"]}

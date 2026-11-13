"""
scam_detector.py - Real-Time Multilingual NLP Scam & Fraud Detection Engine
Detects fraud intents, emergency extortion, banking credential theft, authority impersonation,
and psychological coercion across multiple languages.
"""

import re
from typing import Dict, List, Any

# Multilingual scam keyword bank categorized by fraud typology
SCAM_KNOWLEDGE_BASE = {
    "EMERGENCY_EXTORTION": {
        "label": "Emergency & Bail Extortion",
        "description": "Call claims a family member or friend is in danger, arrested, or injured and demands immediate money.",
        "weight": 40,
        "patterns": {
            "en": [
                r"\barrested\b", r"\bbail\b", r"\bbail bondsman\b", r"\bcar accident\b",
                r"\bpolice station\b", r"\bjail\b", r"\bhospital emergency\b",
                r"\bwire (\$\d+|\d+ dollars|money|cash)\b", r"\bwire transfer\b",
                r"\bsend money immediately\b", r"\bransom\b", r"\bkidnapped\b",
                r"\blawyer fees\b", r"\bbattery is dying\b"
            ],
            "hi": [
                r"गिरफ्तार", r"एक्सीडेंट", r"दुर्घटना", r"पुलिस स्टेशन", r"जेल",
                r"अस्पताल", r"किडनैप", r"फिरौती", r"पैसे भेजो", r"पैसे ट्रांसफर",
                r"जमानत", r"बेल"
            ],
            "regional": [
                r"\baccident\b", r"\bpolice\b", r"\bjail\b", r"\bhospital\b", r"\bmoney\b"
            ]
        }
    },
    "BANKING_CREDENTIAL_THEFT": {
        "label": "Banking & OTP Theft",
        "description": "Call attempts to trick victim into revealing OTPs, passwords, or banking credentials.",
        "weight": 35,
        "patterns": {
            "en": [
                r"\botp\b", r"\bone[\s-]time password\b", r"\bcvv\b", r"\bpin code\b",
                r"\bnet banking\b", r"\baccount (blocked|suspended|frozen)\b",
                r"\bdebit card expired\b", r"\bcredit card (blocked|verification)\b",
                r"\bupdate kyc\b", r"\bkyc verification\b", r"\bverify your bank\b",
                r"\bbank manager\b", r"\bshare the 6[\s-]digit code\b"
            ],
            "hi": [
                r"ओटीपी", r"पिन", r"बैंक खाता बंद", r"केवाईसी", r"एटीएम ब्लॉक",
                r"खाता सस्पेंड", r"वेरिफिकेशन कोड", r"पासवर्ड बताओ"
            ],
            "regional": [
                r"\botp\b", r"\bpassword\b", r"\bkyc\b", r"\baccount blocked\b"
            ]
        }
    },
    "AUTHORITY_IMPERSONATION": {
        "label": "Government & Police Impersonation",
        "description": "Scammer impersonates law enforcement, tax officials (IRS/CBI), or customs.",
        "weight": 35,
        "patterns": {
            "en": [
                r"\binternal revenue service\b", r"\birs\b", r"\bcbi\b", r"\bfbi\b",
                r"\bpolice officer\b", r"\bcustoms department\b", r"\barrest warrant\b",
                r"\blegal action\b", r"\bcourt summons\b", r"\btax evasion\b",
                r"\billegal parcel\b", r"\bdrugs found\b"
            ],
            "hi": [
                r"पुलिस अधिकारी", r"सीबीआई", r"कस्टम विभाग", r"कोर्ट नोटिस",
                r"गिरफ्तारी वारंट", r"गैरकानूनी पार्सल", r"ड्रग्स मिला है"
            ],
            "regional": [
                r"\bcbi\b", r"\bpolice\b", r"\bcourt\b", r"\bwarrant\b", r"\bcustoms\b"
            ]
        }
    },
    "LOTTERY_PRIZE_FRAUD": {
        "label": "Lottery & Gift Card Fraud",
        "description": "Offers fake sweepstakes, prizes, or asks for payment in non-traceable gift cards.",
        "weight": 25,
        "patterns": {
            "en": [
                r"\byou('ve| have) won\b", r"\blottery prize\b", r"\bcash prize\b",
                r"\bprocessing fee\b", r"\bgift card(s)?\b", r"\bapple gift card\b",
                r"\btarget gift card\b", r"\blucky draw\b", r"\bclaim your reward\b"
            ],
            "hi": [
                r"लॉटरी जीती", r"इनाम मिला है", r"लकी ड्रा", r"प्रोसेसिंग फीस",
                r"गिफ्ट कार्ड", r"करोड़ रुपये जीते"
            ],
            "regional": [
                r"\blottery\b", r"\bprize\b", r"\bgift card\b", r"\bwinner\b"
            ]
        }
    },
    "TECH_SUPPORT_SCAM": {
        "label": "Remote Access Tech Scam",
        "description": "Claims victim's device is compromised and asks to install remote desktop tools.",
        "weight": 30,
        "patterns": {
            "en": [
                r"\bcomputer (has a virus|is infected|is compromised)\b",
                r"\bteamviewer\b", r"\banydesk\b", r"\bultraviewer\b", r"\bremote desktop\b",
                r"\bmicrosoft support\b", r"\bapple tech support\b", r"\bdownload this software\b"
            ],
            "hi": [
                r"कंप्यूटर में वायरस", r"एनीडेस्क डाउनलोड", r"रिमोट एक्सेस", r"सिस्टम हैक"
            ],
            "regional": [
                r"\banydesk\b", r"\bteamviewer\b", r"\bvirus\b", r"\bremote access\b"
            ]
        }
    }
}

# Psychological urgency and pressure tactics (cross-cutting multipliers)
URGENCY_PATTERNS = {
    "en": [
        r"\bimmediately\b", r"\bright now\b", r"\bwithin \d+ (minutes|hours)\b",
        r"\bdon't hang up\b", r"\bstay on the line\b", r"\bdon't tell anyone\b",
        r"\bkeep (this|it) secret\b", r"\burgen(t|cy)\b", r"\bhasten\b",
        r"\bbefore it's too late\b", r"\blast warning\b"
    ],
    "hi": [
        r"तुरंत", r"अभी के अभी", r"फोन मत काटना", r"लाइन पर रहो",
        r"किसी को मत बताना", r"जल्दी करो", r"आखरी चेतावनी"
    ]
}

def analyze_transcript(transcript: str, language: str = "en") -> Dict[str, Any]:
    """
    Analyzes spoken transcript using NLP pattern matching and heuristic scoring.
    Returns comprehensive scam assessment with risk score and actionable advice.
    """
    if not transcript or transcript.strip() in ["", "(No spoken words detected)", "(Transcription unavailable)"]:
        return {
            "is_scam": False,
            "scam_score": 0.0,
            "risk_level": "SAFE",
            "detected_category": None,
            "matched_keywords": [],
            "urgency_markers": [],
            "explanation": "No actionable speech detected.",
            "recommended_action": "Safe to proceed."
        }

    text_lower = transcript.lower()
    matched_categories = []
    matched_keywords = []
    total_score = 0.0

    # 1. Match Fraud Typology Patterns
    for cat_id, cat_info in SCAM_KNOWLEDGE_BASE.items():
        cat_hits = []
        # Check patterns for detected language + fallback regional/en
        lang_patterns = cat_info["patterns"].get(language, [])
        generic_patterns = cat_info["patterns"].get("en", []) + cat_info["patterns"].get("regional", [])
        patterns_to_check = set(lang_patterns + generic_patterns)

        for pat in patterns_to_check:
            found = re.findall(pat, text_lower, re.IGNORECASE)
            if found:
                # Capture clean keyword representations
                clean_term = pat.replace(r"\b", "").replace(r"\s-", " ")
                clean_term = re.sub(r"[\\()^$?*+]", "", clean_term).strip()
                cat_hits.append(clean_term)

        if cat_hits:
            cat_score = cat_info["weight"] * min(len(cat_hits), 2)
            total_score += cat_score
            matched_categories.append({
                "id": cat_id,
                "label": cat_info["label"],
                "hits": list(set(cat_hits))
            })
            matched_keywords.extend(list(set(cat_hits)))

    # 2. Check Urgency / Psychological Coercion
    matched_urgency = []
    urgency_pats = URGENCY_PATTERNS.get(language, []) + URGENCY_PATTERNS.get("en", [])
    for pat in set(urgency_pats):
        if re.search(pat, text_lower, re.IGNORECASE):
            clean_term = pat.replace(r"\b", "").strip()
            matched_urgency.append(clean_term)

    # Urgency acts as an escalating multiplier
    if matched_urgency:
        urgency_multiplier = 1.0 + (min(len(matched_urgency), 3) * 0.15)
        total_score *= urgency_multiplier

    # High-Risk Combo Multiplier: Emergency + Wire Money demand
    has_emergency = any(c["id"] == "EMERGENCY_EXTORTION" for c in matched_categories)
    has_finance = any(c["id"] in ["BANKING_CREDENTIAL_THEFT", "LOTTERY_PRIZE_FRAUD"] for c in matched_categories)
    if has_emergency and (has_finance or len(matched_urgency) > 0):
        total_score += 25.0

    # Normalize score 0 - 100
    scam_score = min(round(total_score, 1), 100.0)

    # Determine Risk Tier
    if scam_score >= 70.0:
        risk_level = "CRITICAL"
        is_scam = True
    elif scam_score >= 45.0:
        risk_level = "HIGH"
        is_scam = True
    elif scam_score >= 25.0:
        risk_level = "MEDIUM"
        is_scam = False
    elif scam_score > 0.0:
        risk_level = "LOW"
        is_scam = False
    else:
        risk_level = "SAFE"
        is_scam = False

    # Primary detected category
    primary_cat = matched_categories[0]["label"] if matched_categories else None

    # Generate Human-Friendly Explanation & Advisory
    if is_scam:
        explanation = f"Flagged as {primary_cat}. Detected high-risk intent markers ({', '.join(matched_keywords[:4])})."
        if matched_urgency:
            explanation += f" Coercive urgency detected ({', '.join(matched_urgency[:2])})."
        recommended_action = "🚨 DO NOT send money, share OTPs, or install software. Hang up immediately and verify caller independently."
    elif risk_level == "MEDIUM":
        explanation = f"Moderate suspicion: Detected trigger words ({', '.join(matched_keywords[:3])}) but context is inconclusive."
        recommended_action = "⚠️ Exercise caution. Ask the caller to provide family safe-word or call back on a verified number."
    else:
        explanation = "Conversational pattern appears normal. No scam or extortion vectors detected."
        recommended_action = "Standard conversation verified."

    return {
        "is_scam": is_scam,
        "scam_score": scam_score,
        "risk_level": risk_level,
        "primary_category": primary_cat,
        "categories": matched_categories,
        "matched_keywords": list(dict.fromkeys(matched_keywords)),
        "urgency_markers": list(dict.fromkeys(matched_urgency)),
        "explanation": explanation,
        "recommended_action": recommended_action
    }

"""
sms_service.py - Automated Twilio SMS Security Alert Service
Dispatches instant emergency SMS alerts when high-risk scam intents,
emergency extortion, or deepfake audio signatures are detected.
Includes deduplication/cooldown guards to avoid SMS spamming during live call streaming.
"""

import os
import re
import time
from typing import Dict, Any, Optional

# Simple local .env file reader
ENV_FILE_PATH = os.path.join(os.path.dirname(__file__), ".env")

def _read_env_file():
    if not os.path.exists(ENV_FILE_PATH):
        return
    try:
        with open(ENV_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        print(f"[SMS Service] Warning reading .env file: {e}")

_read_env_file()

# In-memory configuration cache with environment defaults
_CONFIG = {
    "account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
    "auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
    "from_number": os.getenv("TWILIO_PHONE_NUMBER", ""),
    "emergency_contact": os.getenv("EMERGENCY_CONTACT_PHONE", ""),
    "enabled": os.getenv("SMS_ALERTS_ENABLED", "true").lower() in ["1", "true", "yes"]
}

# Alert deduplication cooldown per call session (call_id -> timestamp)
_ALERT_COOLDOWN_CACHE: Dict[str, float] = {}
COOLDOWN_SECONDS = 300  # 5 minutes per call session

# In-memory audit of dispatched alerts
_DISPATCH_LOGS = []

def get_sms_config() -> Dict[str, Any]:
    """Returns safe view of SMS configuration without exposing private auth tokens."""
    sid = _CONFIG["account_sid"] or os.getenv("TWILIO_ACCOUNT_SID", "")
    token = _CONFIG["auth_token"] or os.getenv("TWILIO_AUTH_TOKEN", "")
    from_num = _CONFIG["from_number"] or os.getenv("TWILIO_PHONE_NUMBER", "")
    emergency = _CONFIG["emergency_contact"] or os.getenv("EMERGENCY_CONTACT_PHONE", "")
    
    is_configured = bool(sid and token and from_num and emergency)
    
    return {
        "is_configured": is_configured,
        "enabled": _CONFIG["enabled"],
        "account_sid_masked": f"{sid[:4]}...{sid[-4:]}" if len(sid) > 8 else ("Configured" if sid else "Not set"),
        "from_number": from_num or "Not set",
        "emergency_contact": emergency or "Not set",
        "has_auth_token": bool(token),
        "cooldown_seconds": COOLDOWN_SECONDS,
        "recent_alerts_count": len(_DISPATCH_LOGS)
    }

def update_sms_config(
    account_sid: Optional[str] = None,
    auth_token: Optional[str] = None,
    from_number: Optional[str] = None,
    emergency_contact: Optional[str] = None,
    enabled: Optional[bool] = None
) -> Dict[str, Any]:
    """Dynamically updates SMS settings at runtime."""
    if account_sid is not None:
        _CONFIG["account_sid"] = account_sid.strip()
        os.environ["TWILIO_ACCOUNT_SID"] = _CONFIG["account_sid"]
    if auth_token is not None:
        _CONFIG["auth_token"] = auth_token.strip()
        os.environ["TWILIO_AUTH_TOKEN"] = _CONFIG["auth_token"]
    if from_number is not None:
        _CONFIG["from_number"] = from_number.strip()
        os.environ["TWILIO_PHONE_NUMBER"] = _CONFIG["from_number"]
    if emergency_contact is not None:
        _CONFIG["emergency_contact"] = emergency_contact.strip()
        os.environ["EMERGENCY_CONTACT_PHONE"] = _CONFIG["emergency_contact"]
    if enabled is not None:
        _CONFIG["enabled"] = bool(enabled)
        os.environ["SMS_ALERTS_ENABLED"] = "true" if _CONFIG["enabled"] else "false"

    return get_sms_config()

def format_alert_message(scam_analysis: Dict[str, Any], is_fake_voice: bool, call_id: Optional[str] = None) -> str:
    """
    Creates a concise, high-urgency SMS notification body (under 115 chars).
    Strictly GSM-7 encoded (no emojis) to prevent UCS-2 compression and comply
    with Twilio Trial 160-char total limit (which includes Twilio's 38-char prefix).
    """
    category = scam_analysis.get("primary_category") or "Scam"
    # Shorten common categories for SMS brevity
    category_short = (
        category.replace("Emergency & Bail Extortion", "Bail Scam")
        .replace("Banking & OTP Theft", "Bank/OTP Theft")
        .replace("Government & Police Impersonation", "Police Impersonation")
        .replace("Remote Access Tech Scam", "Tech Support Scam")
        .replace("Lottery & Gift Card Fraud", "Lottery Fraud")
    )
    if len(category_short) > 20:
        category_short = category_short[:20]

    if is_fake_voice and scam_analysis.get("is_scam"):
        header = "[VOICE SHIELD] CRITICAL: AI Clone + Scam!"
    elif is_fake_voice:
        header = "[VOICE SHIELD] ALERT: AI Deepfake Call!"
    else:
        header = "[VOICE SHIELD] ALERT: Suspected Scam Call!"

    # Template: "[HEADER] Type: {cat}. Hang up now! DO NOT send money or OTP."
    body = f"{header} Type: {category_short}. Hang up now! Do NOT send money or OTP."
    
    # Strict clamp to 115 chars max for Twilio Trial compatibility
    if len(body) > 115:
        body = body[:115]
    return body

def send_scam_alert_sms(
    call_id: Optional[str],
    scam_analysis: Dict[str, Any],
    is_fake_voice: bool,
    target_phone: Optional[str] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Dispatches automated SMS when fraud/scam conditions are met.
    Respects session cooldown to avoid spamming the user on repeated 4s audio chunks.
    """
    if not _CONFIG["enabled"]:
        return {"sent": False, "reason": "SMS alerts globally disabled"}

    call_key = str(call_id or "live-active-stream")
    now = time.time()

    # 1. Cooldown & Deduplication check
    if not force and call_key in _ALERT_COOLDOWN_CACHE:
        elapsed = now - _ALERT_COOLDOWN_CACHE[call_key]
        if elapsed < COOLDOWN_SECONDS:
            return {
                "sent": False,
                "reason": f"Cooldown active ({int(COOLDOWN_SECONDS - elapsed)}s remaining for this call)",
                "call_id": call_key
            }

    # 2. Check credentials & phone number
    sid = _CONFIG["account_sid"] or os.getenv("TWILIO_ACCOUNT_SID", "")
    token = _CONFIG["auth_token"] or os.getenv("TWILIO_AUTH_TOKEN", "")
    from_num = _CONFIG["from_number"] or os.getenv("TWILIO_PHONE_NUMBER", "")
    to_num = target_phone or _CONFIG["emergency_contact"] or os.getenv("EMERGENCY_CONTACT_PHONE", "")

    message_body = format_alert_message(scam_analysis, is_fake_voice, call_id)

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "call_id": call_key,
        "to_number": to_num or "None",
        "category": scam_analysis.get("primary_category"),
        "is_fake_voice": is_fake_voice,
        "threat_level": scam_analysis.get("risk_level", "HIGH"),
        "message": message_body,
        "status": "pending",
        "sid": None,
        "error": None
    }

    if not (sid and token and from_num and to_num):
        log_entry["status"] = "simulated"
        log_entry["error"] = "Twilio credentials or recipient phone not configured"
        _DISPATCH_LOGS.insert(0, log_entry)
        print(f"[SMS Service] SIMULATION: Twilio credentials not set. Would send alert to '{to_num}':\n{message_body}")
        # Still record cooldown so simulation also doesn't spam logs
        _ALERT_COOLDOWN_CACHE[call_key] = now
        return {
            "sent": False,
            "simulated": True,
            "message": "Twilio credentials not configured; alert simulated in console.",
            "body": message_body,
            "recipient": to_num
        }

    # 3. Dispatch via official Twilio Client
    try:
        from twilio.rest import Client
        client = Client(sid, token)
        message = client.messages.create(
            body=message_body,
            from_=from_num,
            to=to_num
        )
        _ALERT_COOLDOWN_CACHE[call_key] = now
        log_entry["status"] = "delivered"
        log_entry["sid"] = message.sid
        _DISPATCH_LOGS.insert(0, log_entry)
        if len(_DISPATCH_LOGS) > 100:
            _DISPATCH_LOGS.pop()

        print(f"[SMS Service] Automated SMS dispatched successfully to {to_num}! SID: {message.sid}")
        return {
            "sent": True,
            "sid": message.sid,
            "recipient": to_num,
            "body": message_body,
            "status": message.status
        }
    except Exception as e:
        err_msg = str(e)
        log_entry["status"] = "failed"
        log_entry["error"] = err_msg
        _DISPATCH_LOGS.insert(0, log_entry)
        print(f"[SMS Service] Twilio dispatch error: {err_msg}")
        return {
            "sent": False,
            "error": err_msg,
            "recipient": to_num
        }

def send_test_sms(target_phone: Optional[str] = None) -> Dict[str, Any]:
    """Test endpoint helper to verify Twilio connectivity."""
    sid = _CONFIG["account_sid"] or os.getenv("TWILIO_ACCOUNT_SID", "")
    token = _CONFIG["auth_token"] or os.getenv("TWILIO_AUTH_TOKEN", "")
    from_num = _CONFIG["from_number"] or os.getenv("TWILIO_PHONE_NUMBER", "")
    to_num = target_phone or _CONFIG["emergency_contact"] or os.getenv("EMERGENCY_CONTACT_PHONE", "")

    if not (sid and token and from_num and to_num):
        return {
            "sent": False,
            "error": "Missing configuration. Provide TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, and destination number."
        }

    test_analysis = {
        "primary_category": "Test",
        "matched_keywords": ["test verification"],
        "urgency_markers": [],
        "risk_level": "TEST",
        "is_scam": True
    }
    
    return send_scam_alert_sms(
        call_id=f"test-{int(time.time())}",
        scam_analysis=test_analysis,
        is_fake_voice=True,
        target_phone=to_num,
        force=True
    )

def get_dispatch_logs(limit: int = 20) -> list:
    """Returns recent alert dispatch logs."""
    return _DISPATCH_LOGS[:limit]

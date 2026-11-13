"""
db.py - SQLite Audit Database for Voice Clone & Scam Detection Calls
Stores audit trails, deepfake classifications, NLP scam risk scores, and forensic telemetry.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = "calls.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes calls table schema."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            caller_label TEXT DEFAULT 'Incoming Call',
            verdict TEXT NOT NULL,
            composite_verdict TEXT NOT NULL,
            threat_score REAL NOT NULL,
            scam_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            primary_category TEXT,
            detected_language TEXT DEFAULT 'en',
            transcript TEXT,
            matched_keywords TEXT,
            urgency_markers TEXT,
            duration REAL DEFAULT 0.0,
            flagged INTEGER DEFAULT 0,
            acoustic_data TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON call_logs(timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_flagged ON call_logs(flagged)")
    conn.commit()
    conn.close()

def log_call(
    verdict: str,
    composite_verdict: str,
    threat_score: float,
    scam_score: float,
    risk_level: str,
    primary_category: Optional[str],
    detected_language: str,
    transcript: str,
    matched_keywords: List[str],
    urgency_markers: List[str],
    duration: float = 0.0,
    acoustic_data: Optional[Dict[str, Any]] = None,
    caller_label: str = "Live Audio Stream"
) -> int:
    """Inserts a new call scan into the audit log."""
    conn = get_connection()
    cursor = conn.cursor()
    
    flagged = 1 if (threat_score >= 50.0 or scam_score >= 50.0) else 0
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO call_logs (
            timestamp, caller_label, verdict, composite_verdict, threat_score,
            scam_score, risk_level, primary_category, detected_language,
            transcript, matched_keywords, urgency_markers, duration, flagged, acoustic_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_iso,
        caller_label,
        verdict,
        composite_verdict,
        float(threat_score),
        float(scam_score),
        risk_level,
        primary_category,
        detected_language,
        transcript,
        json.dumps(matched_keywords or []),
        json.dumps(urgency_markers or []),
        float(duration),
        flagged,
        json.dumps(acoustic_data or {})
    ))
    
    call_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return call_id

def get_recent_calls(limit: int = 50, filter_flagged_only: bool = False) -> List[Dict[str, Any]]:
    """Retrieves recent call history with parsed JSON fields."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if filter_flagged_only:
        cursor.execute("SELECT * FROM call_logs WHERE flagged = 1 ORDER BY id DESC LIMIT ?", (limit,))
    else:
        cursor.execute("SELECT * FROM call_logs ORDER BY id DESC LIMIT ?", (limit,))
        
    rows = cursor.fetchall()
    calls = []
    for r in rows:
        calls.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "caller_label": r["caller_label"],
            "verdict": r["verdict"],
            "composite_verdict": r["composite_verdict"],
            "threat_score": r["threat_score"],
            "scam_score": r["scam_score"],
            "risk_level": r["risk_level"],
            "primary_category": r["primary_category"],
            "detected_language": r["detected_language"],
            "transcript": r["transcript"],
            "matched_keywords": json.loads(r["matched_keywords"] or "[]"),
            "urgency_markers": json.loads(r["urgency_markers"] or "[]"),
            "duration": r["duration"],
            "flagged": bool(r["flagged"]),
            "acoustic_data": json.loads(r["acoustic_data"] or "{}")
        })
    conn.close()
    return calls

def get_analytics() -> Dict[str, Any]:
    """Computes high-level fraud prevention metrics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM call_logs")
    total_calls = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM call_logs WHERE composite_verdict = 'CRITICAL_AI_SCAM'")
    critical_ai_scams = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM call_logs WHERE threat_score >= 50.0")
    deepfakes_detected = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM call_logs WHERE scam_score >= 50.0")
    scams_flagged = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT COUNT(*) FROM call_logs WHERE flagged = 0")
    safe_calls = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(scam_score), AVG(threat_score) FROM call_logs")
    avg_row = cursor.fetchone()
    avg_scam = round(avg_row[0] or 0.0, 1)
    avg_threat = round(avg_row[1] or 0.0, 1)

    # Categories breakdown
    cursor.execute("""
        SELECT primary_category, COUNT(*) as cnt 
        FROM call_logs 
        WHERE primary_category IS NOT NULL 
        GROUP BY primary_category 
        ORDER BY cnt DESC
    """)
    categories = [{"name": row[0], "count": row[1]} for row in cursor.fetchall()]

    conn.close()
    return {
        "total_calls": total_calls,
        "critical_ai_scams": critical_ai_scams,
        "deepfakes_detected": deepfakes_detected,
        "scams_flagged": scams_flagged,
        "safe_calls": safe_calls,
        "avg_scam_score": avg_scam,
        "avg_threat_score": avg_threat,
        "categories_breakdown": categories
    }

def clear_all_calls():
    """Wipes calls table for fresh demo sessions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM call_logs")
    conn.commit()
    conn.close()

# Auto-initialize on import
init_db()

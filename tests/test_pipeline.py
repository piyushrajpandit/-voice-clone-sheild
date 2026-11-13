"""
Integration and latency benchmark tests for pipeline.py:
- Full Section 5.2 schema verification
- Real human vs AI clone separation
- 4-second latency constraint (< 5.0 seconds)
"""
import os
import pytest
from pipeline import analyze_voice

def test_analyze_voice_schema_and_performance():
    # Warm-up / load pipeline once
    res_clone = analyze_voice("samples/sample_ai_clone.wav", clip_id="test_perf_clone")

    # 1. Verify schema adherence
    required_keys = [
        "clip_id", "ai_score", "segments", "spectrogram_path",
        "forensics", "processing_time_sec", "duration_sec"
    ]
    for key in required_keys:
        assert key in res_clone, f"Missing key in response: {key}"

    # 2. Check score validity
    assert 0.0 <= res_clone["ai_score"] <= 1.0
    assert res_clone["ai_score"] > 0.8, f"Expected high AI score for clone, got {res_clone['ai_score']}"

    # 3. Check segments timeline
    assert isinstance(res_clone["segments"], list)
    assert len(res_clone["segments"]) >= 1
    for seg in res_clone["segments"]:
        assert "start" in seg
        assert "end" in seg
        assert "ai_score" in seg

    # 4. Check spectrogram image output
    assert os.path.exists(res_clone["spectrogram_path"])
    assert os.path.getsize(res_clone["spectrogram_path"]) > 5000

    # 5. Check forensics dictionary
    forensics_keys = [
        "pitch_f0_mean", "pitch_jitter_std",
        "spectral_centroid", "spectral_flatness", "forensics_score"
    ]
    for f_key in forensics_keys:
        assert f_key in res_clone["forensics"]
        assert 0.0 <= res_clone["forensics"]["forensics_score"] <= 1.0

    # 6. Verify latency on a 4-second clip (< 5.0 seconds requirement)
    # Perform timed run
    timed_res = analyze_voice("samples/sample_real_human.wav", clip_id="test_perf_real")
    print(f"\n[LATENCY BENCHMARK] 4s voice analysis executed in {timed_res['processing_time_sec']}s")
    assert timed_res["processing_time_sec"] < 5.0, (
        f"Latency exceeded 5.0s SLA! Took {timed_res['processing_time_sec']}s"
    )
    assert timed_res["ai_score"] < 0.25, (
        f"Expected low AI score for human, got {timed_res['ai_score']}"
    )

    # Cleanup test spectrograms
    for p in [res_clone["spectrogram_path"], timed_res["spectrogram_path"]]:
        if os.path.exists(p):
            os.remove(p)

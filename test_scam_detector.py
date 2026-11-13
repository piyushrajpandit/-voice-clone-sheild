"""
Unit test suite for scam_detector.py
Tests English, Hindi, and Benign conversational scenarios.
"""

from scam_detector import analyze_transcript

test_cases = [
    {
        "name": "1. AI Grandparent Bail Scam (English)",
        "text": "Mom, please listen to me, I was arrested after car accident and my phone battery is dying. You must wire $3000 to this bail bondsman immediately.",
        "lang": "en",
        "expected_scam": True
    },
    {
        "name": "2. Banking & OTP Phishing Scam (Hindi)",
        "text": "नमस्ते सर, मैं स्टेट बैंक से बोल रहा हूँ। आपका बैंक खाता बंद हो गया है, तुरंत ओटीपी बताओ वरना खाता सस्पेंड हो जाएगा।",
        "lang": "hi",
        "expected_scam": True
    },
    {
        "name": "3. Police Arrest Extortion Scam (Hindi)",
        "text": "आपका बेटा पुलिस स्टेशन में गिरफ्तार है, तुरंत पैसे ट्रांसफर करो वरना जेल भेज देंगे। किसी को मत बताना।",
        "lang": "hi",
        "expected_scam": True
    },
    {
        "name": "4. Lottery & Gift Card Fraud (English)",
        "text": "Congratulations! You have won the $50,000 cash prize. To claim your reward, buy an Apple gift card and share the code immediately.",
        "lang": "en",
        "expected_scam": True
    },
    {
        "name": "5. Normal Family Conversation (Benign)",
        "text": "Hey Dad, just calling to ask if we should meet for dinner at 7 tonight. Let me know when you're free.",
        "lang": "en",
        "expected_scam": False
    },
    {
        "name": "6. Normal Business Chat (Benign)",
        "text": "Good morning team, let us review the quarterly report presentation and discuss next steps for the product launch.",
        "lang": "en",
        "expected_scam": False
    }
]

print("=" * 70)
print("TESTING SCAM DETECTION ENGINE (scam_detector.py)")
print("=" * 70)

all_passed = True
for tc in test_cases:
    result = analyze_transcript(tc["text"], language=tc["lang"])
    passed = result["is_scam"] == tc["expected_scam"]
    if not passed:
        all_passed = False

    status_icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"\nScenario: {tc['name']}")
    print(f"Status:   {status_icon}")
    print(f"Scam?:    {result['is_scam']} (Score: {result['scam_score']}%, Risk: {result['risk_level']})")
    print(f"Category: {result['primary_category']}")
    print(f"Keywords: {result['matched_keywords']}")
    print(f"Urgency:  {result['urgency_markers']}")
    print(f"Advice:   {result['recommended_action'][:80]}...")

print("\n" + "=" * 70)
if all_passed:
    print("🎉 ALL 6 TEST CASES PASSED SUCCESSFULLY!")
else:
    print("⚠️ SOME TEST CASES FAILED!")
print("=" * 70)

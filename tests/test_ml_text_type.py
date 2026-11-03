"""
Unit tests for TextTypeDetector:
- Syntactic token matcher for questions, greetings, requests, and statements.
- English and Hindi communicative intent classification.
- Graceful uncertainty handling.
"""

import pytest
from ml.text_type_detector import TextTypeDetector


@pytest.fixture
def detector():
    return TextTypeDetector(uncertainty_threshold=0.60)


def test_question_detection_english_and_hindi(detector):
    questions = [
        "What time does the meeting start?",
        "Can you help me with this task?",
        "क्या आप मेरी मदद कर सकते हैं?",
        "आप कहाँ जा रहे हैं?",
        "Why is the application not responding?",
    ]
    for q in questions:
        res = detector.predict(q)
        assert res.label == "Question"
        assert res.confidence >= 0.70
        assert not res.is_uncertain


def test_greeting_detection_english_and_hindi(detector):
    greetings = [
        "Good morning, have a wonderful day ahead!",
        "Hello everyone, welcome to the session.",
        "नमस्ते, आपका स्वागत है।",
        "नमस्कार, आपका दिन मंगलमय हो।",
    ]
    for g in greetings:
        res = detector.predict(g)
        assert res.label == "Greeting"
        assert res.confidence >= 0.70
        assert not res.is_uncertain


def test_request_instruction_detection_english_and_hindi(detector):
    requests = [
        "Please close the door behind you.",
        "Kindly submit your assignment by 5 PM.",
        "कृपया दरवाज़ा बंद कर दीजिए।",
        "कृपया यहाँ प्रतीक्षा करें।",
        "Make sure to save your work before exiting.",
    ]
    for r in requests:
        res = detector.predict(r)
        assert res.label == "Request / Instruction"
        assert res.confidence >= 0.70
        assert not res.is_uncertain


def test_statement_detection_english_and_hindi(detector):
    statements = [
        "The train arrives at 6 PM.",
        "यह किताब मेज़ पर रखी है।",
        "कल हम सभी एक साथ पिकनिक पर जाने वाले हैं।",
        "The meeting was held in room 405.",
    ]
    for s in statements:
        res = detector.predict(s)
        assert res.label == "Statement"
        assert res.confidence >= 0.60
        assert not res.is_uncertain


def test_empty_text_type_input(detector):
    res = detector.predict("")
    assert res.label == "Uncertain"
    assert res.is_uncertain is True
    assert res.confidence == 0.0

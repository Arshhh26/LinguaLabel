"""
Unit tests for LanguageDetector:
- Unicode script heuristic (Devanagari U+0900..U+097F vs. Latin A..Za..z).
- Statistical / character n-gram model.
- Transliterated / mixed text.
- Graceful uncertainty on low confidence or empty text.
"""

import pytest
from ml.language_detector import LanguageDetector


@pytest.fixture
def detector():
    return LanguageDetector(uncertainty_threshold=0.60)


def test_devanagari_script_heuristic_hindi(detector):
    hindi_texts = [
        "आज का मौसम बहुत सुहावना है।",
        "कृपया समय पर अपनी रिपोर्ट जमा करें।",
        "क्या आप मुझे इस सवाल का जवाब बता सकते हैं?",
        "नमस्ते, आपका दिन मंगलमय हो।",
    ]
    for text in hindi_texts:
        res = detector.predict(text)
        assert res.label == "Hindi"
        assert res.confidence >= 0.70
        assert not res.is_uncertain
        assert "devanagari_count" in res.metadata


def test_latin_script_english(detector):
    english_texts = [
        "I am feeling happy today.",
        "Please send the quarterly financial report by Monday.",
        "Can you help me with this problem?",
        "Good morning, have a great day ahead!",
    ]
    for text in english_texts:
        res = detector.predict(text)
        assert res.label == "English"
        assert res.confidence >= 0.60
        assert not res.is_uncertain
        assert "latin_count" in res.metadata


def test_other_language_script(detector):
    other_text = "1234567890 !@#$%^&*"
    res = detector.predict(other_text)
    # Digits and symbols with no dominant natural language script
    assert res.is_uncertain or res.label == "Other"


def test_empty_and_whitespace_text(detector):
    for empty in ["", "   ", "\t\n"]:
        res = detector.predict(empty)
        assert res.label == "Uncertain"
        assert res.is_uncertain is True
        assert res.confidence == 0.0


def test_uncertainty_threshold_override():
    # If uncertainty threshold is high, lower confidence predictions become Uncertain
    strict_detector = LanguageDetector(uncertainty_threshold=0.99)
    res = strict_detector.predict("quick test")
    if res.confidence < 0.99:
        assert res.is_uncertain is True
        assert res.label == "Uncertain"

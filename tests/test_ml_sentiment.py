"""
Unit tests for SentimentDetector:
- Bilingual polarity lexicon scoring (English + Hindi).
- Contextual negation parsing (नहीं, मत, ना in Hindi; not, never, no in English).
- Neutral factual statement handling.
- Graceful uncertainty handling.
"""

import pytest
from ml.sentiment_detector import SentimentDetector


@pytest.fixture
def detector():
    return SentimentDetector(uncertainty_threshold=0.60)


def test_positive_sentiment_english(detector):
    texts = [
        "I am feeling happy today.",
        "This is an excellent and wonderful experience!",
        "Thank you so much for your great assistance.",
    ]
    for text in texts:
        res = detector.predict(text)
        assert res.label == "Positive"
        assert res.confidence >= 0.60
        assert not res.is_uncertain


def test_positive_sentiment_hindi(detector):
    texts = [
        "मुझे आज बहुत खुशी हो रही है।",
        "यह खाना बहुत स्वादिष्ट और लजीज था।",
        "शानदार प्रदर्शन, हमें आप पर गर्व है।",
    ]
    for text in texts:
        res = detector.predict(text)
        assert res.label == "Positive"
        assert res.confidence >= 0.60
        assert not res.is_uncertain


def test_negative_sentiment_english(detector):
    texts = [
        "This product is terrible, awful and broken.",
        "Horrible service and completely wasted my time.",
        "The flight was cancelled and luggage was delayed.",
    ]
    for text in texts:
        res = detector.predict(text)
        assert res.label == "Negative"
        assert res.confidence >= 0.60
        assert not res.is_uncertain


def test_negative_sentiment_hindi(detector):
    texts = [
        "यह मौसम बहुत खराब और कष्टदायक है।",
        "सेवा बहुत घटिया और निराशाजनक है।",
        "ट्रेन में बहुत परेशानी और दर्द हुआ।",
    ]
    for text in texts:
        res = detector.predict(text)
        assert res.label == "Negative"
        assert res.confidence >= 0.60
        assert not res.is_uncertain


def test_hindi_negation_flips_polarity(detector):
    # Positive word negated by 'नहीं' or 'ना'
    negated_texts = [
        "यह खाना अच्छा नहीं है।",
        "मुझे यह फ़िल्म पसंद नहीं आई।",
    ]
    for text in negated_texts:
        res = detector.predict(text)
        assert res.label == "Negative"


def test_english_negation_flips_polarity(detector):
    negated_texts = [
        "The product is not good at all.",
        "I was never happy with their response.",
    ]
    for text in negated_texts:
        res = detector.predict(text)
        assert res.label == "Negative"


def test_empty_sentiment_input(detector):
    res = detector.predict("")
    assert res.label == "Uncertain"
    assert res.is_uncertain is True
    assert res.confidence == 0.0

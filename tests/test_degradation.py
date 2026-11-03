"""
Unit tests for graceful degradation, error handling, and advisory contract
in LinguaLabel ML Engine (Part 5).
"""

from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.language_detector import LanguageDetector
from ml.sentiment_detector import SentimentDetector
from ml.text_type_detector import TextTypeDetector


def test_prediction_result_contract():
    res = PredictionResult(
        label="Hindi",
        confidence=0.95432,
        method="Script Heuristic",
        is_uncertain=False,
        metadata={"chars": 15}
    )
    d = res.to_dict()
    assert d["label"] == "Hindi"
    assert d["confidence"] == 0.9543
    assert d["method"] == "Script Heuristic"
    assert d["is_uncertain"] is False
    assert d["metadata"] == {"chars": 15}


def test_engine_availability_never_raises():
    engine = MLEngine()
    # Must return boolean without raising any exception
    avail = engine.is_available()
    assert isinstance(avail, bool)


def test_engine_predict_all_structure():
    engine = MLEngine()
    preds = engine.predict_all("Hello world!")
    assert "language" in preds
    assert "sentiment" in preds
    assert "text_type" in preds

    for task_name, res in preds.items():
        assert isinstance(res, PredictionResult)
        assert hasattr(res, "label")
        assert hasattr(res, "confidence")
        assert hasattr(res, "method")
        assert hasattr(res, "is_uncertain")


def test_engine_handles_none_and_empty():
    engine = MLEngine()
    for bad_input in [None, "", "   ", "\n\t"]:
        preds = engine.predict_all(bad_input)
        for task_name, res in preds.items():
            assert res.is_uncertain is True
            assert res.label == "Uncertain"
            assert res.confidence == 0.0


def test_engine_degrades_gracefully_when_detector_is_none():
    engine = MLEngine()
    # Simulate uninitialized detector
    engine._language_detector = None
    engine._sentiment_detector = None
    engine._text_type_detector = None

    assert engine.is_available() is False

    preds = engine.predict_all("Sample sentence")
    for task_name, res in preds.items():
        assert res.label == "Uncertain"
        assert res.is_uncertain is True
        assert res.method == "Unavailable"


def test_low_confidence_forces_uncertain():
    # Detector with high uncertainty threshold forces Uncertain tag
    lang_det = LanguageDetector(uncertainty_threshold=0.999)
    res = lang_det.predict("bonjour")
    assert res.is_uncertain is True
    assert res.label == "Uncertain"

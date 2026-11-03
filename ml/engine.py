"""
MLEngine — Unified Coordinator for LinguaLabel Machine Learning Inference.

Manages language, sentiment, and text type predictors.
Enforces non-destructive advisory contracts:
- Never commits to disk or mutates datasets.
- Gracefully degrades to manual mode when dependencies or models are unavailable.
- Labels predictions with confidence < 0.60 as Uncertain.
"""

from typing import Dict, Optional
from ml.base import PredictionResult
from ml.language_detector import LanguageDetector
from ml.sentiment_detector import SentimentDetector
from ml.text_type_detector import TextTypeDetector


class MLEngine:
    """
    Unified inference coordinator providing advisory predictions.
    """

    def __init__(self, uncertainty_threshold: float = 0.60):
        self.uncertainty_threshold = uncertainty_threshold
        self._language_detector: Optional[LanguageDetector] = None
        self._sentiment_detector: Optional[SentimentDetector] = None
        self._text_type_detector: Optional[TextTypeDetector] = None
        self._initialize_detectors()

    def _initialize_detectors(self):
        try:
            self._language_detector = LanguageDetector(uncertainty_threshold=self.uncertainty_threshold)
            self._sentiment_detector = SentimentDetector(uncertainty_threshold=self.uncertainty_threshold)
            self._text_type_detector = TextTypeDetector(uncertainty_threshold=self.uncertainty_threshold)
        except Exception:
            # Leave predictors as None for graceful degradation
            pass

    def is_available(self) -> bool:
        """
        Returns True only if all detectors are initialized and ready.
        Does not throw exceptions if models are missing or uninitialized.
        """
        try:
            return bool(
                self._language_detector is not None and self._language_detector.is_ready() and
                self._sentiment_detector is not None and self._sentiment_detector.is_ready() and
                self._text_type_detector is not None and self._text_type_detector.is_ready()
            )
        except Exception:
            return False

    def predict_language(self, text: str) -> PredictionResult:
        if not self._language_detector:
            return PredictionResult("Uncertain", 0.0, "Unavailable", is_uncertain=True)
        try:
            return self._language_detector.predict(text)
        except Exception as e:
            return PredictionResult("Uncertain", 0.0, "Fallback", is_uncertain=True, metadata={"error": str(e)})

    def predict_sentiment(self, text: str) -> PredictionResult:
        if not self._sentiment_detector:
            return PredictionResult("Uncertain", 0.0, "Unavailable", is_uncertain=True)
        try:
            return self._sentiment_detector.predict(text)
        except Exception as e:
            return PredictionResult("Uncertain", 0.0, "Fallback", is_uncertain=True, metadata={"error": str(e)})

    def predict_text_type(self, text: str) -> PredictionResult:
        if not self._text_type_detector:
            return PredictionResult("Uncertain", 0.0, "Unavailable", is_uncertain=True)
        try:
            return self._text_type_detector.predict(text)
        except Exception as e:
            return PredictionResult("Uncertain", 0.0, "Fallback", is_uncertain=True, metadata={"error": str(e)})

    def predict_all(self, text: str) -> Dict[str, PredictionResult]:
        """
        Computes advisory predictions for Language, Sentiment, and Text Type.
        """
        return {
            "language": self.predict_language(text),
            "sentiment": self.predict_sentiment(text),
            "text_type": self.predict_text_type(text),
        }

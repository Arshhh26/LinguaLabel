"""
MLEngine — Unified Hybrid Coordinator for LinguaLabel Machine Learning Inference.

Manages language, sentiment, and text type predictors across multiple backends:
- Baseline: Fast Unicode script heuristics, curated bilingual lexicons, and n-gram classifiers.
- Transformer: Multilingual XLM-RoBERTa sentiment and mDeBERTa zero-shot intent models.

Enforces non-destructive advisory contracts:
- Never commits to disk or mutates datasets automatically.
- Gracefully degrades to baseline or manual mode when dependencies or models are unavailable.
- Labels predictions with confidence < 0.60 as Uncertain.
- Caches model instances to prevent redundant allocations and re-downloads.
"""

from typing import Any, Dict, Optional, Union
from ml.base import BasePredictor, PredictionResult
from ml.language_detector import LanguageDetector
from ml.model_config import (
    DEFAULT_SENTIMENT_CHECKPOINT,
    DEFAULT_TEXT_TYPE_CHECKPOINT,
    MLBackend,
    MLEngineConfig,
)
from ml.model_registry import ModelRegistry
from ml.sentiment_detector import SentimentDetector
from ml.text_type_detector import TextTypeDetector
from ml.transformers.model_loader import get_model_loader


class MLEngine:
    """
    Unified hybrid inference coordinator providing advisory predictions.
    Supports seamless switching between baseline and transformer backends.
    """

    def __init__(
        self,
        uncertainty_threshold: Union[float, MLEngineConfig] = 0.60,
        backend: Union[MLBackend, str] = MLBackend.BASELINE,
        config: Optional[MLEngineConfig] = None,
        device: str = "auto",
    ):
        if isinstance(uncertainty_threshold, MLEngineConfig):
            config = uncertainty_threshold
            uncertainty_threshold = float(config.uncertainty_threshold)

        self.uncertainty_threshold = float(uncertainty_threshold)
        if isinstance(backend, str):
            backend = MLBackend(backend.lower())

        self.config = config or MLEngineConfig(
            active_backend=backend,
            uncertainty_threshold=self.uncertainty_threshold,
            device=device,
        )
        self.config.active_backend = backend
        self.config.uncertainty_threshold = self.uncertainty_threshold

        self.registry = ModelRegistry(self.config)

        # Baseline detectors maintained for direct access and backward compatibility
        self._language_detector: Optional[LanguageDetector] = None
        self._sentiment_detector: Optional[SentimentDetector] = None
        self._text_type_detector: Optional[TextTypeDetector] = None

        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initializes baseline detectors and configures active backends."""
        try:
            self._language_detector = self.registry.get_language_detector(
                uncertainty_threshold=self.uncertainty_threshold
            )
            self._sentiment_detector = self.registry.get_sentiment_detector(
                backend=MLBackend.BASELINE,
                uncertainty_threshold=self.uncertainty_threshold,
            )  # type: ignore
            self._text_type_detector = self.registry.get_text_type_detector(
                backend=MLBackend.BASELINE,
                uncertainty_threshold=self.uncertainty_threshold,
            )  # type: ignore
        except Exception:
            pass

    @property
    def active_backend(self) -> MLBackend:
        return self.config.active_backend

    def set_backend(self, backend: Union[MLBackend, str]) -> None:
        """Switches the active ML prediction backend (baseline or transformer)."""
        if isinstance(backend, str):
            backend = MLBackend(backend.lower())
        self.config.active_backend = backend

    def get_backend_info(self) -> Dict[str, Any]:
        """Returns diagnostic info about the active backend and loaded models."""
        loader = get_model_loader()
        is_tf_avail = loader.is_transformers_available and loader.is_torch_available
        device = loader.get_device(self.config.device)

        return {
            "active_backend": self.config.active_backend.value,
            "uncertainty_threshold": self.uncertainty_threshold,
            "device": device,
            "torch_available": loader.is_torch_available,
            "transformers_available": loader.is_transformers_available,
            "sentiment_checkpoint": self.config.sentiment_checkpoint,
            "text_type_checkpoint": self.config.text_type_checkpoint,
            "sentiment_cached": loader.is_cached(self.config.sentiment_checkpoint),
            "text_type_cached": loader.is_cached(self.config.text_type_checkpoint),
            "fallback_enabled": self.config.enable_fallback_to_baseline,
        }

    def is_available(self) -> bool:
        """
        Returns True only if all required detectors for the active backend are initialized.
        Does not throw exceptions if models are missing or uninitialized.
        """
        try:
            # Baseline language detector is required for all backends
            if not self._language_detector or not self._language_detector.is_ready():
                return False

            if self.config.active_backend == MLBackend.BASELINE:
                return bool(
                    self._sentiment_detector is not None and self._sentiment_detector.is_ready() and
                    self._text_type_detector is not None and self._text_type_detector.is_ready()
                )
            else:
                # Transformer backend availability
                loader = get_model_loader()
                if not loader.is_torch_available or not loader.is_transformers_available:
                    # If dependencies are missing but fallback is allowed, still available via fallback
                    return self.config.enable_fallback_to_baseline

                # Check if transformer detectors can be instantiated
                sent_det = self.registry.get_sentiment_detector(backend=MLBackend.TRANSFORMER)
                tt_det = self.registry.get_text_type_detector(backend=MLBackend.TRANSFORMER)
                return bool(sent_det is not None and tt_det is not None)
        except Exception:
            return False

    def predict_language(self, text: str) -> PredictionResult:
        """Produces language identification prediction."""
        if not self._language_detector:
            return PredictionResult("Uncertain", 0.0, "Unavailable", is_uncertain=True)
        try:
            return self._language_detector.predict(text)
        except Exception as e:
            return PredictionResult("Uncertain", 0.0, "Fallback", is_uncertain=True, metadata={"error": str(e)})

    def predict_sentiment(self, text: str) -> PredictionResult:
        """
        Produces sentiment polarity prediction using active backend.
        Falls back to baseline if transformer is unavailable and fallback is enabled.
        """
        if self.config.active_backend == MLBackend.TRANSFORMER:
            try:
                tf_det = self.registry.get_sentiment_detector(
                    backend=MLBackend.TRANSFORMER,
                    uncertainty_threshold=self.uncertainty_threshold,
                    checkpoint=self.config.sentiment_checkpoint,
                    device=self.config.device,
                )
                if tf_det.is_ready() or tf_det.load():
                    pred = tf_det.predict(text)
                    if not (pred.is_uncertain and "error" in pred.metadata and self.config.enable_fallback_to_baseline):
                        pred.metadata["actual_backend"] = "transformer"
                        return pred
            except Exception as e:
                if not self.config.enable_fallback_to_baseline:
                    return PredictionResult(
                        "Uncertain", 0.0, "Transformer Error",
                        is_uncertain=True,
                        confidence_type="heuristic",
                        metadata={"actual_backend": "transformer", "error": str(e)}
                    )

            # Fallback to baseline
            if self._sentiment_detector:
                base_pred = self._sentiment_detector.predict(text)
                base_pred.metadata["actual_backend"] = "fallback_baseline"
                base_pred.metadata["is_fallback"] = True
                base_pred.metadata["fallback_from"] = "transformer"
                return base_pred

        if not self._sentiment_detector:
            return PredictionResult("Uncertain", 0.0, "Unavailable", is_uncertain=True, confidence_type="heuristic", metadata={"actual_backend": "baseline"})
        try:
            pred = self._sentiment_detector.predict(text)
            pred.metadata.setdefault("actual_backend", "baseline")
            return pred
        except Exception as e:
            return PredictionResult("Uncertain", 0.0, "Fallback", is_uncertain=True, confidence_type="heuristic", metadata={"actual_backend": "baseline", "error": str(e)})

    def predict_text_type(self, text: str) -> PredictionResult:
        """
        Produces text-type / intent prediction using active backend.
        Falls back to baseline if transformer is unavailable and fallback is enabled.
        """
        if self.config.active_backend == MLBackend.TRANSFORMER:
            try:
                tf_det = self.registry.get_text_type_detector(
                    backend=MLBackend.TRANSFORMER,
                    uncertainty_threshold=self.uncertainty_threshold,
                    checkpoint=self.config.text_type_checkpoint,
                    device=self.config.device,
                )
                if tf_det.is_ready() or tf_det.load():
                    pred = tf_det.predict(text)
                    if not (pred.is_uncertain and "error" in pred.metadata and self.config.enable_fallback_to_baseline):
                        pred.metadata["actual_backend"] = "transformer"
                        return pred
            except Exception as e:
                if not self.config.enable_fallback_to_baseline:
                    return PredictionResult(
                        "Uncertain", 0.0, "Transformer Error",
                        is_uncertain=True,
                        confidence_type="heuristic",
                        metadata={"actual_backend": "transformer", "error": str(e)}
                    )

            # Fallback to baseline
            if self._text_type_detector:
                base_pred = self._text_type_detector.predict(text)
                base_pred.metadata["actual_backend"] = "fallback_baseline"
                base_pred.metadata["is_fallback"] = True
                base_pred.metadata["fallback_from"] = "transformer"
                return base_pred

        if not self._text_type_detector:
            return PredictionResult("Uncertain", 0.0, "Unavailable", is_uncertain=True, confidence_type="heuristic", metadata={"actual_backend": "baseline"})
        try:
            pred = self._text_type_detector.predict(text)
            pred.metadata.setdefault("actual_backend", "baseline")
            return pred
        except Exception as e:
            return PredictionResult("Uncertain", 0.0, "Fallback", is_uncertain=True, confidence_type="heuristic", metadata={"actual_backend": "baseline", "error": str(e)})

    def predict_all(self, text: str) -> Dict[str, PredictionResult]:
        """
        Computes advisory predictions for Language, Sentiment, and Text Type.
        """
        return {
            "language": self.predict_language(text),
            "sentiment": self.predict_sentiment(text),
            "text_type": self.predict_text_type(text),
        }

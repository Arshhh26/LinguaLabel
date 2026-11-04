"""
Model Registry for LinguaLabel.

Manages backend discovery, registration, and instance caching for both
baseline detectors and transformer models.
"""

from typing import Any, Dict, List, Optional, Type
from ml.base import BasePredictor
from ml.language_detector import LanguageDetector
from ml.model_config import (
    DEFAULT_SENTIMENT_CHECKPOINT,
    DEFAULT_TEXT_TYPE_CHECKPOINT,
    MLBackend,
    MLEngineConfig,
    VERIFIED_MODELS,
    ModelMetadata,
)
from ml.sentiment_detector import SentimentDetector
from ml.text_type_detector import TextTypeDetector
from ml.transformers.model_loader import get_model_loader
from ml.transformers.sentiment_transformer import SentimentTransformer
from ml.transformers.text_type_transformer import TextTypeTransformer


class ModelRegistry:
    """
    Registry for managing model backends and runtime predictors.
    """

    def __init__(self, config: Optional[MLEngineConfig] = None):
        self.config = config or MLEngineConfig()
        self._instances: Dict[str, BasePredictor] = {}

    def get_supported_backends(self) -> List[str]:
        return [MLBackend.BASELINE.value, MLBackend.TRANSFORMER.value]

    def get_verified_models(self) -> Dict[str, ModelMetadata]:
        return dict(VERIFIED_MODELS)

    def get_sentiment_detector(
        self,
        backend: Optional[MLBackend] = None,
        uncertainty_threshold: Optional[float] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> BasePredictor:
        """Retrieves or creates cached sentiment detector for the specified backend."""
        active_backend = backend or self.config.active_backend
        thresh = uncertainty_threshold if uncertainty_threshold is not None else self.config.uncertainty_threshold
        ckpt = checkpoint or self.config.sentiment_checkpoint
        dev = device or self.config.device

        key = f"sentiment_{active_backend.value}_{ckpt}_{thresh}_{dev}"
        if key in self._instances:
            return self._instances[key]

        if active_backend == MLBackend.TRANSFORMER:
            detector = SentimentTransformer(
                checkpoint=ckpt,
                uncertainty_threshold=thresh,
                device=dev,
                lazy_load=True,
            )
        else:
            detector = SentimentDetector(uncertainty_threshold=thresh)

        self._instances[key] = detector
        return detector

    def get_text_type_detector(
        self,
        backend: Optional[MLBackend] = None,
        uncertainty_threshold: Optional[float] = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> BasePredictor:
        """Retrieves or creates cached text type detector for the specified backend."""
        active_backend = backend or self.config.active_backend
        thresh = uncertainty_threshold if uncertainty_threshold is not None else self.config.uncertainty_threshold
        ckpt = checkpoint or self.config.text_type_checkpoint
        dev = device or self.config.device

        key = f"text_type_{active_backend.value}_{ckpt}_{thresh}_{dev}"
        if key in self._instances:
            return self._instances[key]

        if active_backend == MLBackend.TRANSFORMER:
            detector = TextTypeTransformer(
                checkpoint=ckpt,
                uncertainty_threshold=thresh,
                device=dev,
                lazy_load=True,
            )
        else:
            detector = TextTypeDetector(uncertainty_threshold=thresh)

        self._instances[key] = detector
        return detector

    def get_language_detector(
        self,
        uncertainty_threshold: Optional[float] = None,
    ) -> LanguageDetector:
        """
        Retrieves baseline LanguageDetector (script heuristic + n-gram TF-IDF).
        Preserved as primary detector per Part 6 architecture requirements.
        """
        thresh = uncertainty_threshold if uncertainty_threshold is not None else self.config.uncertainty_threshold
        key = f"language_baseline_{thresh}"
        if key in self._instances:
            return self._instances[key]  # type: ignore

        detector = LanguageDetector(uncertainty_threshold=thresh)
        self._instances[key] = detector
        return detector

    def clear(self) -> None:
        self._instances.clear()
        get_model_loader().clear_cache()


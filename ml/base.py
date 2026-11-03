"""
Base classes and interfaces for LinguaLabel Machine Learning Engine.
Defines standardized prediction result dataclass and predictor interface.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PredictionResult:
    """
    Standardized container for model suggestions.
    All predictions in LinguaLabel are advisory and never commit to disk
    without explicit annotator confirmation.
    """
    label: str
    confidence: float
    method: str
    is_uncertain: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "method": self.method,
            "is_uncertain": bool(self.is_uncertain),
            "metadata": self.metadata,
        }


class BasePredictor:
    """
    Abstract interface for task-specific predictors.
    """

    def is_ready(self) -> bool:
        """Returns True if model/lexicons are loaded and ready."""
        raise NotImplementedError

    def predict(self, text: str) -> PredictionResult:
        """Produces an advisory prediction for the given text."""
        raise NotImplementedError

"""
Transformer-based Multilingual Sentiment Predictor for LinguaLabel.

Leverages pretrained multilingual XLM-RoBERTa classification checkpoints
(e.g., `cardiffnlp/twitter-xlm-roberta-base-sentiment`).

Enforces:
- Non-destructive advisory contracts.
- Explicit uncertainty thresholds (< 0.60).
- Transparent provenance (checkpoint ID, raw probabilities, calibrated flag).
- Non-raising degradation when dependencies or weights are missing.
"""

import time
from typing import Any, Dict, List, Optional
from ml.base import BasePredictor, PredictionResult
from ml.model_config import (
    DEFAULT_SENTIMENT_CHECKPOINT,
    VERIFIED_MODELS,
    SENTIMENT_LABELS,
)
from ml.transformers.model_loader import get_model_loader


class SentimentTransformer(BasePredictor):
    """
    Multilingual transformer predictor for Sentiment Polarity.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_SENTIMENT_CHECKPOINT,
        uncertainty_threshold: float = 0.60,
        device: str = "auto",
        lazy_load: bool = True,
    ):
        self.checkpoint = checkpoint
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device
        self.loader = get_model_loader()
        self._pipeline = None
        self._load_error: Optional[str] = None
        
        # Metadata configuration
        self.model_meta = VERIFIED_MODELS.get(checkpoint)
        self.label_map = (
            self.model_meta.label_mapping
            if self.model_meta
            else {
                "negative": "Negative",
                "neutral": "Neutral",
                "positive": "Positive",
                "LABEL_0": "Negative",
                "LABEL_1": "Neutral",
                "LABEL_2": "Positive",
            }
        )

        if not lazy_load:
            self.load()

    def load(self) -> bool:
        """Loads or retrieves cached pipeline."""
        pipe, err = self.loader.load_sequence_classification_pipeline(
            self.checkpoint, device=self.device
        )
        self._pipeline = pipe
        self._load_error = err
        return pipe is not None

    def is_ready(self) -> bool:
        """Returns True if the underlying pipeline is loaded and ready for inference."""
        if self._pipeline is not None:
            return True
        # If not yet loaded, test if it can be loaded from cache
        if self.loader.is_cached(self.checkpoint):
            return self.load()
        return False

    def predict(self, text: str) -> PredictionResult:
        """
        Runs multilingual sentiment classification on the given text.
        Returns a standardized PredictionResult with full provenance metadata.
        """
        # Guard: invalid or empty inputs
        if text is None or not str(text).strip():
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Transformer Sentiment (Empty)",
                is_uncertain=True,
                metadata={
                    "backend": "transformer",
                    "checkpoint": self.checkpoint,
                    "reason": "Empty or whitespace-only input",
                },
            )

        clean_text = str(text).strip()

        # Ensure pipeline is loaded
        if self._pipeline is None:
            success = self.load()
            if not success or self._pipeline is None:
                return PredictionResult(
                    label="Uncertain",
                    confidence=0.0,
                    method="Transformer Sentiment (Unavailable)",
                    is_uncertain=True,
                    metadata={
                        "backend": "transformer",
                        "checkpoint": self.checkpoint,
                        "error": self._load_error or self.loader.get_last_error(self.checkpoint) or "Model not loaded",
                    },
                )

        t_start = time.perf_counter()
        try:
            # Run inference
            # Hugging Face sequence classification pipeline returns a list of dicts:
            # [{'label': 'positive', 'score': 0.89}, ...] or [[{'label': ...}]]
            raw_output = self._pipeline(clean_text)
            inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

            # Normalize output structure
            if isinstance(raw_output, list) and len(raw_output) > 0:
                if isinstance(raw_output[0], list):
                    candidates = raw_output[0]
                else:
                    candidates = raw_output
            else:
                candidates = []

            # Extract raw scores
            raw_scores: Dict[str, float] = {}
            for item in candidates:
                raw_lbl = item.get("label", "")
                mapped_lbl = self.label_map.get(raw_lbl, self.label_map.get(raw_lbl.lower(), raw_lbl))
                raw_scores[mapped_lbl] = float(item.get("score", 0.0))

            if not raw_scores:
                return PredictionResult(
                    label="Uncertain",
                    confidence=0.0,
                    method="Transformer Sentiment",
                    is_uncertain=True,
                    metadata={"backend": "transformer", "checkpoint": self.checkpoint, "error": "Empty predictions"},
                )

            # Best prediction
            best_label, raw_score = max(raw_scores.items(), key=lambda x: x[1])

            # Ensure label conforms to LinguaLabel SENTIMENT_LABELS
            if best_label not in SENTIMENT_LABELS:
                best_label = "Uncertain"

            # Check uncertainty boundary (< 0.60)
            is_uncertain = raw_score < self.uncertainty_threshold
            final_label = "Uncertain" if is_uncertain else best_label

            return PredictionResult(
                label=final_label,
                confidence=round(raw_score, 4),
                method=f"Transformer ({self.checkpoint.split('/')[-1]})",
                is_uncertain=is_uncertain,
                confidence_type="model_probability",
                metadata={
                    "backend": "transformer",
                    "actual_backend": "transformer",
                    "checkpoint": self.checkpoint,
                    "device": self.loader.get_device(self.device),
                    "raw_best_label": best_label,
                    "raw_softmax_score": round(raw_score, 4),
                    "calibrated": False,  # Explicitly documented: uncalibrated softmax probability
                    "all_scores": {k: round(v, 4) for k, v in raw_scores.items()},
                    "inference_time_ms": inference_time_ms,
                },
            )

        except Exception as e:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Transformer Sentiment (Error)",
                is_uncertain=True,
                metadata={
                    "backend": "transformer",
                    "checkpoint": self.checkpoint,
                    "error": str(e),
                },
            )


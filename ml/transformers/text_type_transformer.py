"""
Transformer-based Multilingual Text Type & Communicative Intent Predictor for LinguaLabel.

Leverages multilingual NLI / Zero-Shot sequence classification checkpoints
(e.g., `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` or XLM-RoBERTa XNLI).

Categorizes communicative intent into:
- Question
- Complaint
- Review
- General
- Greeting
- Request

And maps deterministically to the LinguaLabel UI schema:
- Statement
- Question
- Request / Instruction
- Greeting
- Other
- Uncertain
"""

import time
from typing import Any, Dict, List, Optional
from ml.base import BasePredictor, PredictionResult
from ml.model_config import (
    DEFAULT_TEXT_TYPE_CHECKPOINT,
    EXTENDED_TEXT_TYPE_CATEGORIES,
    EXTENDED_TO_UI_TEXT_TYPE_MAP,
    TEXT_TYPE_LABELS,
    VERIFIED_MODELS,
)
from ml.transformers.model_loader import get_model_loader


class TextTypeTransformer(BasePredictor):
    """
    Multilingual transformer predictor for Text Type and Communicative Intent.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_TEXT_TYPE_CHECKPOINT,
        candidate_labels: Optional[List[str]] = None,
        uncertainty_threshold: float = 0.60,
        device: str = "auto",
        hypothesis_template: str = "This text expresses a {}.",
        lazy_load: bool = True,
    ):
        self.checkpoint = checkpoint
        self.candidate_labels = candidate_labels or list(EXTENDED_TEXT_TYPE_CATEGORIES)
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device
        self.hypothesis_template = hypothesis_template
        self.loader = get_model_loader()
        self._pipeline = None
        self._load_error: Optional[str] = None

        self.model_meta = VERIFIED_MODELS.get(checkpoint)
        self.label_map = (
            self.model_meta.label_mapping
            if self.model_meta
            else EXTENDED_TO_UI_TEXT_TYPE_MAP
        )

        if not lazy_load:
            self.load()

    def load(self) -> bool:
        """Loads or retrieves cached zero-shot pipeline."""
        pipe, err = self.loader.load_zero_shot_pipeline(
            self.checkpoint, device=self.device
        )
        self._pipeline = pipe
        self._load_error = err
        return pipe is not None

    def is_ready(self) -> bool:
        if self._pipeline is not None:
            return True
        if self.loader.is_cached(self.checkpoint):
            return self.load()
        return False

    def predict(self, text: str) -> PredictionResult:
        """
        Runs multilingual text-type / intent classification.
        Returns a standardized PredictionResult with mapped UI labels and extended metadata.
        """
        if text is None or not str(text).strip():
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Transformer Text Type (Empty)",
                is_uncertain=True,
                metadata={
                    "backend": "transformer",
                    "checkpoint": self.checkpoint,
                    "reason": "Empty or whitespace-only input",
                },
            )

        clean_text = str(text).strip()

        if self._pipeline is None:
            success = self.load()
            if not success or self._pipeline is None:
                return PredictionResult(
                    label="Uncertain",
                    confidence=0.0,
                    method="Transformer Text Type (Unavailable)",
                    is_uncertain=True,
                    metadata={
                        "backend": "transformer",
                        "checkpoint": self.checkpoint,
                        "error": self._load_error or self.loader.get_last_error(self.checkpoint) or "Model not loaded",
                    },
                )

        t_start = time.perf_counter()
        try:
            # Run zero-shot inference
            # Output: {'sequence': '...', 'labels': ['Question', ...], 'scores': [0.82, ...]}
            res = self._pipeline(
                clean_text,
                candidate_labels=self.candidate_labels,
                hypothesis_template=self.hypothesis_template,
                multi_label=False,
            )
            inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

            predicted_labels = res.get("labels", [])
            scores = res.get("scores", [])

            if not predicted_labels or not scores:
                return PredictionResult(
                    label="Uncertain",
                    confidence=0.0,
                    method="Transformer Text Type",
                    is_uncertain=True,
                    metadata={"backend": "transformer", "checkpoint": self.checkpoint, "error": "Empty predictions"},
                )

            best_extended = predicted_labels[0]
            raw_score = float(scores[0])

            # Map to standard LinguaLabel UI category
            mapped_ui_label = self.label_map.get(best_extended, "Other")
            if mapped_ui_label not in TEXT_TYPE_LABELS:
                mapped_ui_label = "Other"

            all_scores = {
                lbl: round(float(sc), 4) for lbl, sc in zip(predicted_labels, scores)
            }

            # Check uncertainty threshold (< 0.60)
            is_uncertain = raw_score < self.uncertainty_threshold
            final_label = "Uncertain" if is_uncertain else mapped_ui_label

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
                    "extended_intent": best_extended,
                    "raw_softmax_score": round(raw_score, 4),
                    "calibrated": False,
                    "all_scores": all_scores,
                    "inference_time_ms": inference_time_ms,
                },
            )

        except Exception as e:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Transformer Text Type (Error)",
                is_uncertain=True,
                metadata={
                    "backend": "transformer",
                    "checkpoint": self.checkpoint,
                    "error": str(e),
                },
            )


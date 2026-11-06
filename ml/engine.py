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

from pathlib import Path
from typing import Any, Dict, Optional, Union
from ml.base import BasePredictor, PredictionResult
from ml.abuse_detector import AbuseDetector
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
        self._abuse_detector: Optional[AbuseDetector] = None
        self._custom_predictors: Dict[str, Any] = {}

        self._initialize_detectors()
        self._load_active_custom_models()

    def _load_active_custom_models(self) -> None:
        """Loads promoted custom models from trained_models/active_production.json if present."""
        self._custom_predictors.clear()
        try:
            from ml.training.versioning import ModelVersionRegistry
            from ml.training.trainer import TrainedModelPredictor
            root_dir = Path(__file__).resolve().parent.parent
            registry = ModelVersionRegistry(root_dir)
            active_map = registry.get_active_production_models()
            for task, version_id in active_map.items():
                version = registry.get_version(version_id)
                if version and Path(version.artifacts_path).exists():
                    self._custom_predictors[task] = TrainedModelPredictor(
                        artifacts_dir=Path(version.artifacts_path),
                        task=version.task,
                    )
        except Exception:
            pass

    def reload_active_models(self) -> None:
        """Forces reloading active production custom models."""
        self._load_active_custom_models()

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
            self._abuse_detector = AbuseDetector(
                uncertainty_threshold=self.uncertainty_threshold
            )
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
        if "language" in self._custom_predictors:
            try:
                return self._custom_predictors["language"].predict(text)
            except Exception:
                pass

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
        if "sentiment" in self._custom_predictors:
            try:
                return self._custom_predictors["sentiment"].predict(text)
            except Exception:
                pass

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

    def predict_complaint(self, text: str) -> PredictionResult:
        """
        Produces dedicated complaint detection (Part 11.4).
        Uses promoted complaint custom model if active, else falls back to text_type heuristics.
        """
        if not text or not str(text).strip():
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Input is empty or whitespace",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={"actual_backend": "none", "error": "empty_input"}
            )

        for k in ["complaint", "complaining"]:
            if k in self._custom_predictors:
                try:
                    res = self._custom_predictors[k].predict(text)
                    raw_lbl = str(res.label).strip().lower()
                    if raw_lbl in ("complaining", "complaint", "issue", "1", "true"):
                        norm_lbl = "Complaining"
                    elif raw_lbl in ("not complaining", "non-complaint", "0", "false"):
                        norm_lbl = "Not complaining"
                    elif raw_lbl == "uncertain":
                        norm_lbl = "Uncertain"
                    else:
                        norm_lbl = "Complaining" if "complain" in raw_lbl else "Not complaining"

                    return PredictionResult(
                        label=norm_lbl,
                        confidence=res.confidence,
                        method=res.method,
                        is_uncertain=res.is_uncertain,
                        confidence_type=res.confidence_type,
                        metadata=res.metadata,
                    )
                except Exception:
                    pass

        if not self.is_available() or not self._text_type_detector:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Unavailable",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={"actual_backend": "baseline", "error": "Detector uninitialized or engine unavailable"}
            )

        # Fallback to text_type detector baseline
        if self._text_type_detector:
            try:
                base_pred = self._text_type_detector.predict(text)
                if base_pred.label == "Complaining":
                    return PredictionResult(
                        label="Complaining",
                        confidence=base_pred.confidence,
                        method=base_pred.method,
                        is_uncertain=base_pred.is_uncertain,
                        confidence_type=base_pred.confidence_type,
                        metadata={**base_pred.metadata, "actual_backend": "baseline_detector"},
                    )
                else:
                    return PredictionResult(
                        label="Not complaining",
                        confidence=round(1.0 - base_pred.confidence if not base_pred.is_uncertain else 0.5, 4),
                        method="Baseline Text Type Heuristics",
                        is_uncertain=False,
                        confidence_type="heuristic",
                        metadata={"actual_backend": "baseline_detector"},
                    )
            except Exception:
                pass

        return PredictionResult(
            label="Not complaining",
            confidence=0.5,
            method="Fallback Default",
            is_uncertain=True,
            confidence_type="heuristic",
            metadata={"actual_backend": "fallback"},
        )

    def predict_text_type(self, text: str) -> PredictionResult:
        """
        Produces text-type / intent prediction using active backend.
        Integrates promoted complaint model with safe precedence (Part 11.4).
        """
        # Step 1: Check promoted complaint model for positive complaint signal
        has_promoted_complaint = any(k in self._custom_predictors for k in ("complaint", "complaining"))
        complaint_res = None
        if has_promoted_complaint:
            try:
                complaint_res = self.predict_complaint(text)
                if complaint_res.label == "Complaining" and not complaint_res.is_uncertain and complaint_res.confidence >= 0.60:
                    return PredictionResult(
                        label="Complaining",
                        confidence=complaint_res.confidence,
                        method=complaint_res.method,
                        is_uncertain=False,
                        confidence_type=complaint_res.confidence_type,
                        metadata={
                            **complaint_res.metadata,
                            "provenance": "promoted_complaint_model",
                        },
                    )
            except Exception:
                pass

        # Step 2: Use general text-type custom model if promoted
        general_pred = None
        if "text_type" in self._custom_predictors:
            try:
                general_pred = self._custom_predictors["text_type"].predict(text)
            except Exception:
                pass

        # Step 3: Transformer backend if active
        if general_pred is None and self.config.active_backend == MLBackend.TRANSFORMER:
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
                        general_pred = pred
            except Exception as e:
                if not self.config.enable_fallback_to_baseline:
                    return PredictionResult(
                        "Uncertain", 0.0, "Transformer Error",
                        is_uncertain=True,
                        confidence_type="heuristic",
                        metadata={"actual_backend": "transformer", "error": str(e)}
                    )

            if general_pred is None and self._text_type_detector:
                base_pred = self._text_type_detector.predict(text)
                base_pred.metadata["actual_backend"] = "fallback_baseline"
                base_pred.metadata["is_fallback"] = True
                base_pred.metadata["fallback_from"] = "transformer"
                general_pred = base_pred

        # Step 4: Baseline detector fallback
        if general_pred is None:
            if not self._text_type_detector:
                return PredictionResult("Uncertain", 0.0, "Unavailable", is_uncertain=True, confidence_type="heuristic", metadata={"actual_backend": "baseline"})
            try:
                general_pred = self._text_type_detector.predict(text)
                general_pred.metadata.setdefault("actual_backend", "baseline")
            except Exception as e:
                return PredictionResult("Uncertain", 0.0, "Fallback", is_uncertain=True, confidence_type="heuristic", metadata={"actual_backend": "baseline", "error": str(e)})

        # Step 5: Safe precedence resolution:
        # If the general text type model predicted Complaining, but the promoted specialized complaint model
        # explicitly predicted 'Not complaining' with high certainty (>= 0.70):
        # veto 'Complaining' to prevent false positive complaints on ordinary opinions.
        if (
            general_pred.label == "Complaining"
            and complaint_res is not None
            and complaint_res.label == "Not complaining"
            and not complaint_res.is_uncertain
            and complaint_res.confidence >= 0.70
        ):
            return PredictionResult(
                label="Statement",
                confidence=complaint_res.confidence,
                method="Specialized Complaint Veto -> Statement",
                is_uncertain=False,
                confidence_type=complaint_res.confidence_type,
                metadata={
                    **general_pred.metadata,
                    "veto_by": "promoted_complaint_model",
                    "original_label": "Complaining",
                },
            )

        return general_pred

    def predict_abuse_category(self, text: str) -> PredictionResult:
        """
        Produces abusive language, profanity, harassment, and threat detection category.
        Uses promoted abuse_category custom model if active, else baseline detector.
        """
        for k in ["abuse_category", "abuse"]:
            if k in self._custom_predictors:
                try:
                    return self._custom_predictors[k].predict(text)
                except Exception:
                    pass

        if not self.is_available() or not self._abuse_detector:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Unavailable",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={"actual_backend": "baseline", "error": "Detector uninitialized or engine unavailable"}
            )
        try:
            return self._abuse_detector.predict(text)
        except Exception as e:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Fallback",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={"actual_backend": "baseline", "error": str(e)}
            )

    def predict_abuse(self, text: str) -> PredictionResult:
        """Backward-compatible alias for predict_abuse_category."""
        return self.predict_abuse_category(text)

    def predict_abuse_severity(
        self,
        text: str,
        category_pred: Optional[PredictionResult] = None,
    ) -> PredictionResult:
        """
        Produces dedicated abuse severity inference (None, Mild, Moderate, Severe) (Part 11.4).
        Enforces two mandatory safety guardrails:
        1. 'Not abusive' category ALWAYS produces severity 'None'.
        2. A severity model can NEVER independently declare non-abusive content abusive.
        """
        if not text or not str(text).strip():
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Input is empty or whitespace",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={"actual_backend": "none", "error": "empty_input"}
            )

        if not self.is_available() or not self._abuse_detector:
            return PredictionResult(
                label="Uncertain",
                confidence=0.0,
                method="Unavailable",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={"actual_backend": "baseline", "error": "Detector uninitialized or engine unavailable"}
            )

        # Obtain category if not passed
        if category_pred is None:
            category_pred = self.predict_abuse_category(text)

        cat_label = str(category_pred.label).strip()

        # Guardrail 1: Non-abusive text strictly yields severity 'None'
        if cat_label == "Not abusive":
            return PredictionResult(
                label="None",
                confidence=max(category_pred.confidence, 0.95),
                method="Safety Guardrail (Not Abusive -> None)",
                is_uncertain=False,
                confidence_type="heuristic",
                metadata={
                    "actual_backend": "safety_guardrail",
                    "category": cat_label,
                    "guardrail_enforced": "not_abusive_implies_none",
                },
            )

        # Guardrail 2: Uncertain category defaults to 'None' with is_uncertain=True
        if category_pred.is_uncertain or cat_label == "Uncertain":
            return PredictionResult(
                label="None",
                confidence=0.0,
                method="Safety Guardrail (Uncertain Category)",
                is_uncertain=True,
                confidence_type="heuristic",
                metadata={
                    "actual_backend": "safety_guardrail",
                    "category": cat_label,
                    "guardrail_enforced": "uncertain_category_defaults_none",
                },
            )

        # For abusive categories: check for promoted abuse_severity custom model
        for k in ["abuse_severity", "severity"]:
            if k in self._custom_predictors:
                try:
                    sev_res = self._custom_predictors[k].predict(text)
                    raw_sev = str(sev_res.label).strip()
                    if raw_sev in ("None", "Mild", "Moderate", "Severe"):
                        effective_sev = "Mild" if raw_sev == "None" else raw_sev
                        return PredictionResult(
                            label=effective_sev,
                            confidence=sev_res.confidence,
                            method=sev_res.method,
                            is_uncertain=sev_res.is_uncertain,
                            confidence_type=sev_res.confidence_type,
                            metadata={
                                **sev_res.metadata,
                                "category": cat_label,
                                "raw_predicted_severity": raw_sev,
                            },
                        )
                except Exception:
                    pass

        # Fallback to baseline detector's rule-based severity logic
        if self._abuse_detector:
            try:
                base_abuse = self._abuse_detector.predict(text)
                base_sev = base_abuse.metadata.get("severity", "Moderate") if base_abuse.metadata else "Moderate"
                if base_sev not in ("None", "Mild", "Moderate", "Severe"):
                    base_sev = "Moderate"
                return PredictionResult(
                    label=base_sev,
                    confidence=base_abuse.confidence,
                    method=f"Baseline Severity ({base_abuse.method})",
                    is_uncertain=base_abuse.is_uncertain,
                    confidence_type=base_abuse.confidence_type,
                    metadata={
                        "actual_backend": base_abuse.metadata.get("actual_backend", "baseline"),
                        "category": cat_label,
                        "derived_from": "abuse_detector",
                    },
                )
            except Exception:
                pass

        return PredictionResult(
            label="Moderate",
            confidence=0.5,
            method="Default Abusive Severity",
            is_uncertain=True,
            confidence_type="heuristic",
            metadata={"actual_backend": "fallback", "category": cat_label},
        )

    def predict_all(self, text: str) -> Dict[str, PredictionResult]:
        """
        Computes advisory predictions across all 6 classification dimensions (Part 11.4):
        - language
        - sentiment
        - text_type
        - complaint
        - abuse_category
        - abuse_severity
        - abuse (backward-compatible alias for abuse_category)
        """
        lang_res = self.predict_language(text)
        sent_res = self.predict_sentiment(text)
        tt_res = self.predict_text_type(text)
        complaint_res = self.predict_complaint(text)
        abuse_cat_res = self.predict_abuse_category(text)
        abuse_sev_res = self.predict_abuse_severity(text, category_pred=abuse_cat_res)

        return {
            "language": lang_res,
            "sentiment": sent_res,
            "text_type": tt_res,
            "complaint": complaint_res,
            "abuse_category": abuse_cat_res,
            "abuse_severity": abuse_sev_res,
            "abuse": abuse_cat_res,  # Backward-compatible alias
        }

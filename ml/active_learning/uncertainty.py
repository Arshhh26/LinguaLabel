"""
Uncertainty Sampling Engine for LinguaLabel Active Learning.

Implements rigorous, mathematically sound uncertainty sampling strategies:
1. Lowest-Confidence Sampling:
   - Evaluates prediction confidence c in [0, 1].
   - Uncertainty = 1.0 - c.
   - Treats missing, invalid, or zero confidence as maximal uncertainty (1.0).

2. Margin Sampling:
   - Computes difference between the top two class scores: Delta = p_(1) - p_(2).
   - Margin Uncertainty = 1.0 - Delta.
   - Requires >= 2 class scores from a valid distribution.

3. Shannon Entropy Sampling:
   - Computes normalized Shannon entropy: H_norm(p) = - sum(p_i * ln(p_i)) / ln(K).
   - STRICT CONSTRAINT: Computed ONLY when genuine class probability distributions
     are available (confidence_type in ['model_probability', 'calibrated_probability']
     and sum(p_i) ~ 1.0).
   - Heuristic scores are NEVER treated as probabilities and NEVER used for entropy calculation.

All results record provenance, confidence type, calculation validity, and explicit explanatory notes.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional
from ml.base import PredictionResult


@dataclass
class TaskUncertainty:
    """Detailed uncertainty metrics for a single prediction task."""
    task: str
    label: str
    confidence: float
    confidence_type: str
    is_uncertain: bool  # True if below engine uncertainty threshold (e.g. < 0.60)
    lowest_confidence_uncertainty: float
    margin: Optional[float] = None
    margin_uncertainty: Optional[float] = None
    margin_supported: bool = False
    entropy: Optional[float] = None
    entropy_supported: bool = False
    class_distribution_available: bool = False
    top_score: float = 0.0
    second_score: Optional[float] = None
    all_scores: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "confidence_type": self.confidence_type,
            "is_uncertain": self.is_uncertain,
            "lowest_confidence_uncertainty": round(float(self.lowest_confidence_uncertainty), 4),
            "margin": round(float(self.margin), 4) if self.margin is not None else None,
            "margin_uncertainty": round(float(self.margin_uncertainty), 4) if self.margin_uncertainty is not None else None,
            "margin_supported": self.margin_supported,
            "entropy": round(float(self.entropy), 4) if self.entropy is not None else None,
            "entropy_supported": self.entropy_supported,
            "class_distribution_available": self.class_distribution_available,
            "top_score": round(float(self.top_score), 4),
            "second_score": round(float(self.second_score), 4) if self.second_score is not None else None,
            "notes": self.notes,
        }


@dataclass
class RecordUncertainty:
    """Aggregate uncertainty assessment across all prediction tasks for a record."""
    record_id: str
    tasks: Dict[str, TaskUncertainty] = field(default_factory=dict)
    composite_uncertainty: float = 0.0
    strategy: str = "lowest_confidence"
    has_uncertain_prediction: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "composite_uncertainty": round(float(self.composite_uncertainty), 4),
            "strategy": self.strategy,
            "has_uncertain_prediction": self.has_uncertain_prediction,
            "notes": self.notes,
        }


def compute_task_uncertainty(
    task: str,
    pred: Optional[PredictionResult],
    uncertainty_threshold: float = 0.60,
) -> TaskUncertainty:
    """
    Computes rigorous uncertainty measures for a single task prediction.
    Enforces that heuristic scores are never used for entropy calculations.
    """
    if pred is None:
        return TaskUncertainty(
            task=task,
            label="Missing",
            confidence=0.0,
            confidence_type="missing",
            is_uncertain=True,
            lowest_confidence_uncertainty=1.0,
            notes="No prediction available; maximal uncertainty assigned.",
        )

    conf = float(pred.confidence) if pred.confidence is not None else 0.0
    conf = max(0.0, min(1.0, conf))
    conf_type = getattr(pred, "confidence_type", "heuristic") or "heuristic"
    is_unc = bool(getattr(pred, "is_uncertain", False) or conf < uncertainty_threshold)

    # 1. Lowest confidence uncertainty: 1.0 - conf
    lc_uncertainty = 1.0 - conf

    # 2. Extract class scores if available
    metadata = getattr(pred, "metadata", {}) or {}
    all_scores_raw = metadata.get("all_scores", {})
    all_scores: Dict[str, float] = {}

    if isinstance(all_scores_raw, dict):
        for k, v in all_scores_raw.items():
            try:
                val = float(v)
                if not math.isnan(val) and not math.isinf(val):
                    all_scores[str(k)] = val
            except (ValueError, TypeError):
                continue

    # Margin Sampling
    margin: Optional[float] = None
    margin_uncertainty: Optional[float] = None
    margin_supported = False
    second_score: Optional[float] = None
    top_score = conf

    sorted_scores = sorted(all_scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]
        raw_margin = top_score - second_score
        margin = max(0.0, min(1.0, raw_margin))
        margin_uncertainty = 1.0 - margin
        margin_supported = True

    # Shannon Entropy Sampling
    # STRICT RULE: Must be a genuine probability distribution, NEVER a heuristic score
    entropy: Optional[float] = None
    entropy_supported = False
    class_dist_avail = False
    notes_list: List[str] = []

    is_probabilistic = conf_type in ("model_probability", "calibrated_probability")

    if is_probabilistic and len(sorted_scores) >= 2:
        score_sum = sum(sorted_scores)
        # Check if scores form an approximate probability distribution (sum to ~1.0)
        if 0.90 <= score_sum <= 1.10 and all(s >= 0 for s in sorted_scores):
            class_dist_avail = True
            # Normalize probabilities to sum exactly to 1.0 for numerical safety
            probs = [s / score_sum for s in sorted_scores if s > 0]
            k = len(probs)
            if k >= 2:
                raw_entropy = -sum(p * math.log(p) for p in probs)
                max_entropy = math.log(k)
                if max_entropy > 0:
                    norm_entropy = raw_entropy / max_entropy
                    entropy = max(0.0, min(1.0, norm_entropy))
                    entropy_supported = True
        else:
            notes_list.append("Probability distribution incomplete or unnormalized; entropy omitted.")
    elif not is_probabilistic:
        notes_list.append("Entropy calculation omitted: heuristic confidence is not a valid probability distribution.")
    else:
        notes_list.append("Entropy calculation omitted: fewer than 2 class scores available.")

    if is_unc:
        notes_list.append(f"Confidence {conf:.1%} is below safety threshold {uncertainty_threshold:.1%}.")

    return TaskUncertainty(
        task=task,
        label=pred.label,
        confidence=conf,
        confidence_type=conf_type,
        is_uncertain=is_unc,
        lowest_confidence_uncertainty=lc_uncertainty,
        margin=margin,
        margin_uncertainty=margin_uncertainty,
        margin_supported=margin_supported,
        entropy=entropy,
        entropy_supported=entropy_supported,
        class_distribution_available=class_dist_avail,
        top_score=top_score,
        second_score=second_score,
        all_scores=all_scores,
        notes="; ".join(notes_list) if notes_list else "Prediction evaluated successfully.",
    )


def compute_record_uncertainty(
    record_id: str,
    predictions: Dict[str, PredictionResult],
    strategy: str = "lowest_confidence",
    task_weights: Optional[Dict[str, float]] = None,
    uncertainty_threshold: float = 0.60,
) -> RecordUncertainty:
    """
    Computes multi-task uncertainty for a record across specified sampling strategies:
    - 'lowest_confidence': 1.0 - conf
    - 'margin': 1.0 - margin (falls back to lowest_confidence if margin unsupported)
    - 'entropy': normalized entropy (falls back to lowest_confidence if entropy unsupported)
    """
    weights = task_weights or {"language": 0.20, "sentiment": 0.40, "text_type": 0.40}
    task_metrics: Dict[str, TaskUncertainty] = {}
    any_uncertain = False
    notes: List[str] = []

    # Evaluate each task
    for task_name in ["language", "sentiment", "text_type"]:
        p = predictions.get(task_name)
        tu = compute_task_uncertainty(task_name, p, uncertainty_threshold)
        task_metrics[task_name] = tu
        if tu.is_uncertain:
            any_uncertain = True
            notes.append(f"{task_name.capitalize()} uncertain ({tu.confidence:.0%})")

    # Calculate strategy score per task
    strategy_scores: List[float] = []
    task_weight_list: List[float] = []

    for task_name, tu in task_metrics.items():
        w = weights.get(task_name, 0.33)
        if strategy == "margin" and tu.margin_supported and tu.margin_uncertainty is not None:
            score = tu.margin_uncertainty
        elif strategy == "entropy" and tu.entropy_supported and tu.entropy is not None:
            score = tu.entropy
        else:
            # Default or fallback to lowest confidence uncertainty
            score = tu.lowest_confidence_uncertainty
            if strategy in ("margin", "entropy") and not (
                (strategy == "margin" and tu.margin_supported) or (strategy == "entropy" and tu.entropy_supported)
            ):
                notes.append(f"{strategy.capitalize()} unsupported for {task_name}; defaulted to lowest-confidence.")

        strategy_scores.append(score)
        task_weight_list.append(w)

    total_weight = sum(task_weight_list) or 1.0
    composite = sum(s * w for s, w in zip(strategy_scores, task_weight_list)) / total_weight
    composite = max(0.0, min(1.0, composite))

    return RecordUncertainty(
        record_id=str(record_id),
        tasks=task_metrics,
        composite_uncertainty=composite,
        strategy=strategy,
        has_uncertain_prediction=any_uncertain,
        notes=notes,
    )


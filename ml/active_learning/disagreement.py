"""
Model Disagreement Detection Engine for LinguaLabel Active Learning.

Compares paired predictions between Baseline (heuristics/n-grams) and
Transformer backends (XLM-RoBERTa / mDeBERTa-v3) to identify cross-model conflict.

Strict Integrity Rules:
1. Provenance Verification: Checks actual_backend attribute on both predictions.
2. Fallback Exclusion: Predictions marked as 'fallback_baseline' are NEVER counted
   as genuine transformer outputs. Fallbacks are recorded as 'unavailable', not disagreement.
3. Missing Prediction Safety: Missing or erroneous predictions are not treated as disagreement.
4. Transparent Explanations: Disagreements generate clear, human-readable rationales
   stating each model's predicted class and confidence score.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from ml.base import PredictionResult


@dataclass
class TaskDisagreement:
    """Detailed comparison between Baseline and Transformer for a single task."""
    task: str
    baseline_label: str
    transformer_label: Optional[str]
    baseline_confidence: float
    transformer_confidence: Optional[float]
    baseline_backend: str
    transformer_backend: str
    is_genuine_comparison: bool  # True ONLY if transformer genuinely ran and did not fall back
    is_conflict: bool
    is_fallback: bool
    explanation: str
    baseline_pred: Optional[PredictionResult] = None
    transformer_pred: Optional[PredictionResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "baseline_label": self.baseline_label,
            "transformer_label": self.transformer_label,
            "baseline_confidence": round(float(self.baseline_confidence), 4),
            "transformer_confidence": round(float(self.transformer_confidence), 4) if self.transformer_confidence is not None else None,
            "baseline_backend": self.baseline_backend,
            "transformer_backend": self.transformer_backend,
            "is_genuine_comparison": self.is_genuine_comparison,
            "is_conflict": self.is_conflict,
            "is_fallback": self.is_fallback,
            "explanation": self.explanation,
        }


@dataclass
class RecordDisagreement:
    """Aggregate model disagreement assessment for a single record."""
    record_id: str
    tasks: Dict[str, TaskDisagreement] = field(default_factory=dict)
    has_disagreement: bool = False
    disagreement_count: int = 0
    comparable_task_count: int = 0
    disagreement_score: float = 0.0  # fraction of comparable tasks with conflict
    disagreeing_tasks: List[str] = field(default_factory=list)
    has_fallback: bool = False
    summary_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "has_disagreement": self.has_disagreement,
            "disagreement_count": self.disagreement_count,
            "comparable_task_count": self.comparable_task_count,
            "disagreement_score": round(float(self.disagreement_score), 4),
            "disagreeing_tasks": self.disagreeing_tasks,
            "has_fallback": self.has_fallback,
            "summary_explanation": self.summary_explanation,
        }


def normalize_label(label: Optional[str]) -> str:
    """Normalizes class labels for robust equality comparison."""
    if not label:
        return ""
    return str(label).strip().lower()


def evaluate_task_disagreement(
    task: str,
    baseline_pred: Optional[PredictionResult],
    transformer_pred: Optional[PredictionResult],
) -> TaskDisagreement:
    """
    Evaluates disagreement between baseline and transformer predictions for a task.
    Enforces strict exclusion of fallback predictions from disagreement counts.
    """
    base_lbl = baseline_pred.label if baseline_pred else "Missing"
    base_conf = float(baseline_pred.confidence) if baseline_pred and baseline_pred.confidence is not None else 0.0
    base_backend = getattr(baseline_pred, "actual_backend", "baseline") if baseline_pred else "missing"

    if transformer_pred is None:
        return TaskDisagreement(
            task=task,
            baseline_label=base_lbl,
            transformer_label=None,
            baseline_confidence=base_conf,
            transformer_confidence=None,
            baseline_backend=base_backend,
            transformer_backend="unavailable",
            is_genuine_comparison=False,
            is_conflict=False,
            is_fallback=False,
            explanation="Transformer prediction unavailable; disagreement cannot be evaluated.",
            baseline_pred=baseline_pred,
            transformer_pred=None,
        )

    trans_backend = getattr(transformer_pred, "actual_backend", "baseline")
    is_fallback = trans_backend == "fallback_baseline" or bool(
        transformer_pred.metadata.get("is_fallback", False)
    )
    is_error = "error" in transformer_pred.metadata

    # Check if this is a genuine transformer prediction
    if is_fallback or is_error or trans_backend != "transformer":
        return TaskDisagreement(
            task=task,
            baseline_label=base_lbl,
            transformer_label=transformer_pred.label,
            baseline_confidence=base_conf,
            transformer_confidence=float(transformer_pred.confidence) if transformer_pred.confidence is not None else 0.0,
            baseline_backend=base_backend,
            transformer_backend="fallback_baseline" if is_fallback else trans_backend,
            is_genuine_comparison=False,
            is_conflict=False,
            is_fallback=True,
            explanation=f"Transformer prediction fell back to baseline ({transformer_pred.metadata.get('error', 'fallback active')}); not counted as genuine disagreement.",
            baseline_pred=baseline_pred,
            transformer_pred=transformer_pred,
        )

    trans_lbl = transformer_pred.label
    trans_conf = float(transformer_pred.confidence) if transformer_pred.confidence is not None else 0.0

    # Compare normalized labels
    norm_base = normalize_label(base_lbl)
    norm_trans = normalize_label(trans_lbl)

    is_conflict = norm_base != norm_trans
    if is_conflict:
        c_lbl = "Heuristic" if getattr(baseline_pred, "confidence_type", "") == "heuristic" else "Prob"
        t_lbl = "Prob" if getattr(transformer_pred, "confidence_type", "") == "model_probability" else "Score"
        expl = (
            f"Conflict on {task.replace('_', ' ')}: Baseline predicted '{base_lbl}' ({base_conf:.0%} {c_lbl}), "
            f"whereas Transformer predicted '{trans_lbl}' ({trans_conf:.0%} {t_lbl})."
        )
    else:
        expl = f"Both models agree on '{base_lbl}'."

    return TaskDisagreement(
        task=task,
        baseline_label=base_lbl,
        transformer_label=trans_lbl,
        baseline_confidence=base_conf,
        transformer_confidence=trans_conf,
        baseline_backend=base_backend,
        transformer_backend="transformer",
        is_genuine_comparison=True,
        is_conflict=is_conflict,
        is_fallback=False,
        explanation=expl,
        baseline_pred=baseline_pred,
        transformer_pred=transformer_pred,
    )


def compute_record_disagreement(
    record_id: str,
    baseline_predictions: Dict[str, PredictionResult],
    transformer_predictions: Dict[str, PredictionResult],
    tasks: Optional[List[str]] = None,
) -> RecordDisagreement:
    """
    Computes multi-task model disagreement for a single record.
    Evaluates sentiment and text_type by default (and language if available).
    """
    eval_tasks = tasks or ["sentiment", "text_type"]
    task_disagreements: Dict[str, TaskDisagreement] = {}
    conflicts: List[str] = []
    comparable_count = 0
    has_fallback = False

    for task_name in eval_tasks:
        b_pred = baseline_predictions.get(task_name)
        t_pred = transformer_predictions.get(task_name)
        td = evaluate_task_disagreement(task_name, b_pred, t_pred)
        task_disagreements[task_name] = td

        if td.is_fallback:
            has_fallback = True

        if td.is_genuine_comparison:
            comparable_count += 1
            if td.is_conflict:
                conflicts.append(task_name)

    disagree_count = len(conflicts)
    score = (disagree_count / comparable_count) if comparable_count > 0 else 0.0

    if disagree_count > 0:
        summary = f"Disagreement on {disagree_count} task(s): {', '.join(t.replace('_', ' ') for t in conflicts)}."
    elif comparable_count > 0:
        summary = "Models agree across all evaluated tasks."
    elif has_fallback:
        summary = "Transformer unavailable (fell back to baseline); no cross-model comparison possible."
    else:
        summary = "No comparable predictions available."

    return RecordDisagreement(
        record_id=str(record_id),
        tasks=task_disagreements,
        has_disagreement=(disagree_count > 0),
        disagreement_count=disagree_count,
        comparable_task_count=comparable_count,
        disagreement_score=score,
        disagreeing_tasks=conflicts,
        has_fallback=has_fallback,
        summary_explanation=summary,
    )


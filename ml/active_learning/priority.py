"""
Intelligent Priority Scoring Engine for LinguaLabel Active Learning.

Integrates multiple human-in-the-loop prioritization signals:
1. Prediction Uncertainty (lowest confidence, small margin, or high entropy).
2. Cross-Model Disagreement (genuine conflict between baseline and transformer).
3. Human Review Status (unreviewed vs reviewed vs triage states).
4. Ambiguity / Needs-Second-Review flags.
5. Incomplete Annotation fields (missing schema values).

Prioritization Hierarchy:
- Tier 1: Unreviewed records with genuine model disagreement and high uncertainty.
- Tier 2: Unreviewed records with uncertain predictions (< 0.60 safety threshold).
- Tier 3: Incomplete records missing required annotations.
- Tier 4: Ambiguous records flagged for a second review.
- Demoted: Already reviewed records ('Reviewed — looks correct') receive negligible priority.

Every priority calculation provides fully traceable, human-readable explanations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from ml.active_learning.disagreement import RecordDisagreement
from ml.active_learning.uncertainty import RecordUncertainty


@dataclass
class PriorityWeights:
    """Configurable weights for the multi-criteria prioritization function."""
    weight_disagreement: float = 0.35
    weight_uncertainty: float = 0.35
    weight_missing: float = 0.15
    weight_ambiguous: float = 0.15
    completed_discount: float = 0.05  # Multiplier for already reviewed records


@dataclass
class PriorityRecord:
    """Complete prioritization assessment for a single dataset item."""
    record_id: str
    text: str
    priority_score: float  # [0.0, 1.0]
    priority_level: str  # 'High', 'Medium', 'Low'
    explanations: List[str] = field(default_factory=list)
    primary_reason: str = ""
    uncertainty_score: float = 0.0
    disagreement_score: float = 0.0
    missing_fields: List[str] = field(default_factory=list)
    human_status: str = "Pending"
    is_reviewed: bool = False
    is_ambiguous: bool = False
    uncertainty_report: Optional[RecordUncertainty] = None
    disagreement_report: Optional[RecordDisagreement] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "priority_score": round(float(self.priority_score), 4),
            "priority_level": self.priority_level,
            "primary_reason": self.primary_reason,
            "explanations": self.explanations,
            "uncertainty_score": round(float(self.uncertainty_score), 4),
            "disagreement_score": round(float(self.disagreement_score), 4),
            "missing_fields": self.missing_fields,
            "human_status": self.human_status,
            "is_reviewed": self.is_reviewed,
            "is_ambiguous": self.is_ambiguous,
        }


def compute_priority_record(
    record_id: str,
    text: str,
    row_data: Dict[str, Any],
    uncertainty_report: RecordUncertainty,
    disagreement_report: Optional[RecordDisagreement] = None,
    weights: Optional[PriorityWeights] = None,
    include_completed: bool = False,
) -> PriorityRecord:
    """
    Computes an explainable, multi-factor priority score for a record.
    """
    w = weights or PriorityWeights()
    explanations: List[str] = []

    # 1. Review status inspection
    raw_status = str(row_data.get("review_status", "")).strip()
    is_reviewed = raw_status == "Reviewed — looks correct"
    is_ambiguous = raw_status in ("Needs second review", "Ambiguous")
    human_status = raw_status if raw_status and raw_status != "— Select —" else "Pending"

    # 2. Missing fields inspection
    required_cols = ["language", "sentiment", "text_type", "review_status"]
    missing_fields: List[str] = []
    for col in required_cols:
        val = str(row_data.get(col, "")).strip()
        if not val or val == "— Select —":
            missing_fields.append(col)

    missing_score = len(missing_fields) / len(required_cols)
    if missing_fields:
        explanations.append(f"Incomplete annotation: missing {', '.join(missing_fields)}.")

    # 3. Disagreement inspection
    disag_score = 0.0
    if disagreement_report:
        disag_score = disagreement_report.disagreement_score
        if disagreement_report.has_disagreement:
            for task_name in disagreement_report.disagreeing_tasks:
                td = disagreement_report.tasks.get(task_name)
                if td:
                    explanations.append(f"Model conflict on {task_name}: Baseline '{td.baseline_label}' vs Transformer '{td.transformer_label}'.")

    # 4. Uncertainty inspection
    unc_score = uncertainty_report.composite_uncertainty
    if uncertainty_report.has_uncertain_prediction:
        for t_name, tu in uncertainty_report.tasks.items():
            if tu.is_uncertain:
                explanations.append(f"High uncertainty on {t_name}: confidence {tu.confidence:.0%} is below safety threshold.")
            if tu.margin_supported and tu.margin is not None and tu.margin < 0.20:
                explanations.append(f"Low prediction margin on {t_name}: difference between top two classes is only {tu.margin:.2f}.")

    # 5. Ambiguity score
    amb_score = 1.0 if is_ambiguous else 0.0
    if is_ambiguous:
        explanations.append(f"Previously flagged for review ({human_status}).")

    # 6. Compute raw weighted composite score
    total_w = w.weight_disagreement + w.weight_uncertainty + w.weight_missing + w.weight_ambiguous
    if total_w <= 0:
        total_w = 1.0

    raw_composite = (
        (w.weight_disagreement * disag_score) +
        (w.weight_uncertainty * unc_score) +
        (w.weight_missing * missing_score) +
        (w.weight_ambiguous * amb_score)
    ) / total_w

    raw_composite = max(0.0, min(1.0, raw_composite))

    # 7. Apply review status discount
    final_score = raw_composite
    if is_reviewed and not include_completed:
        final_score = raw_composite * w.completed_discount
        explanations.append("Demoted: record was already verified by human annotator ('Reviewed — looks correct').")

    # 8. Determine categorical priority level
    # Rule 1: Unreviewed records with model disagreement or ambiguity are High Priority
    if (disag_score > 0 and not is_reviewed) or (is_ambiguous and not is_reviewed) or final_score >= 0.55:
        level = "High"
    elif final_score >= 0.30:
        level = "Medium"
    else:
        level = "Low"

    # 9. Establish concise primary reason
    if disag_score > 0 and not is_reviewed:
        primary_reason = f"Model Disagreement ({disagreement_report.disagreement_count} task(s))" if disagreement_report else "Model Disagreement"
    elif uncertainty_report.has_uncertain_prediction and not is_reviewed:
        primary_reason = "Uncertain Prediction (< 60% confidence)"
    elif is_ambiguous:
        primary_reason = f"Flagged for Review ({human_status})"
    elif missing_fields and not is_reviewed:
        primary_reason = f"Missing Annotations ({len(missing_fields)} field(s))"
    elif is_reviewed:
        primary_reason = "Already Reviewed"
    else:
        primary_reason = "Routine Annotation"

    return PriorityRecord(
        record_id=str(record_id),
        text=text,
        priority_score=final_score,
        priority_level=level,
        explanations=explanations,
        primary_reason=primary_reason,
        uncertainty_score=unc_score,
        disagreement_score=disag_score,
        missing_fields=missing_fields,
        human_status=human_status,
        is_reviewed=is_reviewed,
        is_ambiguous=is_ambiguous,
        uncertainty_report=uncertainty_report,
        disagreement_report=disagreement_report,
    )

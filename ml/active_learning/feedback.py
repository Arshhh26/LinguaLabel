"""
Human Feedback Tracking and Analytics Engine for LinguaLabel.

Captures, persists, and analyzes human annotator corrections to model suggestions.
Maintains strict dataset isolation, historical deduplication, and non-destructive persistence.
"""

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class FeedbackRecord:
    """Individual human correction or confirmation of an advisory model prediction."""
    feedback_id: str
    dataset_key: str
    record_id: str
    task: str  # 'language', 'sentiment', 'text_type'
    original_text: str
    model_predicted_label: str
    human_corrected_label: str
    is_correction: bool  # True if human label differs from model suggestion
    model_confidence: float
    confidence_type: str  # 'heuristic', 'model_probability', 'calibrated_probability'
    actual_backend: str  # 'baseline', 'transformer', 'fallback_baseline'
    model_checkpoint: Optional[str]
    timestamp: str  # ISO 8601 UTC
    annotator_notes: str = ""
    was_model_uncertain: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_feedback_store_dir(root_dir: Path) -> Path:
    """Returns the dedicated feedback storage directory, ensuring it exists."""
    p = root_dir / "active_learning_feedback"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_feedback_store_path(root_dir: Path, dataset_key: Optional[str] = None) -> Path:
    """Returns the JSONL store path."""
    d = get_feedback_store_dir(root_dir)
    if dataset_key and dataset_key != "all":
        # Safe filename
        safe_key = "".join(c for c in dataset_key if c.isalnum() or c in ("-", "_"))
        return d / f"feedback_{safe_key}.jsonl"
    return d / "feedback_store.jsonl"


def generate_feedback_dedup_hash(
    dataset_key: str,
    record_id: str,
    task: str,
    model_pred: str,
    human_label: str,
    backend: str,
) -> str:
    """Computes a deterministic hash to prevent duplicate entries on repeated saves."""
    key = f"{dataset_key}|{record_id}|{task}|{model_pred}|{human_label}|{backend}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class FeedbackManager:
    """
    Manages persistence and retrieval of human corrections.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.store_dir = get_feedback_store_dir(root_dir)

    def record_feedback(
        self,
        dataset_key: str,
        record_id: str,
        task: str,
        original_text: str,
        model_prediction: Any,  # PredictionResult or dict
        human_label: str,
        annotator_notes: str = "",
    ) -> Optional[FeedbackRecord]:
        """
        Creates and stores a feedback record if a model prediction was present and evaluated.
        Avoids duplicate entries if the exact same correction was already saved.
        """
        if not model_prediction or not human_label:
            return None

        # Extract attributes from PredictionResult or dict
        if hasattr(model_prediction, "label"):
            m_label = str(model_prediction.label)
            m_conf = float(model_prediction.confidence) if model_prediction.confidence is not None else 0.0
            m_type = str(getattr(model_prediction, "confidence_type", "heuristic"))
            m_backend = str(getattr(model_prediction, "actual_backend", "baseline"))
            m_uncertain = bool(getattr(model_prediction, "is_uncertain", False))
            meta = getattr(model_prediction, "metadata", {}) or {}
            m_checkpoint = meta.get("checkpoint")
        else:
            m_label = str(model_prediction.get("label", ""))
            m_conf = float(model_prediction.get("confidence", 0.0))
            m_type = str(model_prediction.get("confidence_type", "heuristic"))
            m_backend = str(model_prediction.get("actual_backend", "baseline"))
            m_uncertain = bool(model_prediction.get("is_uncertain", False))
            m_checkpoint = model_prediction.get("metadata", {}).get("checkpoint")

        h_label = str(human_label).strip()
        if not h_label or h_label == "— Select —":
            return None

        is_correction = (m_label.lower() != h_label.lower())
        now_utc = datetime.now(timezone.utc).isoformat()

        dedup_hash = generate_feedback_dedup_hash(
            dataset_key=dataset_key,
            record_id=record_id,
            task=task,
            model_pred=m_label,
            human_label=h_label,
            backend=m_backend,
        )

        fb_id = f"fb_{dedup_hash}"

        # Check existing records to prevent redundant entries on repeated saves
        existing = self.load_feedback_records(dataset_key)
        for ex in existing:
            if ex.feedback_id == fb_id:
                # Already captured this exact correction state
                return ex

        record = FeedbackRecord(
            feedback_id=fb_id,
            dataset_key=dataset_key,
            record_id=str(record_id),
            task=task,
            original_text=original_text,
            model_predicted_label=m_label,
            human_corrected_label=h_label,
            is_correction=is_correction,
            model_confidence=m_conf,
            confidence_type=m_type,
            actual_backend=m_backend,
            model_checkpoint=m_checkpoint,
            timestamp=now_utc,
            annotator_notes=annotator_notes,
            was_model_uncertain=m_uncertain,
        )

        # Append to dataset-specific and global feedback store
        self._append_to_file(get_feedback_store_path(self.root_dir, dataset_key), record)
        if dataset_key != "all":
            self._append_to_file(get_feedback_store_path(self.root_dir, "all"), record)

        return record

    def _append_to_file(self, file_path: Path, record: FeedbackRecord) -> None:
        """Appends a feedback record to a JSONL file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def load_feedback_records(self, dataset_key: Optional[str] = None) -> List[FeedbackRecord]:
        """Loads all feedback records for a given dataset (or all datasets)."""
        target_path = get_feedback_store_path(self.root_dir, dataset_key)
        if not target_path.exists():
            return []

        records = []
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        records.append(FeedbackRecord(**d))
                    except Exception:
                        continue
        except Exception:
            return []
        return records

    def get_all_feedback(self) -> List[FeedbackRecord]:
        """Returns all feedback records across datasets."""
        all_recs = self.load_feedback_records(dataset_key="all")
        if not all_recs:
            all_recs = self.load_feedback_records(dataset_key=None)
        return all_recs

    def export_feedback_csv(self, records: List[FeedbackRecord]) -> str:
        """Exports feedback records to CSV string with utf-8-sig encoding support."""
        if not records:
            return ""

        output = []
        headers = [
            "feedback_id", "dataset_key", "record_id", "task", "original_text",
            "model_predicted_label", "human_corrected_label", "is_correction",
            "model_confidence", "confidence_type", "actual_backend",
            "model_checkpoint", "timestamp", "annotator_notes", "was_model_uncertain"
        ]
        
        csv_buffer = []
        # Write header
        csv_buffer.append(",".join(f'"{h}"' for h in headers))
        for r in records:
            row = [
                r.feedback_id,
                r.dataset_key,
                r.record_id,
                r.task,
                r.original_text.replace('"', '""'),
                r.model_predicted_label,
                r.human_corrected_label,
                str(r.is_correction),
                f"{r.model_confidence:.4f}",
                r.confidence_type,
                r.actual_backend,
                r.model_checkpoint or "",
                r.timestamp,
                r.annotator_notes.replace('"', '""'),
                str(r.was_model_uncertain),
            ]
            csv_buffer.append(",".join(f'"{c}"' for c in row))

        return "\n".join(csv_buffer)


def compute_feedback_analytics(records: List[FeedbackRecord]) -> Dict[str, Any]:
    """
    Computes rigorous analytical metrics from recorded human corrections:
    - Most frequently corrected classes.
    - Baseline vs Transformer correction counts.
    - Error confusion transitions (e.g. Model: Negative -> Human: Neutral).
    - Language-wise corrections.
    - Uncertain predictions reviewed & corrected.
    """
    total = len(records)
    if total == 0:
        return {
            "total_feedback_count": 0,
            "correction_count": 0,
            "agreement_count": 0,
            "baseline_corrections": 0,
            "transformer_corrections": 0,
            "top_corrected_classes": [],
            "error_patterns": [],
            "language_breakdown": {},
            "uncertain_reviewed_count": 0,
            "uncertain_corrected_count": 0,
            "disagreements_resolved": 0,
        }

    corrections = [r for r in records if r.is_correction]
    agreements = [r for r in records if not r.is_correction]

    # Corrections by backend
    base_corrs = sum(1 for r in corrections if r.actual_backend in ("baseline", "fallback_baseline"))
    trans_corrs = sum(1 for r in corrections if r.actual_backend == "transformer")

    # Error patterns: (predicted -> corrected)
    patterns: Dict[str, int] = {}
    class_corrs: Dict[str, int] = {}
    for r in corrections:
        pair = f"'{r.model_predicted_label}' → '{r.human_corrected_label}' ({r.task})"
        patterns[pair] = patterns.get(pair, 0) + 1
        class_corrs[r.model_predicted_label] = class_corrs.get(r.model_predicted_label, 0) + 1

    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:8]
    sorted_class_corrs = sorted(class_corrs.items(), key=lambda x: x[1], reverse=True)[:6]

    # Language breakdown (if task == 'language' or inference on text)
    lang_corrs: Dict[str, int] = {}
    for r in corrections:
        t = r.task
        lang_corrs[t] = lang_corrs.get(t, 0) + 1

    # Uncertain items
    unc_reviewed = sum(1 for r in records if r.was_model_uncertain)
    unc_corrected = sum(1 for r in corrections if r.was_model_uncertain)

    return {
        "total_feedback_count": total,
        "correction_count": len(corrections),
        "agreement_count": len(agreements),
        "baseline_corrections": base_corrs,
        "transformer_corrections": trans_corrs,
        "top_corrected_classes": sorted_class_corrs,
        "error_patterns": sorted_patterns,
        "task_breakdown": lang_corrs,
        "uncertain_reviewed_count": unc_reviewed,
        "uncertain_corrected_count": unc_corrected,
        "disagreements_resolved": len(corrections),
    }


"""
LinguaLabel — Human Annotation Feedback Curation Engine (Part 11.3)

Integrates with LinguaLabel's active learning feedback store to curate trusted human corrections:
- Tracks feedback review status: 'pending', 'approved', 'rejected'.
- Ensures only explicitly reviewed & approved human annotations can contribute to training data.
- Never includes unreviewed or rejected examples in trusted training sets.
- Supports corrections across Language, Sentiment, Sentence Type (including Complaining),
  and Abusive Language categories / severity.
- Provides duplicate feedback detection and clean training export.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from ml.active_learning.feedback import (
    FeedbackManager,
    FeedbackRecord,
    get_feedback_store_dir,
)
from ml.model_config import (
    ABUSE_CATEGORIES,
    ABUSE_SEVERITY_LEVELS,
    LANGUAGE_LABELS,
    SENTIMENT_LABELS,
    TEXT_TYPE_LABELS,
)


VALID_LABELS_FOR_FEEDBACK: Dict[str, Set[str]] = {
    "language": set(LANGUAGE_LABELS),
    "sentiment": set(SENTIMENT_LABELS),
    "text_type": set(TEXT_TYPE_LABELS),
    "abuse_category": set(ABUSE_CATEGORIES),
    "abuse_severity": set(ABUSE_SEVERITY_LEVELS),
}


class FeedbackCurationEngine:
    """
    Manages review, approval, rejection, and trusted training export of annotator corrections.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent
        self.feedback_mgr = FeedbackManager(self.root_dir)
        self.store_dir = get_feedback_store_dir(self.root_dir)
        self.feedback_dir = self.store_dir
        self.approvals_file = self.store_dir / "feedback_curation_manifest.json"
        self._ensure_approvals_file()

    def _ensure_approvals_file(self):
        if not self.approvals_file.exists():
            with open(self.approvals_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)

    def _load_approvals(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.approvals_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_approvals(self, data: Dict[str, Dict[str, Any]]):
        tmp = self.store_dir / "feedback_curation_manifest.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.approvals_file)

    def _load_all_records(self) -> List[Any]:
        records = []
        try:
            records = self.feedback_mgr.get_all_feedback()
        except Exception:
            records = []

        if not records:
            # Check for any .jsonl in feedback store directory
            for p in self.feedback_dir.glob("*.jsonl"):
                if p.name.endswith("audit.jsonl") or p.name.endswith("approvals.json"):
                    continue
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            d = json.loads(line)
                            fid = d.get("feedback_id") or d.get("id") or f"fb_{len(records)}"
                            txt = d.get("original_text") or d.get("text") or ""
                            corr = d.get("human_corrected_label") or d.get("corrected_label") or d.get("label") or ""
                            sug = d.get("model_predicted_label") or d.get("suggested_label") or ""
                            task = d.get("task") or "sentiment"
                            notes = d.get("annotator_notes") or d.get("notes") or ""
                            ts = d.get("timestamp") or datetime.now(timezone.utc).isoformat()
                            rec = FeedbackRecord(
                                feedback_id=fid,
                                dataset_key=d.get("dataset_key", "default"),
                                record_id=str(d.get("record_id", fid)),
                                task=task,
                                original_text=txt,
                                model_predicted_label=sug,
                                human_corrected_label=corr,
                                is_correction=bool(sug != corr),
                                model_confidence=float(d.get("confidence", 0.0) or d.get("model_confidence", 0.0)),
                                confidence_type=d.get("confidence_type", "heuristic"),
                                actual_backend=d.get("actual_backend", "baseline"),
                                model_checkpoint=d.get("model_checkpoint"),
                                timestamp=ts,
                                annotator_notes=notes,
                                was_model_uncertain=bool(d.get("was_model_uncertain", False)),
                            )
                            records.append(rec)
                except Exception:
                    continue
        return records

    def list_all_feedback_with_status(
        self,
        task: Optional[str] = None,
        status_filter: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all recorded feedback merged with their approval/rejection status.
        """
        records = self._load_all_records()
        approvals = self._load_approvals()

        results = []
        for r in records:
            # Check task filter
            if task and r.task != task:
                continue

            appr_info = approvals.get(r.feedback_id, {})
            current_status = appr_info.get("status", "pending")

            if status_filter and status_filter != "all" and current_status != status_filter:
                continue

            if search_query:
                q = search_query.strip().lower()
                if q not in r.original_text.lower() and q not in r.human_corrected_label.lower():
                    continue

            d = r.to_dict()
            d["curation_status"] = current_status
            d["reviewed_at"] = appr_info.get("reviewed_at")
            d["reviewed_by"] = appr_info.get("reviewed_by")
            results.append(d)

        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calculates counts of total, pending, approved, and rejected feedback records."""
        records = self._load_all_records()
        approvals = self._load_approvals()

        pending_count = 0
        approved_count = 0
        rejected_count = 0
        by_task: Dict[str, Dict[str, int]] = {}

        for r in records:
            status = approvals.get(r.feedback_id, {}).get("status", "pending")
            if status == "approved":
                approved_count += 1
            elif status == "rejected":
                rejected_count += 1
            else:
                pending_count += 1

            if r.task not in by_task:
                by_task[r.task] = {"total": 0, "approved": 0, "pending": 0, "rejected": 0}
            by_task[r.task]["total"] += 1
            by_task[r.task][status] += 1

        return {
            "total_feedback": len(records),
            "pending_count": pending_count,
            "approved_count": approved_count,
            "rejected_count": rejected_count,
            "by_task": by_task,
        }

    def set_feedback_status(
        self,
        feedback_id: str,
        status: str,
        reviewer: str = "owner",
    ) -> bool:
        """Sets the curation status for a feedback entry ('approved', 'rejected', or 'pending')."""
        if status not in ("approved", "rejected", "pending"):
            return False

        approvals = self._load_approvals()
        approvals[feedback_id] = {
            "status": status,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "reviewed_by": reviewer,
        }
        self._save_approvals(approvals)
        return True

    def bulk_set_status(self, feedback_ids: List[str], status: str, reviewer: str = "owner") -> int:
        """Applies status to multiple feedback IDs in one batch."""
        if status not in ("approved", "rejected", "pending"):
            return 0

        approvals = self._load_approvals()
        count = 0
        ts = datetime.now(timezone.utc).isoformat()
        for fid in feedback_ids:
            approvals[fid] = {
                "status": status,
                "reviewed_at": ts,
                "reviewed_by": reviewer,
            }
            count += 1
        self._save_approvals(approvals)
        return count

    def get_trusted_training_samples(
        self,
        task: Optional[str] = None,
        exclude_uncertain: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extracts only explicitly APPROVED feedback records into a clean training DataFrame.
        STRICT RULE: Pending and Rejected records are NEVER included.
        """
        records = self._load_all_records()
        approvals = self._load_approvals()

        clean_rows = []
        warnings = []
        seen_pairs: Set[str] = set()

        for r in records:
            if task and r.task != task:
                continue

            # Strict gate: only approved records
            status = approvals.get(r.feedback_id, {}).get("status", "pending")
            if status != "approved":
                continue

            lbl = r.human_corrected_label.strip()
            if exclude_uncertain and lbl.lower() in ("uncertain", "not sure", "— select —", ""):
                continue

            valid_labels = VALID_LABELS_FOR_FEEDBACK.get(r.task, set())
            if valid_labels and lbl not in valid_labels:
                warnings.append(f"Approved feedback {r.feedback_id} has non-canonical label '{lbl}' for task '{r.task}'; skipped.")
                continue

            # Deduplication key across exact same text and task
            dup_key = f"{r.task}|{r.original_text.strip()}"
            if dup_key in seen_pairs:
                continue
            seen_pairs.add(dup_key)

            clean_rows.append({
                "text": r.original_text.strip(),
                "label": lbl,
                "task": r.task,
                "record_id": r.record_id,
                "dataset_key": r.dataset_key,
                "source": "human_feedback_approved",
                "annotator_notes": r.annotator_notes,
                "feedback_id": r.feedback_id,
                "timestamp": r.timestamp,
            })

        df = pd.DataFrame(clean_rows)
        return df, warnings

    def get_curation_statistics(self) -> Dict[str, Any]:
        metrics = self.get_summary_metrics()
        return {
            "total_feedback": metrics["total_feedback"],
            "pending": metrics["pending_count"],
            "approved": metrics["approved_count"],
            "rejected": metrics["rejected_count"],
            "by_task": metrics.get("by_task", {}),
        }

    def curate_feedback(self, feedback_id: str, status: str, reviewer_notes: str = "", reviewer: str = "owner") -> bool:
        return self.set_feedback_status(feedback_id, status, reviewer)

    def bulk_curate(self, action: str, reviewer: str = "owner") -> int:
        if action == "approve_all_pending":
            pending_items = self.list_all_feedback_with_status(status_filter="pending")
            ids = [item["feedback_id"] for item in pending_items]
            return self.bulk_set_status(ids, "approved", reviewer)
        return 0

    def list_curated_feedback(self, status_filter: Optional[str] = None):
        items = self.list_all_feedback_with_status(status_filter=status_filter)
        class CuratedItem:
            def __init__(self, d):
                self._d = d
                self.feedback_id = d.get("feedback_id", "")
                self.text = d.get("original_text", "")
                self.suggested_label = d.get("model_predicted_label", "")
                self.corrected_label = d.get("human_corrected_label", "")
                self.task = d.get("task", "")
                self.curation_status = d.get("curation_status", "pending")
                self.notes = d.get("annotator_notes", "")
                self.timestamp = d.get("timestamp", "")
        return [CuratedItem(d) for d in items]

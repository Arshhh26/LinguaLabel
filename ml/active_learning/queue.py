"""
Prioritized Active Learning Queue Engine for LinguaLabel.

Builds, filters, sorts, and serves prioritized record queues for human annotators.
Enforces non-destructive execution, transparent reasoning, and full provenance tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from ml.active_learning.disagreement import compute_record_disagreement, RecordDisagreement
from ml.active_learning.priority import compute_priority_record, PriorityRecord, PriorityWeights
from ml.active_learning.uncertainty import compute_record_uncertainty, RecordUncertainty
from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.model_config import MLBackend


@dataclass
class QueueSummary:
    """High-level summary statistics of the active learning queue."""
    total_records: int = 0
    unreviewed_count: int = 0
    high_priority_count: int = 0
    medium_priority_count: int = 0
    low_priority_count: int = 0
    disagreement_count: int = 0
    uncertain_count: int = 0
    completed_count: int = 0
    ambiguous_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_records": self.total_records,
            "unreviewed_count": self.unreviewed_count,
            "high_priority_count": self.high_priority_count,
            "medium_priority_count": self.medium_priority_count,
            "low_priority_count": self.low_priority_count,
            "disagreement_count": self.disagreement_count,
            "uncertain_count": self.uncertain_count,
            "completed_count": self.completed_count,
            "ambiguous_count": self.ambiguous_count,
        }


class ActiveLearningQueue:
    """
    Coordinates multi-criteria prioritization across dataset records.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ml_engine: MLEngine,
        strategy: str = "lowest_confidence",
        weights: Optional[PriorityWeights] = None,
        run_transformer: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        self.df = df
        self.ml_engine = ml_engine
        self.strategy = strategy
        self.weights = weights or PriorityWeights()
        self.run_transformer = run_transformer
        self.records: List[PriorityRecord] = []
        self.summary = QueueSummary()
        self._build_queue(progress_callback)

    def _build_queue(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> None:
        """Processes all rows in df through baseline and transformer models."""
        if self.df is None or self.df.empty:
            return

        total_rows = len(self.df)
        records_list: List[PriorityRecord] = []

        unrev_c = 0
        high_c = 0
        med_c = 0
        low_c = 0
        disag_c = 0
        unc_c = 0
        comp_c = 0
        amb_c = 0

        # Save previous engine backend state
        orig_backend = self.ml_engine.active_backend

        try:
            for idx, (_, row) in enumerate(self.df.iterrows()):
                if progress_callback:
                    progress_callback(idx + 1, total_rows)

                rec_id = str(row.get("id", f"row_{idx}"))
                text = str(row.get("text", "")).strip()

                # 1. Baseline Predictions
                self.ml_engine.set_backend(MLBackend.BASELINE)
                b_preds = self.ml_engine.predict_all(text)

                # 2. Transformer Predictions (if requested and available)
                t_preds: Dict[str, PredictionResult] = {}
                if self.run_transformer:
                    self.ml_engine.set_backend(MLBackend.TRANSFORMER)
                    t_preds = self.ml_engine.predict_all(text)

                # 3. Model Disagreement
                disag_report: Optional[RecordDisagreement] = None
                if self.run_transformer and t_preds:
                    disag_report = compute_record_disagreement(
                        record_id=rec_id,
                        baseline_predictions=b_preds,
                        transformer_predictions=t_preds,
                    )
                    if disag_report.has_disagreement:
                        disag_c += 1

                # 4. Uncertainty Estimation
                # Use transformer predictions for uncertainty if available genuinely, else baseline
                eval_preds = b_preds
                if self.run_transformer and t_preds:
                    # check if transformer had actual transformer backend
                    eval_preds = {
                        k: (t_preds[k] if t_preds.get(k) and t_preds[k].actual_backend == "transformer" else b_preds[k])
                        for k in b_preds.keys()
                    }

                unc_report = compute_record_uncertainty(
                    record_id=rec_id,
                    predictions=eval_preds,
                    strategy=self.strategy,
                )
                if unc_report.has_uncertain_prediction:
                    unc_c += 1

                # 5. Composite Priority Scoring
                priority_item = compute_priority_record(
                    record_id=rec_id,
                    text=text,
                    row_data=row.to_dict(),
                    uncertainty_report=unc_report,
                    disagreement_report=disag_report,
                    weights=self.weights,
                )

                # Summary Counters
                if priority_item.is_reviewed:
                    comp_c += 1
                else:
                    unrev_c += 1

                if priority_item.is_ambiguous:
                    amb_c += 1

                if priority_item.priority_level == "High":
                    high_c += 1
                elif priority_item.priority_level == "Medium":
                    med_c += 1
                else:
                    low_c += 1

                records_list.append(priority_item)

        finally:
            self.ml_engine.set_backend(orig_backend)

        self.records = records_list
        self.summary = QueueSummary(
            total_records=total_rows,
            unreviewed_count=unrev_c,
            high_priority_count=high_c,
            medium_priority_count=med_c,
            low_priority_count=low_c,
            disagreement_count=disag_c,
            uncertain_count=unc_c,
            completed_count=comp_c,
            ambiguous_count=amb_c,
        )

    def get_filtered_queue(
        self,
        sort_by: str = "priority",
        priority_filter: str = "All",
        status_filter: str = "All",
        condition_filter: str = "All",
        search_query: str = "",
    ) -> List[PriorityRecord]:
        """
        Filters and sorts the priority queue according to user selection.

        sort_by: 'priority', 'uncertainty', 'margin', 'entropy', 'disagreement'
        priority_filter: 'All', 'High', 'Medium', 'Low'
        status_filter: 'All', 'Pending', 'Reviewed', 'Ambiguous / Second Review'
        condition_filter: 'All', 'Model Disagreement Only', 'Uncertain Only', 'Missing Fields Only'
        """
        filtered = list(self.records)

        # 1. Search filter
        if search_query and search_query.strip():
            sq = search_query.strip().lower()
            filtered = [r for r in filtered if sq in r.text.lower() or sq in r.record_id.lower()]

        # 2. Priority Level Filter
        if priority_filter != "All":
            filtered = [r for r in filtered if r.priority_level == priority_filter]

        # 3. Status Filter
        if status_filter == "Pending":
            filtered = [r for r in filtered if not r.is_reviewed]
        elif status_filter == "Reviewed":
            filtered = [r for r in filtered if r.is_reviewed]
        elif status_filter == "Ambiguous / Second Review":
            filtered = [r for r in filtered if r.is_ambiguous]

        # 4. Condition Filter
        if condition_filter == "Model Disagreement Only":
            filtered = [r for r in filtered if r.disagreement_score > 0]
        elif condition_filter == "Uncertain Only":
            filtered = [r for r in filtered if r.uncertainty_report and r.uncertainty_report.has_uncertain_prediction]
        elif condition_filter == "Missing Fields Only":
            filtered = [r for r in filtered if len(r.missing_fields) > 0]

        # 5. Sorting
        if sort_by == "uncertainty":
            filtered.sort(key=lambda r: r.uncertainty_score, reverse=True)
        elif sort_by == "margin":
            # For margin, items with smallest margin have highest margin uncertainty
            def get_margin_val(r: PriorityRecord) -> float:
                if not r.uncertainty_report:
                    return 0.0
                margins = [tu.margin_uncertainty for tu in r.uncertainty_report.tasks.values() if tu.margin_uncertainty is not None]
                return max(margins) if margins else 0.0
            filtered.sort(key=get_margin_val, reverse=True)
        elif sort_by == "entropy":
            def get_entropy_val(r: PriorityRecord) -> float:
                if not r.uncertainty_report:
                    return 0.0
                entropies = [tu.entropy for tu in r.uncertainty_report.tasks.values() if tu.entropy is not None]
                return max(entropies) if entropies else 0.0
            filtered.sort(key=get_entropy_val, reverse=True)
        elif sort_by == "disagreement":
            filtered.sort(key=lambda r: (r.disagreement_score, r.priority_score), reverse=True)
        else:
            # Default: by composite priority score
            filtered.sort(key=lambda r: r.priority_score, reverse=True)

        return filtered

    def to_dataframe(self, records: Optional[List[PriorityRecord]] = None) -> pd.DataFrame:
        """Converts records into a clean display DataFrame."""
        target = records if records is not None else self.records
        rows = []
        for r in target:
            # Baseline vs Transformer Sentiment & Text Type
            b_sent, t_sent = "", ""
            b_tt, t_tt = "", ""
            if r.disagreement_report:
                td_s = r.disagreement_report.tasks.get("sentiment")
                if td_s:
                    b_sent = f"{td_s.baseline_label} ({td_s.baseline_confidence:.0%})"
                    t_sent = f"{td_s.transformer_label} ({td_s.transformer_confidence:.0%})" if td_s.transformer_label else "N/A"

                td_t = r.disagreement_report.tasks.get("text_type")
                if td_t:
                    b_tt = f"{td_t.baseline_label} ({td_t.baseline_confidence:.0%})"
                    t_tt = f"{td_t.transformer_label} ({td_t.transformer_confidence:.0%})" if td_t.transformer_label else "N/A"

            rows.append({
                "ID": r.record_id,
                "Text": r.text,
                "Priority Level": r.priority_level,
                "Priority Score": f"{r.priority_score:.2f}",
                "Primary Reason": r.primary_reason,
                "Disagreement": "Yes" if r.disagreement_score > 0 else "No",
                "Baseline Sentiment": b_sent,
                "Transformer Sentiment": t_sent,
                "Baseline Text Type": b_tt,
                "Transformer Text Type": t_tt,
                "Status": r.human_status,
            })
        return pd.DataFrame(rows)


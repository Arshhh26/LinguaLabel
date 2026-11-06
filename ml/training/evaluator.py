"""
LinguaLabel — Model Evaluation & Comparison Engine (Part 11.3)

Provides rigorous, independent evaluation for candidate models and production models:
1. Evaluates candidate models against isolated test splits and fixed evaluation benchmarks.
2. Computes global metrics: Accuracy, Precision, Recall, Macro F1, Weighted F1.
3. Computes per-class metrics: Precision, Recall, F1, Support.
4. Generates full Confusion Matrices with label alignments.
5. Computes language-stratified metrics (English, Hindi, Hinglish performance).
6. Executes specialized regression diagnostics:
   - Complaining false positives on ordinary negative feedback ("The price was high").
   - Abuse false positives on product criticism ("The camera is horrible"), quoted profanity, academic discussion.
7. Produces side-by-side comparison with Delta indicators (improved, regressed, neutral).
8. Strict guardrails: Prohibits promotion if severe regressions are detected on critical benchmarks.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from ml.base import PredictionResult
from ml.training.versioning import ModelVersion, ModelVersionRegistry


@dataclass
class EvaluationReport:
    model_name: str
    task: str
    dataset_name: str
    num_samples: int
    evaluated_at: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: Dict[str, Any]
    per_language_metrics: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonMetricRow:
    metric_name: str
    production_value: float
    candidate_value: float
    delta: float
    status: str  # "improved", "regressed", "neutral"
    status_icon: str
    description: str


@dataclass
class ModelComparisonReport:
    task: str
    dataset_name: str
    production_model_name: str
    candidate_model_name: str
    compared_at: str
    recommend_promotion: bool
    is_severe_regression: bool
    verdict_summary: str
    comparison_table: List[ComparisonMetricRow]
    per_language_comparison: Dict[str, Dict[str, Any]]
    diagnostic_comparison: Dict[str, Any]
    regression_warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "dataset_name": self.dataset_name,
            "production_model_name": self.production_model_name,
            "candidate_model_name": self.candidate_model_name,
            "compared_at": self.compared_at,
            "recommend_promotion": self.recommend_promotion,
            "is_severe_regression": self.is_severe_regression,
            "verdict_summary": self.verdict_summary,
            "comparison_table": [asdict(r) for r in self.comparison_table],
            "per_language_comparison": self.per_language_comparison,
            "diagnostic_comparison": self.diagnostic_comparison,
            "regression_warnings": self.regression_warnings,
        }


class ModelEvaluationEngine:
    """
    Evaluates ML models against test splits and benchmarks, generating detailed
    performance audits and side-by-side candidate comparisons.
    """

    def __init__(self, root_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path(__file__).resolve().parent.parent.parent
        self.eval_dir = self.root_dir / "evaluation_results"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_dir = self.root_dir / "evaluation_datasets"

    def evaluate(
        self,
        predictor: Any,
        dataset: pd.DataFrame,
        task: str,
        text_col: str = "text",
        label_col: Optional[str] = None,
        model_name: str = "Candidate Model",
        dataset_name: str = "Evaluation Split",
    ) -> EvaluationReport:
        """
        Runs comprehensive evaluation of any predictor against a labelled dataset.
        """
        if label_col is None:
            # Map canonical column name based on task
            task_col_map = {
                "language": "language",
                "sentiment": "sentiment",
                "text_type": "text_type",
                "complaint": "text_type",
                "abuse": "abuse_category",
                "abuse_category": "abuse_category",
                "abuse_severity": "abuse_severity",
            }
            label_col = task_col_map.get(task, task)

        if text_col not in dataset.columns or label_col not in dataset.columns:
            raise ValueError(f"Dataset missing required columns '{text_col}' or '{label_col}'")

        df = dataset.dropna(subset=[text_col, label_col]).copy()
        texts = df[text_col].astype(str).tolist()
        y_true = df[label_col].astype(str).str.strip().tolist()

        y_pred: List[str] = []
        for t in texts:
            try:
                res = predictor.predict(t)
                if isinstance(res, PredictionResult):
                    # If uncertain, label is "Uncertain"
                    y_pred.append(str(res.label).strip())
                elif isinstance(res, dict):
                    y_pred.append(str(res.get("label", "Uncertain")).strip())
                else:
                    y_pred.append(str(res).strip())
            except Exception:
                y_pred.append("Uncertain")

        # Global metrics
        acc = float(accuracy_score(y_true, y_pred))
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        # Unique labels union
        all_unique_labels = sorted(list(set(y_true) | set(y_pred)))
        p_class, r_class, f1_class, supp_class = precision_recall_fscore_support(
            y_true, y_pred, labels=all_unique_labels, zero_division=0
        )
        per_class_metrics: Dict[str, Dict[str, float]] = {}
        for idx, lbl in enumerate(all_unique_labels):
            per_class_metrics[lbl] = {
                "precision": round(float(p_class[idx]), 4),
                "recall": round(float(r_class[idx]), 4),
                "f1": round(float(f1_class[idx]), 4),
                "support": int(supp_class[idx]),
            }

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_unique_labels)
        cm_dict = {
            "labels": all_unique_labels,
            "matrix": cm.tolist(),
        }

        # Per-language Stratified Metrics
        per_language_metrics: Dict[str, Dict[str, float]] = {}
        if "language" in df.columns:
            for lang in ["English", "Hindi", "Hinglish"]:
                lang_sub = df[df["language"].astype(str).str.strip().str.lower() == lang.lower()]
                if not lang_sub.empty:
                    sub_idx = lang_sub.index
                    sub_true = [y_true[i] for i in range(len(df)) if df.index[i] in sub_idx]
                    sub_pred = [y_pred[i] for i in range(len(df)) if df.index[i] in sub_idx]
                    sub_acc = float(accuracy_score(sub_true, sub_pred))
                    _, _, sub_f1, _ = precision_recall_fscore_support(
                        sub_true, sub_pred, average="macro", zero_division=0
                    )
                    per_language_metrics[lang] = {
                        "samples": int(len(lang_sub)),
                        "accuracy": round(sub_acc, 4),
                        "macro_f1": round(float(sub_f1), 4),
                    }

        # Specialized Diagnostics
        diagnostics: Dict[str, Any] = {}
        if task in ("text_type", "complaint", "complaining"):
            # Check false positive complaining on ordinary negative statements
            neg_statements_idx = [
                i for i in range(len(df))
                if df.iloc[i].get("sentiment") == "Negative" and str(y_true[i]).lower() != "complaining"
            ]
            if neg_statements_idx:
                fp_count = sum(1 for i in neg_statements_idx if str(y_pred[i]).lower() == "complaining")
                diagnostics["complaint_fp_on_negative_statements"] = {
                    "total_controls": len(neg_statements_idx),
                    "false_positives": fp_count,
                    "false_positive_rate": round(fp_count / len(neg_statements_idx), 4),
                }

            # Check true complaining recall
            comp_idx = [i for i in range(len(df)) if str(y_true[i]).lower() == "complaining"]
            if comp_idx:
                tp_count = sum(1 for i in comp_idx if str(y_pred[i]).lower() == "complaining")
                diagnostics["complaint_detection_recall"] = {
                    "total_complaints": len(comp_idx),
                    "detected": tp_count,
                    "recall": round(tp_count / len(comp_idx), 4),
                }

        elif task in ("abuse", "abuse_category"):
            # Check false positives on benign controls / product criticism / quoted / academic
            benign_idx = [i for i in range(len(df)) if str(y_true[i]).strip() == "Not abusive"]
            if benign_idx:
                fp_count = sum(1 for i in benign_idx if str(y_pred[i]).strip() != "Not abusive")
                diagnostics["abuse_fp_on_benign_controls"] = {
                    "total_benign": len(benign_idx),
                    "false_positives": fp_count,
                    "false_positive_rate": round(fp_count / len(benign_idx), 4),
                }

            # True abuse recall
            abusive_idx = [i for i in range(len(df)) if str(y_true[i]).strip() != "Not abusive"]
            if abusive_idx:
                tp_count = sum(1 for i in abusive_idx if str(y_pred[i]).strip() != "Not abusive")
                diagnostics["abuse_overall_recall"] = {
                    "total_abusive": len(abusive_idx),
                    "detected": tp_count,
                    "recall": round(tp_count / len(abusive_idx), 4),
                }

        report = EvaluationReport(
            model_name=model_name,
            task=task,
            dataset_name=dataset_name,
            num_samples=len(df),
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            accuracy=round(acc, 4),
            precision_macro=round(float(p_macro), 4),
            recall_macro=round(float(r_macro), 4),
            f1_macro=round(float(f1_macro), 4),
            precision_weighted=round(float(p_weighted), 4),
            recall_weighted=round(float(r_weighted), 4),
            f1_weighted=round(float(f1_weighted), 4),
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm_dict,
            per_language_metrics=per_language_metrics,
            diagnostics=diagnostics,
        )

        return report

    def compare_reports(
        self,
        production_report: EvaluationReport,
        candidate_report: EvaluationReport,
    ) -> ModelComparisonReport:
        """
        Performs a side-by-side delta comparison between production and candidate models.
        """
        table_rows: List[ComparisonMetricRow] = []
        regression_warnings: List[str] = []
        is_severe_regression = False

        metric_defs = [
            ("Accuracy", "Overall exact match ratio", production_report.accuracy, candidate_report.accuracy),
            ("Macro F1", "Unweighted average F1 across all classes", production_report.f1_macro, candidate_report.f1_macro),
            ("Macro Precision", "Unweighted average precision", production_report.precision_macro, candidate_report.precision_macro),
            ("Macro Recall", "Unweighted average recall", production_report.recall_macro, candidate_report.recall_macro),
            ("Weighted F1", "Sample-weighted F1 across classes", production_report.f1_weighted, candidate_report.f1_weighted),
        ]

        for name, desc, prod_v, cand_v in metric_defs:
            delta = round(cand_v - prod_v, 4)
            if delta > 0.005:
                status = "improved"
                icon = "🟢"
            elif delta < -0.005:
                status = "regressed"
                icon = "🔴"
            else:
                status = "neutral"
                icon = "⚪"

            table_rows.append(ComparisonMetricRow(
                metric_name=name,
                production_value=round(prod_v, 4),
                candidate_value=round(cand_v, 4),
                delta=delta,
                status=status,
                status_icon=icon,
                description=desc,
            ))

        # Check for Macro F1 regression
        f1_delta = candidate_report.f1_macro - production_report.f1_macro
        if f1_delta < -0.03:
            is_severe_regression = True
            regression_warnings.append(
                f"Candidate Macro F1 dropped significantly by {abs(f1_delta)*100:.1f}% compared to production."
            )

        # Per-language comparison
        per_lang_comp: Dict[str, Dict[str, Any]] = {}
        for lang in ["English", "Hindi", "Hinglish"]:
            prod_lang = production_report.per_language_metrics.get(lang, {})
            cand_lang = candidate_report.per_language_metrics.get(lang, {})
            if prod_lang and cand_lang:
                p_f1 = prod_lang.get("macro_f1", 0.0)
                c_f1 = cand_lang.get("macro_f1", 0.0)
                d_f1 = round(c_f1 - p_f1, 4)
                per_lang_comp[lang] = {
                    "production_macro_f1": p_f1,
                    "candidate_macro_f1": c_f1,
                    "delta": d_f1,
                    "status": "improved" if d_f1 > 0.005 else ("regressed" if d_f1 < -0.005 else "neutral"),
                }
                if d_f1 < -0.05:
                    regression_warnings.append(
                        f"Candidate performance on {lang} regressed by {abs(d_f1)*100:.1f}%."
                    )

        # Diagnostic comparison
        diagnostic_comp: Dict[str, Any] = {}
        p_diag = production_report.diagnostics
        c_diag = candidate_report.diagnostics

        # Complaining false positive diagnostic
        if "complaint_fp_on_negative_statements" in p_diag and "complaint_fp_on_negative_statements" in c_diag:
            p_fp_rate = p_diag["complaint_fp_on_negative_statements"]["false_positive_rate"]
            c_fp_rate = c_diag["complaint_fp_on_negative_statements"]["false_positive_rate"]
            d_fp = round(c_fp_rate - p_fp_rate, 4)
            diagnostic_comp["complaint_fp_rate"] = {
                "production": p_fp_rate,
                "candidate": c_fp_rate,
                "delta": d_fp,
                "improved": d_fp < 0,  # Lower FP rate is better!
            }
            if d_fp > 0.10:
                is_severe_regression = True
                regression_warnings.append(
                    f"Candidate false positive complaining rate increased by {d_fp*100:.1f}% on ordinary negative statements."
                )

        # Abuse false positive diagnostic
        if "abuse_fp_on_benign_controls" in p_diag and "abuse_fp_on_benign_controls" in c_diag:
            p_fp_rate = p_diag["abuse_fp_on_benign_controls"]["false_positive_rate"]
            c_fp_rate = c_diag["abuse_fp_on_benign_controls"]["false_positive_rate"]
            d_fp = round(c_fp_rate - p_fp_rate, 4)
            diagnostic_comp["abuse_fp_rate"] = {
                "production": p_fp_rate,
                "candidate": c_fp_rate,
                "delta": d_fp,
                "improved": d_fp < 0,  # Lower FP rate is better
            }
            if d_fp > 0.10:
                is_severe_regression = True
                regression_warnings.append(
                    f"Candidate false positive abuse rate increased by {d_fp*100:.1f}% on benign product criticism and quoted terms."
                )

        # Recommendation logic
        recommend_promotion = bool(not is_severe_regression and f1_delta >= -0.01)

        if recommend_promotion:
            if f1_delta > 0.01:
                verdict = f"Recommended for Promotion: Candidate achieves +{f1_delta*100:.1f}% higher Macro F1 without regressions."
            else:
                verdict = "Recommended for Promotion: Candidate achieves parity with production and maintains safety guardrails."
        else:
            if is_severe_regression:
                verdict = f"DO NOT PROMOTE: Severe regression detected. {'; '.join(regression_warnings)}"
            else:
                verdict = f"Promotion Not Recommended: Candidate Macro F1 is lower ({f1_delta*100:.1f}%) than current production."

        return ModelComparisonReport(
            task=production_report.task,
            dataset_name=candidate_report.dataset_name,
            production_model_name=production_report.model_name,
            candidate_model_name=candidate_report.model_name,
            compared_at=datetime.now(timezone.utc).isoformat(),
            recommend_promotion=recommend_promotion,
            is_severe_regression=is_severe_regression,
            verdict_summary=verdict,
            comparison_table=table_rows,
            per_language_comparison=per_lang_comp,
            diagnostic_comparison=diagnostic_comp,
            regression_warnings=regression_warnings,
        )

    def evaluate_benchmark(
        self,
        predictor: Any,
        benchmark_name: str,
        task: str,
        model_name: str = "Candidate Model",
    ) -> Optional[EvaluationReport]:
        """Loads and evaluates against fixed benchmark datasets."""
        file_path = self.benchmark_dir / benchmark_name
        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)
        return self.evaluate(
            predictor=predictor,
            dataset=df,
            task=task,
            model_name=model_name,
            dataset_name=benchmark_name,
        )


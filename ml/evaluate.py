"""
Empirical Model Evaluation CLI for LinguaLabel ML Engine.

Computes exact, un-fabricated evaluation metrics against labeled ground truth data:
- Confusion Matrix
- Per-class Precision, Recall, and F1-Score
- Overall Accuracy and Macro-averaged metrics
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import pandas as pd

from ml.engine import MLEngine


def compute_metrics(
    y_true: List[str], y_pred: List[str]
) -> Tuple[float, float, float, float, Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """
    Computes exact empirical evaluation metrics without fabrication.
    Returns: (accuracy, macro_precision, macro_recall, macro_f1, per_class_metrics, confusion_matrix)
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0, 0.0, 0.0, 0.0, {}, {}

    labels = sorted(list(set(y_true) | set(y_pred)))
    total = len(y_true)

    # Confusion matrix: cm[actual][predicted]
    cm: Dict[str, Dict[str, int]] = {true_lbl: {pred_lbl: 0 for pred_lbl in labels} for true_lbl in labels}
    correct = 0

    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
        if t == p:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    per_class = {}
    precisions = []
    recalls = []
    f1s = []

    for lbl in labels:
        tp = cm[lbl][lbl]
        fp = sum(cm[other][lbl] for other in labels if other != lbl)
        fn = sum(cm[lbl][other] for other in labels if other != lbl)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(cm[lbl].values())

        per_class[lbl] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }

        # Include classes with non-zero support in macro average
        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return accuracy, macro_prec, macro_rec, macro_f1, per_class, cm


def format_report(task_name: str, y_true: List[str], y_pred: List[str]) -> str:
    """Formats an ASCII classification report with confusion matrix."""
    acc, macro_p, macro_r, macro_f1, per_class, cm = compute_metrics(y_true, y_pred)
    lines = []
    lines.append(f"\n========================================================")
    lines.append(f" TASK: {task_name.upper()}")
    lines.append(f" Total Samples Evaluated: {len(y_true)}")
    lines.append(f" Overall Accuracy:        {acc:.4f} ({acc*100:.1f}%)")
    lines.append(f" Macro Precision:         {macro_p:.4f}")
    lines.append(f" Macro Recall:            {macro_r:.4f}")
    lines.append(f" Macro F1-Score:          {macro_f1:.4f}")
    lines.append(f"--------------------------------------------------------")
    lines.append(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    lines.append(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for cls_name, m in sorted(per_class.items()):
        if m["support"] > 0 or m["precision"] > 0:
            lines.append(f"{cls_name:<25} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1']:<10.4f} {m['support']:<8}")

    lines.append(f"--------------------------------------------------------")
    lines.append("Confusion Matrix (Rows: Actual, Columns: Predicted):")
    all_classes = sorted(list(cm.keys()))
    header = f"{'Actual \\ Pred':<20} " + " ".join([f"{c[:8]:>8}" for c in all_classes])
    lines.append(header)
    for row_cls in all_classes:
        row_str = f"{row_cls:<20} " + " ".join([f"{cm[row_cls][col_cls]:>8}" for col_cls in all_classes])
        lines.append(row_str)
    lines.append(f"========================================================\n")
    return "\n".join(lines)


def run_evaluation(dataset_path: Path) -> int:
    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        return 1

    print(f"\nLoading evaluation ground truth dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, keep_default_na=False, dtype=str)
    if "text" not in df.columns:
        print("Error: Dataset must contain a 'text' column.")
        return 1

    engine = MLEngine()
    if not engine.is_available():
        print("Error: MLEngine is not available.")
        return 1

    tasks = [
        ("Language Identification", "language", engine.predict_language),
        ("Sentiment Polarity", "sentiment", engine.predict_sentiment),
        ("Text Type Classification", "text_type", engine.predict_text_type),
    ]

    for task_title, col_name, predict_func in tasks:
        if col_name not in df.columns:
            print(f"Skipping {task_title}: column '{col_name}' not found in dataset.")
            continue

        valid_df = df[df[col_name].astype(str).str.strip().ne("")].copy()
        if valid_df.empty:
            print(f"Skipping {task_title}: no labeled samples in column '{col_name}'.")
            continue

        y_true = valid_df[col_name].astype(str).str.strip().tolist()
        y_pred = [predict_func(str(txt)).label for txt in valid_df["text"]]

        report = format_report(task_title, y_true, y_pred)
        print(report)

    return 0


def main():
    parser = argparse.ArgumentParser(description="LinguaLabel Empirical Model Evaluation CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sample_data.csv",
        help="Path to labeled CSV dataset (default: sample_data.csv)",
    )
    args = parser.parse_args()
    sys.exit(run_evaluation(Path(args.dataset)))


if __name__ == "__main__":
    main()

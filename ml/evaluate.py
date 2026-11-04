"""
Empirical Model Evaluation & Comparison CLI for LinguaLabel ML Engine.

Computes exact, un-fabricated evaluation metrics against labeled ground truth data
for both Baseline and Transformer backends:
- Side-by-side comparative accuracy, macro precision, recall, and F1-score
- Confusion matrices
- Per-language performance breakdown (English, Hindi, etc.)
- Model inference latency (total & mean per sample)
- Memory and model parameter profile
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

# Ensure project root is on sys.path for direct script execution
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from typing import Any, Callable, Dict, List, Optional, Tuple
import pandas as pd

from ml.engine import MLEngine
from ml.model_config import MLBackend, VERIFIED_MODELS


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

        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_prec = sum(precisions) / len(precisions) if precisions else 0.0
    macro_rec = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return accuracy, macro_prec, macro_rec, macro_f1, per_class, cm


def evaluate_task(
    df: pd.DataFrame,
    task_col: str,
    predict_func: Callable[[str], Any],
    lang_col: Optional[str] = "language",
) -> Dict[str, Any]:
    """
    Evaluates a specific task column against predictions.
    Computes overall and per-language metrics plus execution latency.
    """
    valid_df = df[df[task_col].astype(str).str.strip().ne("")].copy()
    if valid_df.empty:
        return {"support": 0, "accuracy": 0.0, "macro_f1": 0.0, "latency_ms": 0.0}

    y_true = valid_df[task_col].astype(str).str.strip().tolist()
    texts = valid_df["text"].astype(str).tolist()

    t_start = time.perf_counter()
    predictions = [predict_func(txt) for txt in texts]
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    y_pred = [p.label for p in predictions]
    acc, macro_p, macro_r, macro_f1, per_class, cm = compute_metrics(y_true, y_pred)

    actual_backends = [p.actual_backend for p in predictions]
    fallback_count = sum(1 for b in actual_backends if b == "fallback_baseline")
    is_fallback = (fallback_count == len(predictions)) if predictions else False
    has_partial_fallback = (fallback_count > 0 and not is_fallback) if predictions else False

    # Per-language breakdown
    by_lang = {}
    if lang_col and lang_col in valid_df.columns:
        valid_df["pred"] = y_pred
        for lang_val, group in valid_df.groupby(valid_df[lang_col].astype(str).str.strip()):
            if not lang_val:
                continue
            sub_acc, sub_mp, sub_mr, sub_f1, _, _ = compute_metrics(
                group[task_col].tolist(), group["pred"].tolist()
            )
            by_lang[lang_val] = {
                "support": len(group),
                "accuracy": round(sub_acc, 4),
                "macro_f1": round(sub_f1, 4),
            }

    return {
        "support": len(y_true),
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "by_language": by_lang,
        "total_latency_ms": round(total_time_ms, 2),
        "avg_latency_ms": round(total_time_ms / len(y_true), 2) if y_true else 0.0,
        "is_fallback": is_fallback,
        "fallback_count": fallback_count,
        "has_partial_fallback": has_partial_fallback,
        "actual_backend": "fallback_baseline" if is_fallback else ("mixed" if has_partial_fallback else (actual_backends[0] if actual_backends else "unknown")),
    }


def compare_backends(dataset_path: Path) -> Dict[str, Any]:
    """
    Runs benchmark comparison across Baseline and Transformer backends.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    df = pd.read_csv(dataset_path, keep_default_na=False, dtype=str)
    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    baseline_engine = MLEngine(backend=MLBackend.BASELINE)
    transformer_engine = MLEngine(backend=MLBackend.TRANSFORMER)

    tasks = [
        ("Sentiment Polarity", "sentiment", baseline_engine.predict_sentiment, transformer_engine.predict_sentiment),
        ("Text Type Classification", "text_type", baseline_engine.predict_text_type, transformer_engine.predict_text_type),
    ]

    results = {
        "dataset": str(dataset_path),
        "total_rows": len(df),
        "tasks": {},
    }

    for task_name, col_name, base_fn, tf_fn in tasks:
        if col_name not in df.columns:
            continue
        valid_count = len(df[df[col_name].astype(str).str.strip().ne("")])
        if valid_count == 0:
            continue

        base_res = evaluate_task(df, col_name, base_fn)
        tf_res = evaluate_task(df, col_name, tf_fn)

        results["tasks"][col_name] = {
            "title": task_name,
            "baseline": base_res,
            "transformer": tf_res,
        }

    return results


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Formats side-by-side terminal comparison between backends."""
    print("\n" + "=" * 78)
    print(" LINGUALABEL — MODEL BACKEND BENCHMARK & COMPARISON REPORT")
    print("=" * 78)
    print(f" Dataset Evaluated: {results['dataset']} (Total rows: {results['total_rows']})")

    for col_name, t_info in results.get("tasks", {}).items():
        title = t_info["title"]
        base = t_info["baseline"]
        tf = t_info["transformer"]

        print("\n" + "-" * 78)
        print(f" TASK: {title.upper()} (Samples Evaluated: {base['support']})")
        print("-" * 78)
        print(f"{'Metric':<25} {'Baseline':<25} {'Transformer':<25}")
        print(f"{'-'*25} {'-'*25} {'-'*25}")

        if tf.get("is_fallback"):
            print(f"{'Accuracy':<25} {base['accuracy']*100:>6.2f}%{' ':18} [UNAVAILABLE — FALLBACK]")
            print(f"{'Macro Precision':<25} {base['macro_precision']:>6.4f}{' ':18} [N/A — FALLBACK]")
            print(f"{'Macro Recall':<25} {base['macro_recall']:>6.4f}{' ':18} [N/A — FALLBACK]")
            print(f"{'Macro F1-Score':<25} {base['macro_f1']:>6.4f}{' ':18} [N/A — FALLBACK]")
            print(f"{'Mean Latency / sample':<25} {base['avg_latency_ms']:>6.2f} ms{' ':15} {tf['avg_latency_ms']:>6.2f} ms (Fallback)")
            print("\n  ⚠️  NOTICE: Transformer backend did NOT execute. All predictions fell back to baseline")
            print("      because transformer dependencies or weights are not loaded in the environment.")
            print("      Baseline fallback metrics are not counted as transformer performance.")
        else:
            print(f"{'Accuracy':<25} {base['accuracy']*100:>6.2f}%{' ':18} {tf['accuracy']*100:>6.2f}%")
            print(f"{'Macro Precision':<25} {base['macro_precision']:>6.4f}{' ':18} {tf['macro_precision']:>6.4f}")
            print(f"{'Macro Recall':<25} {base['macro_recall']:>6.4f}{' ':18} {tf['macro_recall']:>6.4f}")
            print(f"{'Macro F1-Score':<25} {base['macro_f1']:>6.4f}{' ':18} {tf['macro_f1']:>6.4f}")
            print(f"{'Mean Latency / sample':<25} {base['avg_latency_ms']:>6.2f} ms{' ':15} {tf['avg_latency_ms']:>6.2f} ms")

        # Per-language breakdown
        all_langs = sorted(list(set(base.get("by_language", {}).keys()) | set(tf.get("by_language", {}).keys())))
        if all_langs:
            print(f"\n Per-Language Performance (Accuracy):")
            for lang in all_langs:
                b_acc = base.get("by_language", {}).get(lang, {}).get("accuracy", 0.0)
                if tf.get("is_fallback"):
                    t_str = "[Fallback]"
                else:
                    t_acc = tf.get("by_language", {}).get(lang, {}).get("accuracy", 0.0)
                    t_str = f"{t_acc*100:>5.1f}%"
                n_samples = base.get("by_language", {}).get(lang, {}).get("support", 0)
                print(f"  • {lang:<15} (N={n_samples:<3}): Baseline={b_acc*100:>5.1f}% | Transformer={t_str}")

        if base["support"] < 30:
            print(f"\n ⚠️ Notice: Sample size is small (N={base['support']}).")
            print("   Expand annotated ground truth to >=50 samples per class for statistical significance.")

    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="LinguaLabel Model Evaluation & Comparison CLI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="annotated_output.csv",
        help="Path to labeled CSV dataset (default: annotated_output.csv)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Run side-by-side comparative evaluation of Baseline vs Transformer",
    )
    args = parser.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        # Fallback to sample_data.csv if annotated_output.csv is not yet present
        alt_path = Path("sample_data.csv")
        if alt_path.exists():
            ds_path = alt_path

    try:
        results = compare_backends(ds_path)
        print_comparison_report(results)
    except Exception as e:
        print(f"Evaluation error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

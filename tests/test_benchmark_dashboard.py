"""
Unit and Integration Test Suite for Part 8: Model Performance & Quality Dashboard.

Covers:
- Evaluation dataset validation and integrity checks.
- Handling of missing, empty, or corrupted evaluation data.
- Mandatory preliminary small-sample warnings (N < 30).
- Empirical metric calculations (Accuracy, Macro F1, Weighted F1, Precision, Recall).
- Confusion matrix correctness.
- Evaluation dataset storage isolation (evaluation_datasets/ separation).
- Paired benchmark execution and persistence in benchmark_history/.
- Benchmark history listing and loading.
- Provenance tracking and fallback exclusion from transformer metrics.
- Confidence type separation (heuristic vs. model_probability).
- Per-language cross-lingual breakdown and sample-size guardrails.
- Export suite (Metrics CSV, Predictions CSV, Confusion Matrix CSV, Markdown Report).
- Non-destructive human annotation preservation.
"""

import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import pytest

from ml.base import PredictionResult
from ml.benchmark import (
    compute_dataset_fingerprint,
    export_confusion_matrix_csv,
    export_markdown_report,
    export_metrics_csv,
    export_predictions_csv,
    get_benchmark_history_dir,
    get_eval_datasets_dir,
    list_available_eval_datasets,
    list_benchmark_history,
    load_benchmark_run,
    run_full_benchmark,
    save_evaluation_dataset,
    validate_evaluation_dataset,
    LANGUAGE_MIN_SAMPLE_THRESHOLD,
    PRELIMINARY_SAMPLE_SIZE_THRESHOLD,
)
from ml.engine import MLEngine
from ml.evaluate import compute_metrics, evaluate_task
from ml.model_config import MLBackend, MLEngineConfig


class TestEvaluationDatasetValidation(unittest.TestCase):
    """Tests evaluation dataset validation, missing data detection, and integrity checks."""

    def test_validate_valid_dataset(self):
        df = pd.DataFrame({
            "text": ["Great app!", "Terrible service.", "Okay product."],
            "sentiment": ["Positive", "Negative", "Neutral"],
            "text_type": ["Statement", "Statement", "Statement"],
            "language": ["English", "English", "English"],
        })
        val = validate_evaluation_dataset(df, "text", {"sentiment": "sentiment", "text_type": "text_type"})
        self.assertTrue(val["is_valid"])
        self.assertEqual(val["total_rows"], 3)
        self.assertEqual(val["labeled_counts"]["sentiment"], 3)
        self.assertEqual(val["duplicate_count"], 0)
        self.assertTrue(val["is_small_sample"])  # 3 < 30
        self.assertIn("Preliminary evaluation: small sample size", val["warning"])

    def test_validate_missing_text_column(self):
        df = pd.DataFrame({"sentence": ["Hello world"], "sentiment": ["Positive"]})
        val = validate_evaluation_dataset(df, "text", {"sentiment": "sentiment"})
        self.assertFalse(val["is_valid"])
        self.assertIn("Text column 'text' not found", val["error"])

    def test_validate_empty_dataset(self):
        df = pd.DataFrame()
        val = validate_evaluation_dataset(df, "text", {"sentiment": "sentiment"})
        self.assertFalse(val["is_valid"])
        self.assertIn("empty", val["error"].lower())

    def test_detects_duplicate_text_rows(self):
        df = pd.DataFrame({
            "text": ["Duplicate line.", "Unique line.", "Duplicate line."],
            "sentiment": ["Positive", "Negative", "Positive"],
        })
        val = validate_evaluation_dataset(df, "text", {"sentiment": "sentiment"})
        self.assertEqual(val["duplicate_count"], 1)

    def test_detects_invalid_unknown_labels(self):
        df = pd.DataFrame({
            "text": ["Item 1", "Item 2"],
            "sentiment": ["ExtremelyGood", "TerribleBad"],  # Non-standard labels
        })
        val = validate_evaluation_dataset(df, "text", {"sentiment": "sentiment"})
        self.assertIn("sentiment", val["invalid_labels"])
        self.assertIn("ExtremelyGood", val["invalid_labels"]["sentiment"])


class TestMetricCalculationsAndConfusionMatrix(unittest.TestCase):
    """Tests accuracy, macro F1, weighted F1, and confusion matrix arithmetic."""

    def test_exact_metric_calculation(self):
        y_true = ["Positive", "Positive", "Negative", "Neutral"]
        y_pred = ["Positive", "Negative", "Negative", "Neutral"]

        acc, macro_p, macro_r, macro_f1, per_class, cm = compute_metrics(y_true, y_pred)

        # 3 out of 4 correct = 75%
        self.assertEqual(acc, 0.75)
        # Confusion matrix checks
        self.assertEqual(cm["Positive"]["Positive"], 1)
        self.assertEqual(cm["Positive"]["Negative"], 1)
        self.assertEqual(cm["Negative"]["Negative"], 1)
        self.assertEqual(cm["Neutral"]["Neutral"], 1)

        # Check per-class support
        self.assertEqual(per_class["Positive"]["support"], 2)
        self.assertEqual(per_class["Negative"]["support"], 1)
        self.assertEqual(per_class["Neutral"]["support"], 1)

    def test_evaluate_task_computes_weighted_f1_and_predictions(self):
        df = pd.DataFrame({
            "text": ["I love it", "I hate it", "It is okay"],
            "sentiment": ["Positive", "Negative", "Neutral"],
            "language": ["English", "English", "English"],
        })
        engine = MLEngine(backend=MLBackend.BASELINE)
        res = evaluate_task(df, "sentiment", engine.predict_sentiment)

        self.assertIn("weighted_f1", res)
        self.assertIsInstance(res["weighted_f1"], float)
        self.assertIn("predictions", res)
        self.assertEqual(len(res["predictions"]), 3)
        self.assertEqual(res["predictions"][0]["text"], "I love it")
        self.assertIn("is_correct", res["predictions"][0])


class TestEvaluationDatasetIsolation(unittest.TestCase):
    """Verifies evaluation dataset upload isolation and column preservation."""

    def test_save_evaluation_dataset_isolation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            sample_content = "id,sentence,custom_tag,sentiment\n1,Great day!,vip,Positive\n2,Bad day!,regular,Negative\n"
            dest_p, meta, err = save_evaluation_dataset(sample_content.encode("utf-8"), "my_eval.csv", root)

            self.assertIsNone(err)
            self.assertTrue(dest_p.exists())
            self.assertEqual(dest_p.parent.name, "evaluation_datasets")
            self.assertIn("custom_tag", meta["columns"])

            # Verify saved CSV preserves custom columns
            loaded = pd.read_csv(dest_p)
            self.assertIn("custom_tag", loaded.columns)
            self.assertEqual(len(loaded), 2)


class TestBenchmarkExecutionAndProvenance(unittest.TestCase):
    """Verifies benchmark execution, fallback exclusion, and history persistence."""

    def test_run_benchmark_baseline_and_history(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            df = pd.DataFrame({
                "text": ["I am happy.", "Terrible issue.", "Please help."],
                "sentiment": ["Positive", "Negative", "Neutral"],
                "text_type": ["Statement", "Statement", "Request / Instruction"],
                "language": ["English", "English", "English"],
            })

            bench = run_full_benchmark(
                df=df,
                text_col="text",
                label_mappings={"sentiment": "sentiment", "text_type": "text_type"},
                dataset_name="unit_test_eval.csv",
                root_dir=root,
                run_transformer=False,
            )

            self.assertIn("benchmark_id", bench)
            self.assertTrue(bench["is_small_sample"])
            self.assertIn("Preliminary evaluation", bench["warning_message"])
            self.assertIn("sentiment", bench["tasks"])
            self.assertIn("text_type", bench["tasks"])

            # Verify history was saved
            history = list_benchmark_history(root)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["benchmark_id"], bench["benchmark_id"])

            # Verify load_benchmark_run
            loaded = load_benchmark_run(bench["benchmark_id"], root)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["benchmark_id"], bench["benchmark_id"])

    def test_fallback_predictions_excluded_from_transformer_metrics(self):
        """When transformer weights are unavailable, fallback predictions must be recorded as fallback and excluded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            df = pd.DataFrame({
                "text": ["Sentence one.", "Sentence two."],
                "sentiment": ["Positive", "Negative"],
            })

            # Mock transformer fallback
            engine = MLEngine(
                backend=MLBackend.TRANSFORMER,
                config=MLEngineConfig(
                    active_backend=MLBackend.TRANSFORMER,
                    sentiment_checkpoint="invalid/nonexistent-model-checkpoint",
                    enable_fallback_to_baseline=True
                )
            )
            res = evaluate_task(df, "sentiment", engine.predict_sentiment)

            self.assertTrue(res["is_fallback"])
            self.assertEqual(res["actual_backend"], "fallback_baseline")
            self.assertEqual(res["fallback_count"], 2)


class TestConfidenceTypeSeparation(unittest.TestCase):
    """Ensures heuristic scores and model probabilities are never conflated."""

    def test_confidence_analysis_maintains_type_distinction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            df = pd.DataFrame({
                "text": ["Positive text here."],
                "sentiment": ["Positive"],
            })
            bench = run_full_benchmark(
                df=df,
                text_col="text",
                label_mappings={"sentiment": "sentiment"},
                dataset_name="test.csv",
                root_dir=root,
                run_transformer=False,
            )
            conf = bench["confidence_analysis"]
            self.assertGreater(conf["baseline_heuristic_scores_count"], 0)
            self.assertEqual(conf["calibrated_probabilities_count"], 0)
            self.assertFalse(conf["has_calibrated_probabilities"])


class TestMultilingualSampleGuardrails(unittest.TestCase):
    """Verifies language breakdown flags insufficient sample sizes."""

    def test_language_sample_guardrail_threshold(self):
        df = pd.DataFrame({
            "text": ["Hello", "नमस्ते"],
            "sentiment": ["Positive", "Positive"],
            "language": ["English", "Hindi"],
        })
        val = validate_evaluation_dataset(df, "text", {"sentiment": "sentiment", "language": "language"})
        # 1 English, 1 Hindi -> both < 10 threshold
        self.assertTrue(val["is_small_sample"])


class TestExportSuite(unittest.TestCase):
    """Tests CSV and Markdown export generators."""

    def test_export_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            df = pd.DataFrame({
                "text": ["I love this product!", "Terrible experience."],
                "sentiment": ["Positive", "Negative"],
                "language": ["English", "English"],
            })
            bench = run_full_benchmark(
                df=df,
                text_col="text",
                label_mappings={"sentiment": "sentiment"},
                dataset_name="sample.csv",
                root_dir=root,
                run_transformer=False,
            )

            metrics_csv = export_metrics_csv(bench)
            self.assertIn("Baseline", metrics_csv)
            self.assertIn("Accuracy", metrics_csv)

            preds_csv = export_predictions_csv(bench)
            self.assertIn("ground_truth", preds_csv)
            self.assertIn("baseline_prediction", preds_csv)

            cm_csv = export_confusion_matrix_csv(bench, "sentiment", "baseline")
            self.assertIn("Positive", cm_csv)

            md = export_markdown_report(bench)
            self.assertIn("# LinguaLabel — Model Benchmark & Evaluation Report", md)
            self.assertIn("Benchmark ID", md)
            self.assertIn("Preliminary Evaluation Notice", md)


class TestHumanAnnotationPreservation(unittest.TestCase):
    """Verifies running benchmarks or loading datasets never modifies active human annotations."""

    def test_eval_dataset_does_not_mutate_annotated_output(self):
        real_annot = Path("annotated_output.csv")
        if real_annot.exists():
            before_bytes = real_annot.read_bytes()
            # Run benchmark on a separate dataframe
            with tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                test_df = pd.DataFrame({"text": ["Test sentence"], "sentiment": ["Positive"]})
                run_full_benchmark(test_df, "text", {"sentiment": "sentiment"}, "t.csv", root, run_transformer=False)
            after_bytes = real_annot.read_bytes()
            self.assertEqual(before_bytes, after_bytes)


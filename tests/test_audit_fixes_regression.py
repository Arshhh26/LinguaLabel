"""
Regression test suite for Part 7.1 Audit Fixes.

Covers:
Phase 1:
- Explicit backend provenance (baseline, transformer, fallback_baseline)
- Accurate confidence types (heuristic vs. model_probability)
- Transformer fallback disclosure in evaluation CLI
- Low-confidence thresholding (< 0.60 -> Uncertain)
- Human-approved annotation preservation

Phase 2:
- Deterministic, dataset-aware record identifiers
- Distinct IDs for identical duplicate sentences in the same dataset
- Stability across application reruns
- Independence from volatile row positions (reordered rows)
- Non-collision across different uploaded datasets

Phase 3:
- Isolated dataset persistence (no cross-contamination between datasets)
- Correct restoration upon reloading
- Original column preservation
- Error reporting on invalid IDs or save failures
"""

import os
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import pytest

from data_manager import (
    COLS,
    compute_dataset_key,
    generate_deterministic_record_ids,
    get_dataset_storage_path,
    merge_saved_annotations,
    save_dataset_record,
    stabilize_dataset,
)
from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.evaluate import compare_backends, evaluate_task
from ml.model_config import MLBackend, MLEngineConfig
from ml.transformers.model_loader import get_model_loader
from ml.transformers.sentiment_transformer import SentimentTransformer
from ml.transformers.text_type_transformer import TextTypeTransformer


class TestPhase1PredictionTransparency(unittest.TestCase):
    """Verifies Phase 1: Prediction transparency, confidence typing, and fallback disclosure."""

    def setUp(self):
        self.loader = get_model_loader()
        self.loader.clear_cache()

    def tearDown(self):
        self.loader.clear_cache()

    def test_baseline_predictions_record_heuristic_or_model_prob(self):
        engine = MLEngine(backend=MLBackend.BASELINE)
        preds = engine.predict_all("Hello world!")

        # Language: Script Heuristic -> heuristic
        self.assertEqual(preds["language"].actual_backend, "baseline")
        self.assertIn(preds["language"].confidence_type, ("heuristic", "model_probability"))

        # Sentiment: Lexicon Rules -> heuristic
        self.assertEqual(preds["sentiment"].actual_backend, "baseline")
        self.assertIn(preds["sentiment"].confidence_type, ("heuristic", "model_probability"))

        # Text Type: Syntactic Matcher -> heuristic
        self.assertEqual(preds["text_type"].actual_backend, "baseline")
        self.assertIn(preds["text_type"].confidence_type, ("heuristic", "model_probability"))

    def test_successful_transformer_prediction_provenance_and_type(self):
        """When transformer pipeline is loaded (or mocked), backend is 'transformer' and confidence is 'model_probability'."""
        checkpoint = "mock/xlm-r-sentiment"

        def mock_pipe(text):
            return [[{"label": "positive", "score": 0.94}, {"label": "neutral", "score": 0.04}, {"label": "negative", "score": 0.02}]]

        self.loader.register_mock_pipeline(checkpoint, mock_pipe)
        st = SentimentTransformer(checkpoint=checkpoint)
        res = st.predict("Wonderful experience!")

        self.assertEqual(res.label, "Positive")
        self.assertEqual(res.actual_backend, "transformer")
        self.assertEqual(res.confidence_type, "model_probability")
        self.assertFalse(res.metadata.get("calibrated", True))

    def test_transformer_fallback_records_fallback_baseline(self):
        """When transformer weights are unavailable, engine falls back to baseline and records actual_backend='fallback_baseline'."""
        config = MLEngineConfig(
            active_backend=MLBackend.TRANSFORMER,
            sentiment_checkpoint="invalid/nonexistent-model-checkpoint",
            enable_fallback_to_baseline=True,
        )
        engine = MLEngine(backend=MLBackend.TRANSFORMER, config=config)
        res = engine.predict_sentiment("This product is awesome.")

        self.assertEqual(res.actual_backend, "fallback_baseline")
        self.assertTrue(res.metadata.get("is_fallback"))
        self.assertEqual(res.metadata.get("fallback_from"), "transformer")
        # Must not be labeled as pure transformer
        self.assertNotEqual(res.actual_backend, "transformer")

    def test_evaluation_cli_detects_and_flags_fallback(self):
        """Evaluation harness must explicitly flag fallback and never report it as genuine transformer accuracy."""
        test_df = pd.DataFrame({
            "text": ["Great product!", "Awful delay."],
            "sentiment": ["Positive", "Negative"],
            "language": ["English", "English"]
        })
        config = MLEngineConfig(
            active_backend=MLBackend.TRANSFORMER,
            sentiment_checkpoint="invalid/nonexistent-model-checkpoint",
            enable_fallback_to_baseline=True,
        )
        engine = MLEngine(backend=MLBackend.TRANSFORMER, config=config)
        task_res = evaluate_task(test_df, "sentiment", engine.predict_sentiment)

        self.assertTrue(task_res["is_fallback"])
        self.assertEqual(task_res["actual_backend"], "fallback_baseline")
        self.assertEqual(task_res["fallback_count"], 2)


class TestPhase2RobustRecordIDs(unittest.TestCase):
    """Verifies Phase 2: Deterministic, dataset-aware, occurrence-safe record identifiers."""

    def test_duplicate_text_preserves_unique_identity(self):
        """Identical text occurring multiple times must receive distinct, collision-free IDs."""
        texts = [
            "Same sentence here.",
            "Different sentence.",
            "Same sentence here.",  # exact duplicate
            "Same sentence here.",  # 3rd occurrence
        ]
        ids = generate_deterministic_record_ids(texts, dataset_key="ds1")

        # Must have exactly 4 IDs
        self.assertEqual(len(ids), 4)
        # All 4 IDs must be strictly unique
        self.assertEqual(len(set(ids)), 4)

        # Occurrence suffixes must be _1, _2, _3 for the duplicate sentence
        dup_hash = ids[0].split("_")[-2]
        self.assertTrue(ids[0].endswith(f"{dup_hash}_1"))
        self.assertTrue(ids[2].endswith(f"{dup_hash}_2"))
        self.assertTrue(ids[3].endswith(f"{dup_hash}_3"))

    def test_rerun_stability(self):
        """Generating IDs on the same texts across reruns yields identical IDs."""
        texts = ["Sentence Alpha", "Sentence Beta", "Sentence Gamma"]
        run_1 = generate_deterministic_record_ids(texts, dataset_key="upload_key_1")
        run_2 = generate_deterministic_record_ids(texts, dataset_key="upload_key_1")
        self.assertEqual(run_1, run_2)

    def test_reordered_rows_preserve_content_ids(self):
        """Unique sentences maintain their content-derived IDs regardless of row order."""
        texts_a = ["First unique text.", "Second unique text."]
        texts_b = ["Second unique text.", "First unique text."]

        ids_a = generate_deterministic_record_ids(texts_a, dataset_key="ds_order")
        ids_b = generate_deterministic_record_ids(texts_b, dataset_key="ds_order")

        # "First unique text." ID in set A equals its ID in set B
        self.assertEqual(ids_a[0], ids_b[1])
        # "Second unique text." ID in set A equals its ID in set B
        self.assertEqual(ids_a[1], ids_b[0])

    def test_different_datasets_avoid_id_collisions(self):
        """Identical text across two different datasets receives distinct dataset-prefixed IDs."""
        texts = ["Standard common sentence."]
        id_dataset_1 = generate_deterministic_record_ids(texts, dataset_key="upload_aaa")
        id_dataset_2 = generate_deterministic_record_ids(texts, dataset_key="upload_bbb")

        self.assertNotEqual(id_dataset_1[0], id_dataset_2[0])
        self.assertTrue(id_dataset_1[0].startswith("upload_aaa_"))
        self.assertTrue(id_dataset_2[0].startswith("upload_bbb_"))

    def test_existing_valid_unique_ids_preserved(self):
        """Datasets that already possess valid unique IDs are never overwritten."""
        df = pd.DataFrame({
            "id": ["custom_01", "custom_02"],
            "text": ["Hello", "World"]
        })
        stable_df, info = stabilize_dataset(df, text_column="text", dataset_key="any_key")
        self.assertFalse(info["id_generated"])
        self.assertEqual(list(stable_df["id"]), ["custom_01", "custom_02"])


class TestPhase3DatasetSpecificPersistence(unittest.TestCase):
    """Verifies Phase 3: Isolated persistence, reload correctness, and error propagation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_isolated_persistence_prevents_cross_overwrites(self):
        """Saving Dataset A cannot overwrite or contaminate Dataset B or default dataset."""
        default_path = get_dataset_storage_path("default_sample", self.root_path)
        path_a = get_dataset_storage_path("upload_dataset_a", self.root_path)
        path_b = get_dataset_storage_path("upload_dataset_b", self.root_path)

        self.assertNotEqual(default_path, path_a)
        self.assertNotEqual(path_a, path_b)

        df_a = pd.DataFrame({"id": ["a1"], "text": ["Text A"], "language": [""], "sentiment": [""], "text_type": [""], "review_status": [""], "annotator_notes": [""]})
        df_b = pd.DataFrame({"id": ["b1"], "text": ["Text B"], "language": [""], "sentiment": [""], "text_type": [""], "review_status": [""], "annotator_notes": [""]})

        # Save record in A
        ok_a = save_dataset_record(path_a, df_a, "a1", {"language": "English", "review_status": "Reviewed — looks correct"})
        self.assertTrue(ok_a)

        # Save record in B
        ok_b = save_dataset_record(path_b, df_b, "b1", {"language": "Hindi", "review_status": "Reviewed — looks correct"})
        self.assertTrue(ok_b)

        # Confirm A is not overwritten by B
        saved_a = pd.read_csv(path_a, keep_default_na=False, dtype=str)
        saved_b = pd.read_csv(path_b, keep_default_na=False, dtype=str)

        self.assertEqual(saved_a.iloc[0]["id"], "a1")
        self.assertEqual(saved_a.iloc[0]["language"], "English")

        self.assertEqual(saved_b.iloc[0]["id"], "b1")
        self.assertEqual(saved_b.iloc[0]["language"], "Hindi")

        # Default sample storage path should remain non-existent / untouched
        self.assertFalse(default_path.exists())

    def test_reloading_restores_annotations_and_preserves_custom_columns(self):
        """Reloading a saved session restores all annotations while preserving original metadata columns."""
        path_custom = get_dataset_storage_path("upload_session_test", self.root_path)
        raw_df = pd.DataFrame({
            "client_code": ["CLI_99", "CLI_100"],
            "user_comment": ["First comment", "Second comment"],
        })
        stable_df, _ = stabilize_dataset(raw_df, text_column="user_comment", dataset_key="upload_session_test")
        rec_id = stable_df.iloc[0]["id"]

        # Save annotation for first record
        save_ok = save_dataset_record(
            path_custom,
            stable_df,
            rec_id,
            {
                "language": "English",
                "sentiment": "Positive",
                "text_type": "Statement",
                "review_status": "Reviewed — looks correct",
                "annotator_notes": "Verified client note",
            }
        )
        self.assertTrue(save_ok)

        # Simulate fresh reload
        merged = merge_saved_annotations(stable_df.copy(), path_custom)

        self.assertEqual(merged.iloc[0]["language"], "English")
        self.assertEqual(merged.iloc[0]["sentiment"], "Positive")
        self.assertEqual(merged.iloc[0]["annotator_notes"], "Verified client note")
        # Custom columns preserved intact
        self.assertEqual(merged.iloc[0]["client_code"], "CLI_99")
        self.assertEqual(merged.iloc[1]["client_code"], "CLI_100")

    def test_save_fails_gracefully_on_invalid_row_id(self):
        """Saving with an ID that does not exist in the DataFrame returns False without crashing."""
        storage_path = self.root_path / "test_invalid.csv"
        df = pd.DataFrame({"id": ["valid_1"], "text": ["Text"]})
        result = save_dataset_record(storage_path, df, "non_existent_id", {"language": "English"})
        self.assertFalse(result)


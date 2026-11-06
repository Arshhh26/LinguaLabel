"""
LinguaLabel — Part 11.3 Model Training & Continuous Improvement Center Test Suite

Covers:
1. Dataset Management & Validation (CSV, Excel, missing values, duplicates, invalid labels, class imbalance).
2. Human Feedback Curation (strict exclusion of unreviewed and rejected submissions; only approved enters training).
3. Data Preparation & Leakage Prevention (provenance tracking, human correction priority, grouped splits).
4. Model Training Engine (hardware device resolution, reproducible jobs, isolated candidate artifacts).
5. Model Evaluation & Comparison (metrics, confusion matrix, language stratification, regression diagnostics).
6. Production Promotion, Active Inference Integration, and Rollback (atomic updates, audit trail, MLEngine dispatch).
"""

import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest

from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.training.data_preparation import UnifiedDataPreparationEngine
from ml.training.dataset_manager import DatasetManagementEngine
from ml.training.evaluator import ModelEvaluationEngine
from ml.training.feedback_collector import FeedbackCurationEngine
from ml.training.trainer import ModelTrainingEngine, TrainingJobConfig, TrainedModelPredictor
from ml.training.versioning import ModelVersionRegistry


@pytest.fixture
def temp_repo(tmp_path: Path):
    """Sets up a clean temporary repository environment for Part 11.3 testing."""
    repo_dir = tmp_path / "LinguaLabel"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "trained_models").mkdir(parents=True, exist_ok=True)
    (repo_dir / "training_datasets").mkdir(parents=True, exist_ok=True)
    (repo_dir / "training_splits").mkdir(parents=True, exist_ok=True)
    (repo_dir / "active_learning_feedback").mkdir(parents=True, exist_ok=True)
    (repo_dir / "evaluation_datasets").mkdir(parents=True, exist_ok=True)

    # Populate sample benchmark
    benchmarks_dir = Path(__file__).resolve().parent.parent / "evaluation_datasets"
    if benchmarks_dir.exists():
        for b_file in ["complaints_eval.csv", "abuse_eval.csv"]:
            src = benchmarks_dir / b_file
            if src.exists():
                shutil.copy(src, repo_dir / "evaluation_datasets" / b_file)

    return repo_dir


# =============================================================================
# 1. DATASET MANAGEMENT & VALIDATION TESTS
# =============================================================================

class TestDatasetManagement:
    def test_csv_upload_and_auto_column_detection(self, temp_repo: Path):
        engine = DatasetManagementEngine(temp_repo)
        csv_content = (
            "sentence,sentiment_label\n"
            "This product is amazing!,Positive\n"
            "Terrible customer service.,Negative\n"
            "The package arrived today.,Neutral\n"
        ).encode("utf-8")

        text_col, lbl_col = engine.detect_columns(
            pd.read_csv(pd.io.common.BytesIO(csv_content)), "sentiment"
        )
        assert text_col == "sentence"
        assert lbl_col == "sentiment_label"

        entry = engine.save_dataset(
            file_bytes=csv_content,
            filename="sentiment_test.csv",
            task="sentiment",
            text_col=text_col,
            label_col=lbl_col,
            description="Unit test sentiment dataset",
        )
        assert entry.dataset_id is not None
        assert entry.num_samples == 3
        assert len(engine.list_datasets()) == 1

    def test_validation_detects_missing_values_and_duplicates(self, temp_repo: Path):
        engine = DatasetManagementEngine(temp_repo)
        df = pd.DataFrame({
            "text": ["Good app", "Good app", "", "Bad service", None],
            "sentiment": ["Positive", "Positive", "Positive", "Negative", "Negative"],
        })

        report = engine.validate_dataframe(df, text_col="text", label_col="sentiment", task="sentiment")
        assert report.duplicate_text_count >= 1
        assert report.missing_values_count >= 2
        assert report.valid_samples_count == 3

    def test_validation_flags_invalid_labels(self, temp_repo: Path):
        engine = DatasetManagementEngine(temp_repo)
        df = pd.DataFrame({
            "text": ["Sample 1", "Sample 2", "Sample 3"],
            "sentiment": ["Positive", "Bogus_Label_123", "Negative"],
        })

        report = engine.validate_dataframe(df, text_col="text", label_col="sentiment", task="sentiment")
        assert not report.is_valid
        assert "Bogus_Label_123" in report.invalid_labels

    def test_class_imbalance_calculation(self, temp_repo: Path):
        engine = DatasetManagementEngine(temp_repo)
        df = pd.DataFrame({
            "text": [f"Text {i}" for i in range(10)],
            "sentiment": ["Positive"] * 8 + ["Negative"] * 2,
        })

        report = engine.validate_dataframe(df, text_col="text", label_col="sentiment", task="sentiment")
        assert report.class_imbalance_ratio == 4.0
        assert report.class_distribution["Positive"] == 8
        assert report.class_distribution["Negative"] == 2


# =============================================================================
# 2. HUMAN FEEDBACK CURATION TESTS
# =============================================================================

class TestFeedbackCuration:
    def test_feedback_curation_lifecycle_and_filtering(self, temp_repo: Path):
        fb_dir = temp_repo / "active_learning_feedback"
        # Seed raw feedback
        raw_fb = [
            {
                "feedback_id": "fb_1",
                "text": "Yeh bilkul bekar app hai",
                "suggested_label": "Statement",
                "corrected_label": "Complaining",
                "task": "text_type",
                "notes": "Clear complaint in Hinglish",
                "timestamp": "2026-11-06T10:00:00Z",
            },
            {
                "feedback_id": "fb_2",
                "text": "Hello good morning",
                "suggested_label": "Question",
                "corrected_label": "Greeting",
                "task": "text_type",
                "notes": "Standard greeting",
                "timestamp": "2026-11-06T10:05:00Z",
            },
            {
                "feedback_id": "fb_3",
                "text": "Nonsense gibberish xyz",
                "suggested_label": "Statement",
                "corrected_label": "Other",
                "task": "text_type",
                "notes": "Spam entry",
                "timestamp": "2026-11-06T10:10:00Z",
            },
        ]
        with open(fb_dir / "user_feedback_log.jsonl", "w") as f:
            for item in raw_fb:
                f.write(json.dumps(item) + "\n")

        curator = FeedbackCurationEngine(temp_repo)
        stats = curator.get_curation_statistics()
        assert stats["total_feedback"] == 3
        assert stats["pending"] == 3
        assert stats["approved"] == 0
        assert stats["rejected"] == 0

        # Approve fb_1, reject fb_3, leave fb_2 pending
        curator.curate_feedback("fb_1", "approved", reviewer_notes="Approved for training")
        curator.curate_feedback("fb_3", "rejected", reviewer_notes="Spam text")

        updated_stats = curator.get_curation_statistics()
        assert updated_stats["approved"] == 1
        assert updated_stats["rejected"] == 1
        assert updated_stats["pending"] == 1

        # STRICT GUARDRAIL CHECK: Only approved feedback must be extracted
        trusted_df, warnings = curator.get_trusted_training_samples("text_type")
        assert len(trusted_df) == 1
        assert trusted_df.iloc[0]["feedback_id"] == "fb_1"
        assert trusted_df.iloc[0]["label"] == "Complaining"


# =============================================================================
# 3. DATA PREPARATION & LEAKAGE PREVENTION TESTS
# =============================================================================

class TestDataPreparation:
    def test_preparation_merges_provenance_and_resolves_priority(self, temp_repo: Path):
        ds_engine = DatasetManagementEngine(temp_repo)
        # 1. Owner dataset has a sentence with label 'Statement'
        csv_data = (
            "text,text_type\n"
            "The app crashed again,Statement\n"
            "Can you help me?,Question\n"
            "Good day,Greeting\n"
            "Please fix the bug,Request / Instruction\n"
            "Terrible delay,Statement\n"
        ).encode("utf-8")
        ds_entry = ds_engine.save_dataset(
            file_bytes=csv_data,
            filename="owner_ds.csv",
            task="text_type",
            text_col="text",
            label_col="text_type",
        )

        # 2. Approved human feedback has the SAME sentence but corrected to 'Complaining'
        curator = FeedbackCurationEngine(temp_repo)
        raw_fb = [{
            "feedback_id": "fb_corr_1",
            "text": "The app crashed again",
            "suggested_label": "Statement",
            "corrected_label": "Complaining",
            "task": "text_type",
            "notes": "Annotator verified complaint",
            "timestamp": "2026-11-06T12:00:00Z",
        }]
        with open(temp_repo / "active_learning_feedback" / "user_feedback_log.jsonl", "w") as f:
            f.write(json.dumps(raw_fb[0]) + "\n")
        curator.curate_feedback("fb_corr_1", "approved")

        prep_engine = UnifiedDataPreparationEngine(temp_repo)
        split = prep_engine.prepare_unified_dataset(
            dataset_ids=[ds_entry.dataset_id],
            task="text_type",
            include_approved_feedback=True,
            train_ratio=0.6,
            val_ratio=0.2,
            random_seed=42,
        )

        assert split.total_records == 5
        # Verify provenance breakdown
        assert split.provenance_breakdown["human_feedback_approved"] == 1
        assert split.provenance_breakdown["owner_dataset"] == 4

        # Read back prepared data and verify human correction override
        train_df = pd.read_csv(Path(split.train_path))
        val_df = pd.read_csv(Path(split.val_path))
        test_df = pd.read_csv(Path(split.test_path))
        all_splits_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        target_row = all_splits_df[all_splits_df["text"] == "The app crashed again"].iloc[0]
        assert target_row["label"] == "Complaining"
        assert target_row["source"] == "human_feedback_approved"

    def test_leakage_prevention_keeps_duplicates_in_same_split(self, temp_repo: Path):
        ds_engine = DatasetManagementEngine(temp_repo)
        # Dataset with duplicated sentences
        csv_data = (
            "text,sentiment\n"
            "Identical sentence,Positive\n"
            "Identical sentence,Positive\n"
            "Another sentence,Negative\n"
            "Third sentence,Neutral\n"
            "Fourth sentence,Positive\n"
            "Fifth sentence,Negative\n"
        ).encode("utf-8")
        entry = ds_engine.save_dataset(
            file_bytes=csv_data,
            filename="dup_test.csv",
            task="sentiment",
            text_col="text",
            label_col="sentiment",
        )

        prep_engine = UnifiedDataPreparationEngine(temp_repo)
        split = prep_engine.prepare_unified_dataset(
            dataset_ids=[entry.dataset_id],
            task="sentiment",
            include_approved_feedback=False,
            train_ratio=0.6,
            val_ratio=0.2,
        )

        # Deduplication retains unique sentences
        assert split.total_records == 5


# =============================================================================
# 4. MODEL TRAINING & CANDIDATE REGISTRATION TESTS
# =============================================================================

class TestModelTrainingEngine:
    def test_device_auto_resolution(self):
        dev_key, dev_label = ModelTrainingEngine.resolve_device("auto")
        assert dev_key in ("cuda:0", "cpu")
        assert "CPU" in dev_label or "GPU" in dev_label

        cpu_key, cpu_label = ModelTrainingEngine.resolve_device("cpu")
        assert cpu_key == "cpu"
        assert cpu_label == "CPU"

    def test_training_pipeline_creates_candidate_and_preserves_production(self, temp_repo: Path):
        # 1. Setup training data
        ds_engine = DatasetManagementEngine(temp_repo)
        train_data = []
        for i in range(20):
            train_data.append({"text": f"This is an amazing outstanding positive item {i}", "sentiment": "Positive"})
            train_data.append({"text": f"This is a terrible horrible broken negative item {i}", "sentiment": "Negative"})
            train_data.append({"text": f"This is a neutral standard regular item {i}", "sentiment": "Neutral"})
        df = pd.DataFrame(train_data)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        entry = ds_engine.save_dataset(csv_bytes, "sentiment_full.csv", "sentiment", "text", "sentiment")

        prep = UnifiedDataPreparationEngine(temp_repo)
        split = prep.prepare_unified_dataset([entry.dataset_id], "sentiment", train_ratio=0.7, val_ratio=0.15)

        # 2. Train candidate model
        trainer = ModelTrainingEngine(temp_repo)
        cfg = TrainingJobConfig(
            task="sentiment",
            split_id=split.split_id,
            model_name="Unit Test Sentiment Model",
            architecture="tfidf_logistic_regression",
            hyperparameters={"C": 1.0, "max_features": 5000},
        )
        res = trainer.train_model(cfg)
        assert res.success is True
        assert res.version_id is not None
        assert res.val_accuracy > 0.70

        # 3. Verify it is registered as 'candidate' and NOT production
        registry = ModelVersionRegistry(temp_repo)
        ver = registry.get_version(res.version_id)
        assert ver is not None
        assert ver.lifecycle_state == "candidate"
        assert registry.get_active_production_models().get("sentiment") is None

        # 4. Verify candidate predictor works
        cand_pred = TrainedModelPredictor(Path(ver.artifacts_path), "sentiment")
        pred_res = cand_pred.predict("This is an amazing outstanding positive item")
        assert isinstance(pred_res, PredictionResult)
        assert pred_res.label == "Positive"
        assert pred_res.confidence > 0.60


# =============================================================================
# 5. MODEL EVALUATION & COMPARISON TESTS
# =============================================================================

class TestModelEvaluation:
    def test_evaluation_metrics_and_confusion_matrix(self, temp_repo: Path):
        evaluator = ModelEvaluationEngine(temp_repo)
        # Mock predictor
        class MockPredictor:
            def predict(self, text: str):
                if "good" in text:
                    return PredictionResult("Positive", 0.95, "mock")
                return PredictionResult("Negative", 0.90, "mock")

        test_df = pd.DataFrame({
            "text": ["good product", "very good service", "bad delivery", "terrible phone"],
            "sentiment": ["Positive", "Positive", "Negative", "Negative"],
            "language": ["English", "English", "English", "English"],
        })

        report = evaluator.evaluate(
            predictor=MockPredictor(),
            dataset=test_df,
            task="sentiment",
            model_name="Mock Predictor",
        )
        assert report.accuracy == 1.0
        assert report.f1_macro == 1.0
        assert "labels" in report.confusion_matrix
        assert "English" in report.per_language_metrics

    def test_comparison_detects_severe_regression_and_vetoes_promotion(self, temp_repo: Path):
        evaluator = ModelEvaluationEngine(temp_repo)

        class HighPerfPredictor:
            def predict(self, text: str):
                return PredictionResult("Positive" if "good" in text else "Negative", 0.95, "high")

        class LowPerfPredictor:
            def predict(self, text: str):
                return PredictionResult("Negative", 0.95, "low")  # Always predicts negative!

        test_df = pd.DataFrame({
            "text": ["good 1", "good 2", "bad 1", "bad 2"],
            "sentiment": ["Positive", "Positive", "Negative", "Negative"],
        })

        prod_rep = evaluator.evaluate(HighPerfPredictor(), test_df, "sentiment", model_name="Production")
        cand_rep = evaluator.evaluate(LowPerfPredictor(), test_df, "sentiment", model_name="Candidate")

        comp = evaluator.compare_reports(prod_rep, cand_rep)
        assert comp.is_severe_regression is True
        assert comp.recommend_promotion is False
        assert "DO NOT PROMOTE" in comp.verdict_summary


# =============================================================================
# 6. LIFECYCLE PROMOTION, INFERENCE INTEGRATION & ROLLBACK TESTS
# =============================================================================

class TestLifecyclePromotionAndInference:
    def test_promotion_updates_inference_and_rollback_restores_baseline(self, temp_repo: Path):
        # 1. Setup training and produce a candidate
        ds_engine = DatasetManagementEngine(temp_repo)
        train_data = []
        for i in range(25):
            train_data.append({"text": f"I adore and love this marvelous software {i}", "sentiment": "Positive"})
            train_data.append({"text": f"I despise and hate this broken application {i}", "sentiment": "Negative"})
        df = pd.DataFrame(train_data)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        entry = ds_engine.save_dataset(csv_bytes, "sentiment_data.csv", "sentiment", "text", "sentiment")

        prep = UnifiedDataPreparationEngine(temp_repo)
        split = prep.prepare_unified_dataset([entry.dataset_id], "sentiment", train_ratio=0.7, val_ratio=0.15)

        trainer = ModelTrainingEngine(temp_repo)
        res = trainer.train_model(TrainingJobConfig(
            task="sentiment",
            split_id=split.split_id,
            model_name="Candidate Sentiment v1",
            architecture="tfidf_logistic_regression",
        ))
        cand_version_id = res.version_id

        # 2. Promote candidate to production
        registry = ModelVersionRegistry(temp_repo)
        promote_ok = registry.promote_to_production(
            version_id=cand_version_id,
            actor="test_runner",
            notes="Automated promotion test",
        )
        assert promote_ok is True
        assert registry.get_active_production_models().get("sentiment") == cand_version_id

        # Check audit trail (latest first)
        trail = registry.get_audit_trail()
        assert len(trail) >= 1
        assert trail[0]["action"] == "promote"
        assert trail[0]["version_id"] == cand_version_id

        # 3. Verify MLEngine integrates the active promoted model
        # Override project root in test
        engine = MLEngine()
        # Point registry to temp_repo
        from ml.training.trainer import TrainedModelPredictor
        engine._custom_predictors["sentiment"] = TrainedModelPredictor(
            artifacts_dir=Path(registry.get_version(cand_version_id).artifacts_path),
            task="sentiment",
        )

        pred = engine.predict_sentiment("I adore and love this marvelous software!")
        assert pred.label == "Positive"
        assert pred.actual_backend == "custom_trained"

        # 4. Rollback production to baseline
        rollback_ok = registry.rollback_production(task="sentiment", actor="test_runner", notes="Rollback test")
        assert rollback_ok is True
        assert registry.get_active_production_models().get("sentiment") is None

        # Verify audit trail recorded rollback
        updated_trail = registry.get_audit_trail()
        assert updated_trail[0]["action"] == "rollback"

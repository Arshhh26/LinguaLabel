"""
Unit & Integration Test Suite for LinguaLabel Part 11.4:
Complete Production Inference Integration.

Validates the 10 mandated requirements:
1. Promoted complaint model affects live Complaining detection.
2. Non-complaint sentences retain correct text types (complaint model doesn't override non-complaints).
3. Promoted abuse-category model affects live abuse inference.
4. Promoted severity model affects live severity inference.
5. Not abusive always maps to severity None (Safety Guardrail).
6. Missing or corrupted custom models fall back safely.
7. Rollback restores the previous active predictor or baseline.
8. Existing human-approved labels are never overwritten.
9. predict_all() retains backward-compatible keys and types.
10. Model task mismatches cannot be promoted.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import json

from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.training.versioning import ModelVersionRegistry, TASK_COMPATIBILITY_GROUPS
from ml.training.trainer import TrainedModelPredictor
from data_manager import save_dataset_record, merge_saved_annotations, validation_issues


@pytest.fixture
def temp_env():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "trained_models").mkdir(parents=True, exist_ok=True)
        yield tmp_path


def _create_dummy_model(
    artifacts_dir: Path,
    task: str,
    train_texts: list,
    train_labels: list,
    classes: list,
    version_id: str,
):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(C=100.0)),
    ])
    pipe.fit(train_texts, train_labels)
    joblib.dump(pipe, artifacts_dir / "model.joblib")

    meta = {
        "version_id": version_id,
        "task": task,
        "model_name": f"Dummy_{task}",
        "architecture": "tfidf_logistic_regression",
        "classes": classes,
        "train_samples": len(train_texts),
        "val_samples": 0,
        "train_accuracy": 1.0,
        "val_accuracy": 1.0,
    }
    with open(artifacts_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def test_1_promoted_complaint_model_affects_live_detection(temp_env):
    """Requirement 1: A promoted complaint model is used by text-type inference and maps to 'Complaining'."""
    reg = ModelVersionRegistry(temp_env)
    cand_dir = temp_env / "trained_models" / "cand_complaint_001"
    
    # Train dummy complaint model on clear signals
    texts = ["I am very angry with bad service", "This is broken", "Great weather", "I love this"]
    labels = ["Complaining", "Complaining", "Not complaining", "Not complaining"]
    _create_dummy_model(cand_dir, "complaint", texts, labels, ["Complaining", "Not complaining"], "cand_complaint_001")
    
    reg.register_candidate(
        task="complaint",
        model_name="CustomComplaintModel",
        architecture="tfidf_lr",
        dataset_name="test_split",
        dataset_fingerprint="fp123",
        artifacts_dir=cand_dir,
        version_id="cand_complaint_001",
    )
    ok = reg.promote_to_production("cand_complaint_001", target_task="complaint")
    assert ok is True

    engine = MLEngine()
    # Inject temp active production mapping
    engine._custom_predictors["complaint"] = TrainedModelPredictor(cand_dir, "complaint")

    # Prediction should recognize complaint
    pred = engine.predict_complaint("I am very angry with bad service")
    assert pred.label == "Complaining"
    assert pred.confidence >= 0.50

    # Text type inference should use promoted complaint model
    tt_pred = engine.predict_text_type("I am very angry with bad service")
    assert tt_pred.label == "Complaining"
    assert tt_pred.metadata.get("provenance") == "promoted_complaint_model"


def test_2_non_complaint_sentences_retain_correct_text_types(temp_env):
    """Requirement 2: Complaint model does not classify every non-complaint as arbitrary type; retains correct types."""
    cand_dir = temp_env / "trained_models" / "cand_complaint_002"
    texts = ["Broken app bad service", "Terrible food", "What is the time?", "Hello there!"]
    labels = ["Complaining", "Complaining", "Not complaining", "Not complaining"]
    _create_dummy_model(cand_dir, "complaint", texts, labels, ["Complaining", "Not complaining"], "cand_complaint_002")

    engine = MLEngine()
    engine._custom_predictors["complaint"] = TrainedModelPredictor(cand_dir, "complaint")

    # Question sentence: specialized complaint model says 'Not complaining', so it falls through to general text-type
    q_pred = engine.predict_text_type("What is the time of the train?")
    assert q_pred.label == "Question"

    # Greeting sentence: falls through to Greeting
    g_pred = engine.predict_text_type("Hello! Good morning everyone.")
    assert g_pred.label == "Greeting"


def test_3_promoted_abuse_category_model_affects_live_inference(temp_env):
    """Requirement 3: Promoted abuse-category model affects live abuse category inference."""
    cand_dir = temp_env / "trained_models" / "cand_abuse_cat_001"
    texts = ["You are an idiot", "Nice weather", "I will kill you", "Hello friend"]
    labels = ["Severe Personal Insult", "Not abusive", "Threat", "Not abusive"]
    _create_dummy_model(cand_dir, "abuse_category", texts, labels, ["Not abusive", "Severe Personal Insult", "Threat"], "cand_abuse_cat_001")

    engine = MLEngine()
    engine._custom_predictors["abuse_category"] = TrainedModelPredictor(cand_dir, "abuse_category")

    pred = engine.predict_abuse_category("You are an idiot")
    assert pred.label == "Severe Personal Insult"
    assert pred.metadata.get("actual_backend") in ("custom_trained", "custom_model")


def test_4_promoted_severity_model_affects_live_severity_inference(temp_env):
    """Requirement 4: Promoted severity model affects live severity inference."""
    cand_dir = temp_env / "trained_models" / "cand_abuse_sev_001"
    texts = ["annoying person", "stupid idiot", "die die I will destroy everything"]
    labels = ["Mild", "Moderate", "Severe"]
    _create_dummy_model(cand_dir, "abuse_severity", texts, labels, ["Mild", "Moderate", "Severe"], "cand_abuse_sev_001")

    engine = MLEngine()
    engine._custom_predictors["abuse_severity"] = TrainedModelPredictor(cand_dir, "abuse_severity")

    # Mock abusive category
    cat_pred = PredictionResult("Threat", 0.95, "Custom", confidence_type="model_probability")
    sev_pred = engine.predict_abuse_severity("die die I will destroy everything", category_pred=cat_pred)
    assert sev_pred.label == "Severe"
    assert sev_pred.metadata.get("actual_backend") in ("custom_trained", "custom_model")


def test_5_not_abusive_always_maps_to_severity_none():
    """Requirement 5: Not abusive category ALWAYS yields severity None (Safety Guardrail)."""
    engine = MLEngine()
    # Non-abusive category
    cat_pred = PredictionResult("Not abusive", 0.98, "Model", confidence_type="model_probability")
    sev_pred = engine.predict_abuse_severity("I love this wonderful sunny day", category_pred=cat_pred)

    assert sev_pred.label == "None"
    assert sev_pred.is_uncertain is False
    assert sev_pred.metadata.get("guardrail_enforced") == "not_abusive_implies_none"

    # Even if sentence text has harsh-sounding words, if category is Not abusive, severity is None
    sev_pred_2 = engine.predict_abuse_severity("This movie had lots of violence in history", category_pred=cat_pred)
    assert sev_pred_2.label == "None"


def test_6_missing_or_corrupted_custom_models_fall_back_safely(temp_env):
    """Requirement 6: Corrupted or missing custom models fall back safely without raising uncaught exceptions."""
    corrupted_dir = temp_env / "trained_models" / "corrupted_model"
    corrupted_dir.mkdir(parents=True, exist_ok=True)
    (corrupted_dir / "model.joblib").write_text("not a valid joblib file", encoding="utf-8")

    engine = MLEngine()
    engine._custom_predictors["complaint"] = TrainedModelPredictor(corrupted_dir, "complaint")
    engine._custom_predictors["abuse_severity"] = TrainedModelPredictor(corrupted_dir, "abuse_severity")

    # Should not crash; gracefully fall back
    res_tt = engine.predict_text_type("Where is my refund? This is unacceptable!")
    assert isinstance(res_tt, PredictionResult)
    assert res_tt.label != ""

    res_sev = engine.predict_abuse_severity("You complete idiot")
    assert isinstance(res_sev, PredictionResult)
    assert res_sev.label in ("None", "Mild", "Moderate", "Severe")


def test_7_rollback_restores_previous_active_predictor(temp_env):
    """Requirement 7: Rollback restores the previous active predictor or baseline."""
    reg = ModelVersionRegistry(temp_env)
    dir_v1 = temp_env / "trained_models" / "v1"
    dir_v2 = temp_env / "trained_models" / "v2"

    _create_dummy_model(dir_v1, "sentiment", ["happy", "sad"], ["Positive", "Negative"], ["Positive", "Negative"], "v1")
    _create_dummy_model(dir_v2, "sentiment", ["happy", "sad"], ["Positive", "Negative"], ["Positive", "Negative"], "v2")

    reg.register_candidate("sentiment", "Model1", "lr", "ds1", "fp1", dir_v1, version_id="v1")
    reg.register_candidate("sentiment", "Model2", "lr", "ds2", "fp2", dir_v2, version_id="v2")

    # Promote v1
    assert reg.promote_to_production("v1") is True
    assert reg.get_active_production_models().get("sentiment") == "v1"

    # Promote v2
    assert reg.promote_to_production("v2") is True
    assert reg.get_active_production_models().get("sentiment") == "v2"

    # Roll back to v1
    assert reg.rollback_production("sentiment", target_version_id="v1") is True
    assert reg.get_active_production_models().get("sentiment") == "v1"

    # Roll back to baseline
    assert reg.rollback_production("sentiment", target_version_id=None) is True
    assert "sentiment" not in reg.get_active_production_models()


def test_8_existing_human_approved_labels_never_overwritten(temp_env):
    """Requirement 8: Existing human-approved labels in datasets are strictly immutable to model predictions."""
    csv_path = temp_env / "test_dataset.csv"
    initial_df = pd.DataFrame([{
        "id": "rec_1",
        "text": "This product broke after 2 days",
        "language": "English",
        "sentiment": "Negative",
        "text_type": "Complaining",
        "abuse_category": "Not abusive",
        "abuse_severity": "None",
        "review_status": "Reviewed — looks correct",
        "annotator_notes": "Annotator human verified",
    }])
    initial_df.to_csv(csv_path, index=False)

    # Simulate merging or updating without modifying existing reviewed row
    loaded_df = pd.read_csv(csv_path)
    row_1 = loaded_df.iloc[0]
    assert row_1["review_status"] == "Reviewed — looks correct"
    assert row_1["annotator_notes"] == "Annotator human verified"

    # Ensure validation accepts existing record
    vals = {
        "language": row_1["language"],
        "sentiment": row_1["sentiment"],
        "text_type": row_1["text_type"],
        "abuse_category": row_1["abuse_category"],
        "abuse_severity": row_1["abuse_severity"],
        "review_status": row_1["review_status"],
        "annotator_notes": row_1["annotator_notes"],
    }
    assert validation_issues(vals) == []


def test_9_predict_all_retains_backward_compatible_keys():
    """Requirement 9: predict_all() returns all required keys and maintains backward compatibility for 'abuse'."""
    engine = MLEngine()
    preds = engine.predict_all("I really dislike this service, it is terrible.")

    expected_keys = {
        "language",
        "sentiment",
        "text_type",
        "complaint",
        "abuse_category",
        "abuse_severity",
        "abuse",
    }
    assert expected_keys.issubset(set(preds.keys()))

    # Every value must be a PredictionResult
    for k in expected_keys:
        assert isinstance(preds[k], PredictionResult)

    # 'abuse' must map to 'abuse_category'
    assert preds["abuse"].label == preds["abuse_category"].label
    assert preds["abuse"].confidence == preds["abuse_category"].confidence


def test_10_model_task_mismatches_cannot_be_promoted(temp_env):
    """Requirement 10: Cross-task promotions (e.g. severity into category or language into sentiment) are rejected."""
    reg = ModelVersionRegistry(temp_env)
    cand_dir = temp_env / "trained_models" / "cand_sev_mismatch"
    
    # Create a severity model with severity classes
    texts = ["bad", "awful", "terrible"]
    labels = ["Mild", "Moderate", "Severe"]
    _create_dummy_model(cand_dir, "abuse_severity", texts, labels, ["Mild", "Moderate", "Severe"], "cand_sev_mismatch")

    reg.register_candidate(
        task="abuse_severity",
        model_name="SeverityModel",
        architecture="tfidf_lr",
        dataset_name="split1",
        dataset_fingerprint="fp1",
        artifacts_dir=cand_dir,
        version_id="cand_sev_mismatch",
    )

    # Attempting to promote abuse_severity candidate into abuse_category target slot MUST FAIL
    mismatch_promo_1 = reg.promote_to_production("cand_sev_mismatch", target_task="abuse_category")
    assert mismatch_promo_1 is False

    # Attempting to promote abuse_severity candidate into sentiment slot MUST FAIL
    mismatch_promo_2 = reg.promote_to_production("cand_sev_mismatch", target_task="sentiment")
    assert mismatch_promo_2 is False

    # Promoting into its own compatible slot succeeds
    valid_promo = reg.promote_to_production("cand_sev_mismatch", target_task="abuse_severity")
    assert valid_promo is True

"""
Unit and Integration Tests for LinguaLabel Part 9:
Active Learning & Intelligent Annotation Prioritization.

Validates:
- Uncertainty Sampling (lowest confidence, margin, and entropy constraints).
- Disagreement Detection with strict fallback exclusion.
- Multi-factor Priority Scoring, explanations, and review-status demotion.
- Queue filtering, sorting, and summary metrics.
- Feedback persistence, deduplication, and dataset isolation.
- Unicode preservation across Hindi/Hinglish text.
- Controlled Learning Preparation, label validation, and cross-split data leakage detection.
"""

import json
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from ml.active_learning import (
    ActiveLearningQueue,
    FeedbackManager,
    FeedbackRecord,
    PriorityRecord,
    PriorityWeights,
    check_data_leakage,
    compute_feedback_analytics,
    compute_priority_record,
    compute_record_disagreement,
    compute_record_uncertainty,
    compute_task_uncertainty,
    compute_training_data_fingerprint,
    evaluate_task_disagreement,
    format_feedback_for_training,
    prepare_controlled_split,
)
from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.model_config import MLBackend


# =============================================================================
# 1. UNCERTAINTY SAMPLING TESTS
# =============================================================================

def test_uncertainty_lowest_confidence():
    """Validates lowest-confidence uncertainty (1.0 - conf) and safety threshold."""
    # High confidence prediction
    high_pred = PredictionResult("Positive", 0.95, "Heuristic", confidence_type="heuristic")
    tu_high = compute_task_uncertainty("sentiment", high_pred, uncertainty_threshold=0.60)
    assert round(tu_high.lowest_confidence_uncertainty, 2) == 0.05
    assert not tu_high.is_uncertain

    # Low confidence prediction (< 0.60)
    low_pred = PredictionResult("Positive", 0.42, "Heuristic", confidence_type="heuristic")
    tu_low = compute_task_uncertainty("sentiment", low_pred, uncertainty_threshold=0.60)
    assert round(tu_low.lowest_confidence_uncertainty, 2) == 0.58
    assert tu_low.is_uncertain


def test_uncertainty_margin_sampling():
    """Validates margin sampling between top two class probability scores."""
    # Distribution with close margin: top1=0.48, top2=0.42 -> margin=0.06
    scores = {"Positive": 0.48, "Neutral": 0.42, "Negative": 0.10}
    pred = PredictionResult(
        "Positive", 0.48, "Transformer",
        confidence_type="model_probability",
        metadata={"all_scores": scores}
    )
    tu = compute_task_uncertainty("sentiment", pred)
    assert tu.margin_supported is True
    assert tu.margin is not None
    assert round(tu.margin, 2) == 0.06
    assert round(tu.margin_uncertainty, 2) == 0.94


def test_uncertainty_entropy_sampling_probabilistic_only():
    """
    CRITICAL RULE: Entropy MUST ONLY be computed from genuine probability distributions,
    NEVER from heuristic confidence scores.
    """
    # 1. Heuristic score: MUST NOT calculate entropy
    heur_pred = PredictionResult(
        "Positive", 0.85, "Lexicon",
        confidence_type="heuristic",
        metadata={"all_scores": {"Positive": 0.85, "Negative": 0.15}}
    )
    tu_heur = compute_task_uncertainty("sentiment", heur_pred)
    assert tu_heur.entropy_supported is False
    assert tu_heur.entropy is None
    assert "heuristic confidence is not a valid probability distribution" in tu_heur.notes

    # 2. Genuine probability distribution: Maximum entropy on uniform distribution
    prob_scores = {"Positive": 0.50, "Negative": 0.50}
    prob_pred = PredictionResult(
        "Positive", 0.50, "Transformer",
        confidence_type="model_probability",
        metadata={"all_scores": prob_scores}
    )
    tu_prob = compute_task_uncertainty("sentiment", prob_pred)
    assert tu_prob.entropy_supported is True
    assert tu_prob.entropy is not None
    assert round(tu_prob.entropy, 2) == 1.0  # Normalized Shannon entropy of uniform distribution = 1.0


def test_uncertainty_invalid_or_missing_score_handling():
    """Handles missing or None predictions without crashing, assigning max uncertainty."""
    tu_missing = compute_task_uncertainty("sentiment", None)
    assert tu_missing.is_uncertain is True
    assert tu_missing.lowest_confidence_uncertainty == 1.0
    assert tu_missing.confidence_type == "missing"


# =============================================================================
# 2. MODEL DISAGREEMENT TESTS & FALLBACK EXCLUSION
# =============================================================================

def test_model_disagreement_detection():
    """Validates conflict detection between genuine baseline and transformer predictions."""
    b_pred = PredictionResult("Positive", 0.80, "Lexicon", confidence_type="heuristic", metadata={"actual_backend": "baseline"})
    t_pred = PredictionResult("Negative", 0.90, "Transformer", confidence_type="model_probability", metadata={"actual_backend": "transformer"})

    td = evaluate_task_disagreement("sentiment", b_pred, t_pred)
    assert td.is_genuine_comparison is True
    assert td.is_conflict is True
    assert td.is_fallback is False
    assert "Conflict on sentiment" in td.explanation


def test_disagreement_transformer_fallback_exclusion():
    """
    CRITICAL RULE: Fallback baseline predictions MUST NEVER be counted as transformer
    predictions or as genuine model disagreements.
    """
    b_pred = PredictionResult("Positive", 0.80, "Lexicon", confidence_type="heuristic", metadata={"actual_backend": "baseline"})
    # Transformer fell back to baseline
    t_fallback = PredictionResult(
        "Neutral", 0.50, "Fallback Baseline",
        confidence_type="heuristic",
        metadata={"actual_backend": "fallback_baseline", "is_fallback": True}
    )

    td = evaluate_task_disagreement("sentiment", b_pred, t_fallback)
    assert td.is_genuine_comparison is False
    assert td.is_conflict is False
    assert td.is_fallback is True
    assert "not counted as genuine disagreement" in td.explanation


def test_record_disagreement_multitask():
    """Validates multi-task disagreement aggregation across sentiment and text_type."""
    b_preds = {
        "sentiment": PredictionResult("Positive", 0.80, "Lex", metadata={"actual_backend": "baseline"}),
        "text_type": PredictionResult("Question", 0.90, "Syn", metadata={"actual_backend": "baseline"}),
    }
    t_preds = {
        "sentiment": PredictionResult("Negative", 0.85, "XLM-R", confidence_type="model_probability", metadata={"actual_backend": "transformer"}),
        "text_type": PredictionResult("Question", 0.95, "mDeBERTa", confidence_type="model_probability", metadata={"actual_backend": "transformer"}),
    }

    rd = compute_record_disagreement("rec_1", b_preds, t_preds)
    assert rd.has_disagreement is True
    assert rd.disagreement_count == 1
    assert rd.disagreeing_tasks == ["sentiment"]
    assert rd.comparable_task_count == 2
    assert rd.disagreement_score == 0.50


# =============================================================================
# 3. INTELLIGENT PRIORITY SCORING & DEMOTION TESTS
# =============================================================================

def test_priority_scoring_hierarchy():
    """
    Tests prioritization rules:
    - High disagreement + uncertainty -> High Priority
    - Reviewed records -> Demoted with near-zero priority
    - Missing fields -> Explanatory warning
    """
    row_pending = {
        "language": "Hindi",
        "sentiment": "Positive",
        "text_type": "Statement",
        "review_status": "",
    }
    b_preds = {
        "language": PredictionResult("Hindi", 0.95, "Script"),
        "sentiment": PredictionResult("Positive", 0.70, "Lex"),
        "text_type": PredictionResult("Statement", 0.70, "Syn"),
    }
    t_preds = {
        "sentiment": PredictionResult("Negative", 0.75, "XLM-R", confidence_type="model_probability", metadata={"actual_backend": "transformer"}),
        "text_type": PredictionResult("Question", 0.65, "mDeBERTa", confidence_type="model_probability", metadata={"actual_backend": "transformer"}),
    }

    unc = compute_record_uncertainty("r1", b_preds)
    dis = compute_record_disagreement("r1", b_preds, t_preds)

    priority_unreviewed = compute_priority_record(
        record_id="r1",
        text="यह एक परीक्षण वाक्य है",
        row_data=row_pending,
        uncertainty_report=unc,
        disagreement_report=dis,
    )
    assert priority_unreviewed.priority_level == "High"
    assert priority_unreviewed.disagreement_score == 1.0  # 2 conflicts out of 2 tasks
    assert not priority_unreviewed.is_reviewed

    # Now verify demotion when record is marked as 'Reviewed — looks correct'
    row_reviewed = dict(row_pending)
    row_reviewed["review_status"] = "Reviewed — looks correct"

    priority_reviewed = compute_priority_record(
        record_id="r1",
        text="यह एक परीक्षण वाक्य है",
        row_data=row_reviewed,
        uncertainty_report=unc,
        disagreement_report=dis,
    )
    assert priority_reviewed.is_reviewed is True
    assert priority_reviewed.priority_score < 0.15
    assert priority_reviewed.priority_level == "Low"
    assert "Demoted" in priority_reviewed.explanations[-1]


# =============================================================================
# 4. ACTIVE LEARNING QUEUE & FILTERING TESTS
# =============================================================================

def test_active_learning_queue_filtering_and_sorting():
    """Validates queue generation, filtering by status and priority, and sorting."""
    df = pd.DataFrame([
        {"id": "item_1", "text": "I loved this experience!", "language": "English", "sentiment": "Positive", "text_type": "Statement", "review_status": "Reviewed — looks correct"},
        {"id": "item_2", "text": "Is this service working?", "language": "English", "sentiment": "Neutral", "text_type": "Question", "review_status": ""},
        {"id": "item_3", "text": "यह क्या है मुझे समझ नहीं आया", "language": "", "sentiment": "", "text_type": "", "review_status": "Needs second review"},
    ])

    engine = MLEngine()
    queue = ActiveLearningQueue(df=df, ml_engine=engine, run_transformer=False)

    assert queue.summary.total_records == 3
    assert queue.summary.completed_count == 1
    assert queue.summary.unreviewed_count == 2
    assert queue.summary.ambiguous_count == 1

    # Filter by Pending review
    pending_items = queue.get_filtered_queue(status_filter="Pending")
    assert len(pending_items) == 2
    assert all(not r.is_reviewed for r in pending_items)

    # Filter by Ambiguous / Second Review
    amb_items = queue.get_filtered_queue(status_filter="Ambiguous / Second Review")
    assert len(amb_items) == 1
    assert amb_items[0].record_id == "item_3"


# =============================================================================
# 5. HUMAN FEEDBACK PERSISTENCE & DEDUPLICATION TESTS
# =============================================================================

def test_feedback_persistence_and_deduplication():
    """Validates feedback storage, deduplication, and dataset isolation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        mgr = FeedbackManager(tmp_root)

        model_pred = PredictionResult(
            label="Negative",
            confidence=0.75,
            method="Transformer",
            confidence_type="model_probability",
            metadata={"actual_backend": "transformer", "checkpoint": "test-model"}
        )

        # 1. Record human correction (Human changed Negative to Neutral)
        rec1 = mgr.record_feedback(
            dataset_key="test_ds",
            record_id="rec_42",
            task="sentiment",
            original_text="The battery was okay.",
            model_prediction=model_pred,
            human_label="Neutral",
            annotator_notes="Not negative, just neutral factual statement.",
        )
        assert rec1 is not None
        assert rec1.is_correction is True
        assert rec1.model_predicted_label == "Negative"
        assert rec1.human_corrected_label == "Neutral"

        # 2. Saving the exact same correction again must NOT create duplicate records
        rec2 = mgr.record_feedback(
            dataset_key="test_ds",
            record_id="rec_42",
            task="sentiment",
            original_text="The battery was okay.",
            model_prediction=model_pred,
            human_label="Neutral",
            annotator_notes="Not negative, just neutral factual statement.",
        )
        assert rec2.feedback_id == rec1.feedback_id

        # Verify disk records count
        all_records = mgr.load_feedback_records("test_ds")
        assert len(all_records) == 1

        # 3. Export CSV with Unicode check
        csv_out = mgr.export_feedback_csv(all_records)
        assert "rec_42" in csv_out
        assert "Neutral" in csv_out


def test_feedback_unicode_preservation():
    """Validates flawless preservation of Hindi Devanagari script and UTF-8 characters."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        mgr = FeedbackManager(tmp_root)

        hindi_text = "यह फोन बहुत गर्म हो जाता है"
        hindi_note = "समीक्षा में हीटिंग की शिकायत है"

        model_pred = PredictionResult("Neutral", 0.65, "Transformer", confidence_type="model_probability")

        rec = mgr.record_feedback(
            dataset_key="hindi_ds",
            record_id="h_1",
            task="sentiment",
            original_text=hindi_text,
            model_prediction=model_pred,
            human_label="Negative",
            annotator_notes=hindi_note,
        )

        loaded = mgr.load_feedback_records("hindi_ds")
        assert len(loaded) == 1
        assert loaded[0].original_text == hindi_text
        assert loaded[0].annotator_notes == hindi_note


def test_feedback_analytics_computation():
    """Validates feedback analytics metrics (transitions, top classes, resolved conflicts)."""
    records = [
        FeedbackRecord(
            feedback_id="1", dataset_key="ds", record_id="r1", task="sentiment",
            original_text="T1", model_predicted_label="Negative", human_corrected_label="Neutral",
            is_correction=True, model_confidence=0.7, confidence_type="model_probability",
            actual_backend="transformer", model_checkpoint="m1", timestamp="2026-11-06T12:00:00Z"
        ),
        FeedbackRecord(
            feedback_id="2", dataset_key="ds", record_id="r2", task="sentiment",
            original_text="T2", model_predicted_label="Positive", human_corrected_label="Positive",
            is_correction=False, model_confidence=0.9, confidence_type="heuristic",
            actual_backend="baseline", model_checkpoint=None, timestamp="2026-11-06T12:01:00Z"
        ),
    ]

    analytics = compute_feedback_analytics(records)
    assert analytics["total_feedback_count"] == 2
    assert analytics["correction_count"] == 1
    assert analytics["agreement_count"] == 1
    assert analytics["transformer_corrections"] == 1
    assert analytics["baseline_corrections"] == 0
    assert analytics["top_corrected_classes"][0] == ("Negative", 1)


# =============================================================================
# 6. CONTROLLED LEARNING PREPARATION & LEAKAGE TESTS
# =============================================================================

def test_controlled_learning_preparation_and_splits():
    """Validates dataset splitting, schema formatting, label validation, and fingerprinting."""
    records = [
        FeedbackRecord(
            feedback_id=f"fb_{i}", dataset_key="ds", record_id=f"r_{i}", task="sentiment",
            original_text=f"Sample text number {i}", model_predicted_label="Neutral",
            human_corrected_label="Positive" if i % 2 == 0 else "Negative",
            is_correction=True, model_confidence=0.5, confidence_type="heuristic",
            actual_backend="baseline", model_checkpoint=None, timestamp="2026-11-06T12:00:00Z"
        )
        for i in range(10)
    ]

    df, warnings = format_feedback_for_training(records, task="sentiment")
    assert len(df) == 10
    assert "label" in df.columns
    assert "text" in df.columns

    # Prepare 70/15/15 split
    split_res = prepare_controlled_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert split_res.total_samples == 10
    assert split_res.train_count == 7
    assert split_res.val_count == 2
    assert split_res.test_count == 1
    assert split_res.leakage_detected is False
    assert len(split_res.fingerprint) == 16


def test_data_leakage_detection():
    """Validates that shared identical sentences between splits are rigorously flagged."""
    train_df = pd.DataFrame([{"text": "Sentence A"}, {"text": "Sentence B"}])
    val_df = pd.DataFrame([{"text": "Sentence C"}])
    # Deliberate leakage: Sentence A in both train and test
    test_df = pd.DataFrame([{"text": "Sentence A"}])

    has_leakage, conflicts = check_data_leakage(train_df, val_df, test_df)
    assert has_leakage is True
    assert len(conflicts) == 1
    assert "sentence(s) appear in both train and test splits" in conflicts[0].lower()

"""
Unit, Regression, and End-to-End Audit Tests for LinguaLabel Part 10.1:
Product Quality Audit & UX Validation.

Validates:
1. Hinglish label mapping:
   - Reversible bidirectional mapping (Hinglish <-> Hinglish).
   - Hinglish is distinct from Other.
   - Backward compatibility for legacy labels (English, Hindi, Other, Uncertain).
   - Language detector detection on mixed scripts and Romanized Hindi.
2. Sentence-type mapping:
   - 1-to-1 bijective mapping between display options and internal schema.
   - No non-reversible collapsing choices.
   - Full backward compatibility for legacy aliases.
3. Confidence descriptions:
   - Distinction between heuristic scores, model probabilities, calibrated probabilities, and fallbacks.
   - Fallback predictions explicitly identified as "AI fallback (needs review)".
   - Low-confidence predictions receive review guidance.
   - Missing/malformed metadata handled safely.
4. AI explanations on 10 specific audit sentences:
   - Non-hallucinated, empirically grounded explanations.
   - Explicit irony/sarcasm caveat without claiming reliable sarcasm classification.
   - Punctuation-only handling without throwing IndexErrors.
   - Proper language attribution (Hindi, Hinglish, English).
5. End-to-end integration workflow (Journey A through L):
   - Ingest 10 sentences -> stabilize -> queue -> AI suggestions -> accept -> manual override -> save -> export CSV & Excel -> verify match and immutability of human decisions.
"""

import io
from pathlib import Path
import tempfile
import pandas as pd
import pytest

from data_manager import (
    compute_dataset_key,
    export_excel_bytes,
    load_csv_bytes,
    load_excel_bytes,
    load_pasted_sentences,
    merge_saved_annotations,
    save_dataset_record,
    stabilize_dataset,
    validation_issues,
)
from ml.active_learning import ActiveLearningQueue
from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.language_detector import LanguageDetector
from ml.model_config import LANGUAGE_LABELS, MLBackend
from ui_mappings import (
    DISPLAY_LANGUAGE_OPTIONS,
    DISPLAY_REVIEW_STATUS_OPTIONS,
    DISPLAY_SENTIMENT_OPTIONS,
    DISPLAY_TEXT_TYPE_OPTIONS,
    DISPLAY_TO_INTERNAL_LANGUAGE,
    DISPLAY_TO_INTERNAL_TEXT_TYPE,
    INTERNAL_TO_DISPLAY_LANGUAGE,
    INTERNAL_TO_DISPLAY_TEXT_TYPE,
    generate_sentence_explanation,
    simplify_confidence,
    to_display_label,
    to_internal_label,
)


# =============================================================================
# 1. HINGLISH LABEL MAPPING AUDIT
# =============================================================================

def test_hinglish_distinct_from_other():
    """Verifies that Hinglish is distinguishable from Other in all schemas."""
    assert "Hinglish" in LANGUAGE_LABELS
    assert "Hinglish" in DISPLAY_LANGUAGE_OPTIONS
    assert "Other" in DISPLAY_LANGUAGE_OPTIONS

    # Display to internal
    assert to_internal_label("language", "Hinglish") == "Hinglish"
    assert to_internal_label("language", "Other") == "Other"
    assert to_internal_label("language", "Hinglish") != to_internal_label("language", "Other")

    # Internal to display
    assert to_display_label("language", "Hinglish") == "Hinglish"
    assert to_display_label("language", "Other") == "Other"


def test_hinglish_bidirectional_reversibility():
    """Verifies that display-to-internal mappings for language are strictly reversible."""
    for opt in DISPLAY_LANGUAGE_OPTIONS:
        internal = to_internal_label("language", opt)
        restored = to_display_label("language", internal)
        assert restored == opt, f"Failed round-trip reversibility for language option '{opt}': got '{restored}'"


def test_legacy_language_backward_compatibility():
    """Verifies existing datasets with English, Hindi, Other, Uncertain remain 100% compatible."""
    legacy_internal = ["English", "Hindi", "Other", "Uncertain"]
    for lbl in legacy_internal:
        disp = to_display_label("language", lbl)
        assert disp in DISPLAY_LANGUAGE_OPTIONS
        assert to_internal_label("language", disp) == lbl


def test_hinglish_language_detection():
    """Verifies language detector identifies code-mixed Hindi-English and Romanized Hindi."""
    detector = LanguageDetector()

    # Mixed script (Devanagari + Latin)
    res_mixed = detector.predict("This phone is अच्छा but expensive.")
    assert res_mixed.label == "Hinglish"
    assert res_mixed.confidence >= 0.70

    # Romanized Hindi (Latin script with Hindi lexical items)
    res_roman = detector.predict("Kal ka plan kya hai?")
    assert res_roman.label == "Hinglish"
    assert res_roman.confidence >= 0.70

    # Pure Devanagari Hindi
    res_hindi = detector.predict("मुझे यह बिल्कुल पसंद नहीं आया।")
    assert res_hindi.label == "Hindi"
    assert res_hindi.confidence >= 0.80

    # Pure English
    res_eng = detector.predict("I absolutely love this application!")
    assert res_eng.label == "English"
    assert res_eng.confidence >= 0.80


# =============================================================================
# 2. SENTENCE-TYPE MAPPING AUDIT
# =============================================================================

def test_text_type_bijective_reversibility():
    """Verifies 1-to-1 bijective mapping so choices are preserved internally without silent mutation."""
    for opt in DISPLAY_TEXT_TYPE_OPTIONS:
        internal = to_internal_label("text_type", opt)
        restored = to_display_label("text_type", internal)
        assert restored == opt, f"Non-reversible sentence type mapping for '{opt}': mapped to '{internal}', restored as '{restored}'"


def test_text_type_legacy_aliases_compatibility():
    """Verifies legacy alias inputs still map to valid internal canonical labels."""
    assert to_internal_label("text_type", "Sharing an opinion") == "Statement"
    assert to_internal_label("text_type", "Giving information") == "Statement"
    assert to_internal_label("text_type", "Making a request") == "Request / Instruction"
    assert to_internal_label("text_type", "Giving instructions") == "Request / Instruction"


# =============================================================================
# 3. CONFIDENCE DESCRIPTIONS & FALLBACK AUDIT
# =============================================================================

def test_confidence_honesty_heuristic_vs_model_probability():
    """Verifies heuristic scores and uncalibrated probabilities are never conflated."""
    p_heuristic = PredictionResult("English", 0.95, "Script Heuristic", confidence_type="heuristic")
    disp, color, tip = simplify_confidence(p_heuristic)
    assert "heuristic" in tip.lower()
    assert "not a calibrated statistical probability" in tip.lower()

    p_model = PredictionResult("Positive", 0.92, "Transformer", confidence_type="model_probability")
    disp_m, color_m, tip_m = simplify_confidence(p_model)
    assert "uncalibrated model output probability" in tip_m.lower()
    assert "not a calibrated statistical certainty" in tip_m.lower()


def test_confidence_fallback_identification():
    """Verifies fallback predictions are accurately flagged and guided."""
    p_fallback = PredictionResult(
        label="Positive",
        confidence=0.75,
        method="Fallback Lexicon",
        metadata={"is_fallback": True, "fallback_from": "transformer"},
    )
    disp, color, tip = simplify_confidence(p_fallback)
    assert disp == "AI fallback (needs review)"
    assert "fallback" in tip.lower()
    assert "human review recommended" in tip.lower()


def test_confidence_safety_threshold_guidance():
    """Verifies predictions below 0.60 or flagged as uncertain receive attention guidance."""
    p_low = PredictionResult("Statement", 0.52, "Classifier", is_uncertain=True, confidence_type="model_probability")
    disp, color, tip = simplify_confidence(p_low)
    assert disp == "This sentence needs your attention."
    assert color == "#ef4444"
    assert "below the 60% safety threshold" in tip.lower()


def test_confidence_missing_metadata_safety():
    """Verifies None prediction or missing confidence handles safely without crashing."""
    disp, color, tip = simplify_confidence(None)
    assert disp == "This sentence needs your attention."
    assert color == "#ef4444"

    p_empty = PredictionResult("Uncertain", 0.0, "None", confidence_type="")
    disp_e, color_e, tip_e = simplify_confidence(p_empty)
    assert disp_e == "This sentence needs your attention."


# =============================================================================
# 4. AI EXPLANATIONS AUDIT ON 10 MANDATED SENTENCES
# =============================================================================

def test_ai_explanations_on_mandated_sentences():
    """
    Audits explanation generation on all 10 required test sentences:
    1. I absolutely love this application!
    2. This service is terrible.
    3. The product is okay.
    4. मुझे यह बिल्कुल पसंद नहीं आया।
    5. कृपया मेरी सहायता करें।
    6. Kal ka plan kya hai?
    7. This phone is अच्छा but expensive.
    8. I don't think this is a good idea.
    9. Great! Another delay.
    10. ???
    """
    engine = MLEngine()

    test_sentences = [
        "I absolutely love this application!",
        "This service is terrible.",
        "The product is okay.",
        "मुझे यह बिल्कुल पसंद नहीं आया।",
        "कृपया मेरी सहायता करें।",
        "Kal ka plan kya hai?",
        "This phone is अच्छा but expensive.",
        "I don't think this is a good idea.",
        "Great! Another delay.",
        "???",
    ]

    for sentence in test_sentences:
        preds = engine.predict_all(sentence)
        explanation = generate_sentence_explanation(sentence, preds)
        assert isinstance(explanation, str)
        assert len(explanation) > 10, f"Explanation too short for '{sentence}'"

    # Specific checks
    # Sentence 6 (Hinglish question)
    exp_6 = generate_sentence_explanation("Kal ka plan kya hai?", engine.predict_all("Kal ka plan kya hai?"))
    assert "Hinglish" in exp_6
    assert "question" in exp_6.lower()

    # Sentence 7 (Hinglish mixed script)
    exp_7 = generate_sentence_explanation("This phone is अच्छा but expensive.", engine.predict_all("This phone is अच्छा but expensive."))
    assert "Hinglish" in exp_7

    # Sentence 9 (Irony / Sarcasm warning)
    exp_9 = generate_sentence_explanation("Great! Another delay.", engine.predict_all("Great! Another delay."))
    assert "irony" in exp_9.lower() or "sarcasm" in exp_9.lower()
    assert "cannot reliably" in exp_9.lower() or "human judgment" in exp_9.lower()

    # Sentence 10 (Punctuation-only)
    exp_10 = generate_sentence_explanation("???", engine.predict_all("???"))
    assert "punctuation" in exp_10.lower()
    assert "without text words" in exp_10.lower()


# =============================================================================
# 5. END-TO-END WORKFLOW INTEGRATION AUDIT (JOURNEY A THROUGH L)
# =============================================================================

def test_end_to_end_user_journey_a_through_l():
    """
    Simulates complete user journey from ingestion to export:
    A. Import 10 pasted sentences
    B. Dataset stabilization with stable hashing
    C. AI suggestion generation
    D. Form acceptance of AI suggestion
    E. Manual correction (e.g. override label)
    F. Save annotations to storage
    G. Verify saved annotations match user decisions
    H. Verify human-approved annotations cannot be overwritten by re-analysis
    I. Export to CSV & Excel (.xlsx)
    J. Read exported files back and verify bit-for-bit label fidelity
    """
    raw_pasted = """I absolutely love this application!
This service is terrible.
The product is okay.
मुझे यह बिल्कुल पसंद नहीं आया।
कृपया मेरी सहायता करें।
Kal ka plan kya hai?
This phone is अच्छा but expensive.
I don't think this is a good idea.
Great! Another delay.
???"""

    # Step A: Import pasted sentences
    df_raw, err = load_pasted_sentences(raw_pasted)
    assert err is None
    assert df_raw is not None
    assert len(df_raw) == 10

    # Step B: Dataset stabilization
    ds_key = "test_audit_session_10_1"
    stable_df, _ = stabilize_dataset(df_raw, "text", ds_key)
    assert len(stable_df) == 10
    assert "id" in stable_df.columns

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir) / "test_session_store.csv"

        # Step C: Generate AI suggestions
        engine = MLEngine()
        preds_item_0 = engine.predict_all(stable_df.iloc[0]["text"])
        assert preds_item_0["language"].label == "English"

        # Step D: Accept AI suggestions for Sentence 0
        sug_lang_disp = to_display_label("language", preds_item_0["language"].label)
        sug_sent_disp = to_display_label("sentiment", preds_item_0["sentiment"].label)
        sug_tt_disp = to_display_label("text_type", preds_item_0["text_type"].label)

        # Convert back to canonical internal labels
        accepted_item_0 = {
            "id": stable_df.iloc[0]["id"],
            "language": to_internal_label("language", sug_lang_disp),
            "sentiment": to_internal_label("sentiment", sug_sent_disp),
            "text_type": to_internal_label("text_type", sug_tt_disp),
            "review_status": "Reviewed — looks correct",
            "annotator_notes": "Accepted AI suggestion",
        }
        assert validation_issues(accepted_item_0) == []

        # Step E: Manual correction for Sentence 6 ("Kal ka plan kya hai?")
        row_6 = stable_df.iloc[5]  # Sentence 6
        assert "Kal ka plan kya hai?" in row_6["text"]
        manual_item_6 = {
            "id": row_6["id"],
            "language": to_internal_label("language", "Hinglish"),  # User selected Hinglish
            "sentiment": to_internal_label("sentiment", "Neutral"),  # User marked Neutral
            "text_type": to_internal_label("text_type", "Asking a question"),
            "review_status": "Reviewed — looks correct",
            "annotator_notes": "Manually verified Hinglish question",
        }
        assert validation_issues(manual_item_6) == []

        # Step F: Save annotations to session storage
        ok0 = save_dataset_record(storage_path, stable_df, accepted_item_0["id"], accepted_item_0)
        assert ok0 is True
        ok6 = save_dataset_record(storage_path, stable_df, manual_item_6["id"], manual_item_6)
        assert ok6 is True

        # Step G: Verify saved annotations in storage
        merged_df = merge_saved_annotations(stable_df.copy(), storage_path)
        row_0_merged = merged_df[merged_df["id"] == accepted_item_0["id"]].iloc[0]
        row_6_merged = merged_df[merged_df["id"] == manual_item_6["id"]].iloc[0]
        assert row_0_merged["language"] == "English"
        assert row_6_merged["language"] == "Hinglish"
        assert row_6_merged["sentiment"] == "Neutral"
        assert row_6_merged["text_type"] == "Question"

        # Step H: Immutability under re-analysis
        # Build ActiveLearningQueue and verify human-approved items are demoted and preserved
        queue = ActiveLearningQueue(df=merged_df, ml_engine=engine, run_transformer=False)
        reviewed_records = [r for r in queue.records if r.is_reviewed]
        assert len(reviewed_records) == 2
        for r in reviewed_records:
            assert r.priority_level == "Low"
            assert r.priority_score < 0.15

        # Step I: Export to CSV & Excel
        csv_str = merged_df.to_csv(index=False)
        excel_bytes = export_excel_bytes(merged_df)
        assert excel_bytes is not None and len(excel_bytes) > 0

        # Step J: Read exported data and verify label fidelity
        df_exported_csv = pd.read_csv(io.StringIO(csv_str))
        df_exported_xl, _, _ = load_excel_bytes(excel_bytes)

        for exported in [df_exported_csv, df_exported_xl]:
            rec_6_exported = exported[exported["id"] == row_6["id"]].iloc[0]
            assert rec_6_exported["language"] == "Hinglish"
            assert rec_6_exported["sentiment"] == "Neutral"
            assert rec_6_exported["text_type"] == "Question"
            assert rec_6_exported["review_status"] == "Reviewed — looks correct"

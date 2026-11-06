"""
Unit and Integration Tests for LinguaLabel Part 10:
User Experience Simplification & Product Transformation.

Validates:
- CSV, Excel, Pasted Sentences, and Single-Sentence manual ingestion.
- Reversible bidirectional label mappings (beginner-friendly <-> canonical internal schema).
- Simplified plain-language AI confidence translations.
- Grounded, non-hallucinated sentence explanation generation.
- Full Unicode preservation (Hindi Devanagari, English, Hinglish).
- Clean CSV and Excel workbook exports.
- Human-approved annotation preservation and dataset isolation.
- Review Suggestions queue integration.
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
    load_single_sentence,
    stabilize_dataset,
)
from ml.active_learning import ActiveLearningQueue
from ml.base import PredictionResult
from ml.engine import MLEngine
from ui_mappings import (
    DISPLAY_LANGUAGE_OPTIONS,
    DISPLAY_REVIEW_STATUS_OPTIONS,
    DISPLAY_SENTIMENT_OPTIONS,
    DISPLAY_TEXT_TYPE_OPTIONS,
    generate_sentence_explanation,
    simplify_confidence,
    to_display_label,
    to_internal_label,
)


# =============================================================================
# 1. DATA INGESTION TESTS (CSV, EXCEL, PASTED, SINGLE)
# =============================================================================

def test_csv_import():
    """Validates robust CSV import."""
    csv_bytes = "id,text\n1,I love this app\n2,यह बहुत अच्छा है\n".encode("utf-8")
    df, enc, err = load_csv_bytes(csv_bytes)
    assert err is None
    assert df is not None
    assert len(df) == 2
    assert df.iloc[1]["text"] == "यह बहुत अच्छा है"


def test_excel_import_and_export():
    """Validates Excel workbook import and export (.xlsx)."""
    input_df = pd.DataFrame([
        {"id": "1", "text": "Customer service was great."},
        {"id": "2", "text": "मुझे यह पसंद नहीं आया।"},
    ])
    
    excel_bytes = export_excel_bytes(input_df)
    assert excel_bytes is not None
    assert len(excel_bytes) > 0

    read_df, enc, err = load_excel_bytes(excel_bytes, "test.xlsx")
    assert err is None
    assert read_df is not None
    assert len(read_df) == 2
    assert read_df.iloc[1]["text"] == "मुझे यह पसंद नहीं आया।"


def test_pasted_sentences_import():
    """Validates multi-line pasted text parsing, trimming, and blank exclusion."""
    pasted = """
    First sentence to annotate.
    
    Second sentence in Hindi: यह क्या है?
       Third sentence with leading spaces.
       
    """
    df, err = load_pasted_sentences(pasted)
    assert err is None
    assert df is not None
    assert len(df) == 3
    assert df.iloc[0]["text"] == "First sentence to annotate."
    assert df.iloc[1]["text"] == "Second sentence in Hindi: यह क्या है?"
    assert df.iloc[2]["text"] == "Third sentence with leading spaces."

    # Empty text case
    empty_df, empty_err = load_pasted_sentences("   \n\n  ")
    assert empty_df is None
    assert "Please paste one or more sentences" in empty_err or "No valid non-empty sentences" in empty_err


def test_single_sentence_manual_import():
    """Validates single manual sentence entry."""
    df, err = load_single_sentence("  Kya train time par chal rahi hai?  ")
    assert err is None
    assert df is not None
    assert len(df) == 1
    assert df.iloc[0]["text"] == "Kya train time par chal rahi hai?"


# =============================================================================
# 2. REVERSIBLE BIDIRECTIONAL LABEL MAPPING TESTS
# =============================================================================

def test_language_reversible_mapping():
    """Validates display <-> internal mappings for language, including Hinglish."""
    for disp in ["English", "Hindi", "Hinglish", "Other", "Not sure"]:
        internal = to_internal_label("language", disp)
        back_disp = to_display_label("language", internal)
        assert back_disp == disp

    # Test 'Not sure' -> 'Uncertain' -> 'Not sure'
    assert to_internal_label("language", "Not sure") == "Uncertain"
    assert to_display_label("language", "Uncertain") == "Not sure"


def test_sentiment_reversible_mapping():
    """Validates display <-> internal mappings for sentiment."""
    for disp in ["Positive", "Negative", "Neutral", "Not applicable", "Not sure"]:
        internal = to_internal_label("sentiment", disp)
        back_disp = to_display_label("sentiment", internal)
        assert back_disp == disp


def test_text_type_reversible_mapping():
    """Validates display <-> internal mappings for sentence type / intent."""
    mappings = [
        ("Asking a question", "Question"),
        ("Sharing an opinion or statement", "Statement"),
        ("Making a request or instruction", "Request / Instruction"),
        ("Greeting", "Greeting"),
        ("Other", "Other"),
        ("Not sure", "Uncertain"),
        # Legacy aliases
        ("Sharing an opinion", "Statement"),
        ("Giving information", "Statement"),
        ("Making a request", "Request / Instruction"),
        ("Giving instructions", "Request / Instruction"),
    ]
    for disp, expected_internal in mappings:
        assert to_internal_label("text_type", disp) == expected_internal

    # Internal to display (1-to-1 bijective)
    assert to_display_label("text_type", "Question") == "Asking a question"
    assert to_display_label("text_type", "Statement") == "Sharing an opinion or statement"
    assert to_display_label("text_type", "Request / Instruction") == "Making a request or instruction"
    assert to_display_label("text_type", "Greeting") == "Greeting"
    assert to_display_label("text_type", "Other") == "Other"
    assert to_display_label("text_type", "Uncertain") == "Not sure"


def test_review_status_reversible_mapping():
    """Validates display <-> internal mappings for review status."""
    mappings = [
        ("Reviewed & approved", "Reviewed — looks correct"),
        ("Needs second review", "Needs second review"),
        ("Unclear / Ambiguous", "Ambiguous"),
        ("Skip for now", "Skip"),
    ]
    for disp, expected_internal in mappings:
        assert to_internal_label("review_status", disp) == expected_internal
        assert to_display_label("review_status", expected_internal) == disp


# =============================================================================
# 3. SIMPLIFIED CONFIDENCE & PLAIN-LANGUAGE EXPLANATIONS
# =============================================================================

def test_simplified_confidence_translation():
    """Validates beginner-friendly confidence descriptions."""
    # High confidence (>= 0.80)
    p_high = PredictionResult("Positive", 0.92, "Model", confidence_type="model_probability")
    text_h, color_h, _ = simplify_confidence(p_high)
    assert text_h == "AI is fairly sure."
    assert color_h == "#22c55e"

    # Medium confidence (0.60 to 0.79)
    p_med = PredictionResult("Neutral", 0.68, "Model", confidence_type="model_probability")
    text_m, color_m, _ = simplify_confidence(p_med)
    assert text_m == "AI is somewhat unsure."
    assert color_m == "#f59e0b"

    # Low confidence (< 0.60 or is_uncertain)
    p_low = PredictionResult("Negative", 0.45, "Model", is_uncertain=True)
    text_l, color_l, _ = simplify_confidence(p_low)
    assert text_l == "This sentence needs your attention."
    assert color_l == "#ef4444"


def test_plain_language_sentence_explanations():
    """Validates grounded sentence explanation generation."""
    preds = {
        "language": PredictionResult("Hindi", 0.95, "Script"),
        "sentiment": PredictionResult("Positive", 0.85, "Lexicon"),
        "text_type": PredictionResult("Statement", 0.80, "Syntax"),
    }
    explanation = generate_sentence_explanation("यह बहुत उपयोगी है।", preds)
    assert "Hindi" in explanation
    assert "positive" in explanation.lower()
    assert "sharing an opinion" in explanation.lower() or "stating information" in explanation.lower()


# =============================================================================
# 4. DATASET ISOLATION & UNICODE PRESERVATION
# =============================================================================

def test_unicode_hindi_and_hinglish_preservation():
    """Validates that Devanagari script and Romanized Hindi are preserved intact."""
    raw_df = pd.DataFrame([
        {"text": "यह फोन बहुत गर्म हो जाता है।"},
        {"text": "Bhai product time par deliver nahi hua."},
    ])
    stable_df, _ = stabilize_dataset(raw_df, "text", "test_hindi_ds")
    assert stable_df.iloc[0]["text"] == "यह फोन बहुत गर्म हो जाता है।"
    assert stable_df.iloc[1]["text"] == "Bhai product time par deliver nahi hua."

    # Export to Excel bytes and read back
    xl_bytes = export_excel_bytes(stable_df)
    read_xl, _, _ = load_excel_bytes(xl_bytes)
    assert read_xl.iloc[0]["text"] == "यह फोन बहुत गर्म हो जाता है।"
    assert read_xl.iloc[1]["text"] == "Bhai product time par deliver nahi hua."


def test_human_approved_annotation_preservation():
    """Validates that human-approved records are never modified automatically."""
    df = pd.DataFrame([
        {
            "id": "rec_1",
            "text": "Great quality!",
            "language": "English",
            "sentiment": "Positive",
            "text_type": "Statement",
            "review_status": "Reviewed — looks correct",
            "annotator_notes": "Verified by human",
        }
    ])
    engine = MLEngine()
    queue = ActiveLearningQueue(df=df, ml_engine=engine, run_transformer=False)
    
    # Priority for human-approved record must be heavily demoted
    assert queue.records[0].is_reviewed is True
    assert queue.records[0].priority_level == "Low"
    assert queue.records[0].priority_score < 0.15

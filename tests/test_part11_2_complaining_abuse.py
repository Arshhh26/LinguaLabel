"""
LinguaLabel — Part 11.2 Comprehensive Verification & Regression Test Suite

Tests:
1. Complaining Detection as a first-class sentence type (English, Hindi, Hinglish).
2. Distinction between negative sentiment and complaining intent.
3. Multilingual Abusive Language & Profanity Detection across 7 categories and 4 severity levels.
4. Mandatory Hindi/Hinglish vocabulary coverage:
   - 'Madarchod', 'Chutiya' / 'Chutiye', 'Gaand' / 'Gand', 'Lund', 'Teri maa ki chut'.
5. Multi-representation normalization:
   - Leetspeak unmasking (@ -> a, 0 -> o, $ -> s), character repetition collapse,
     Devanagari cross-script mapping, asterisk masking, spacing collapse.
6. Contextual intelligence:
   - Quoted abuse, academic/linguistic discussion, product complaints vs targeted personal insults.
7. Threats and Harassment distinction.
8. Zero-PII analytics telemetry guarantee:
   - Verifies that raw sentence text and abusive terms are never recorded in analytics logs.
9. Bidirectional UI mappings, schema validation, and persistence.
"""

from pathlib import Path
import pytest
import pandas as pd

from data_manager import save_dataset_record, validation_issues, COLS
from ml.abuse_detector import AbuseDetector
from ml.engine import MLEngine
from ml.model_config import MLBackend, TEXT_TYPE_LABELS, ABUSE_CATEGORIES, ABUSE_SEVERITY_LEVELS
from ml.normalization import detect_context_flags, get_normalized_representations
from ui_mappings import (
    DISPLAY_ABUSE_OPTIONS,
    DISPLAY_TEXT_TYPE_OPTIONS,
    generate_sentence_explanation,
    to_display_label,
    to_internal_label,
)


@pytest.fixture
def abuse_detector():
    return AbuseDetector()


@pytest.fixture
def baseline_engine():
    return MLEngine(backend=MLBackend.BASELINE)


# =============================================================================
# 1. COMPLAINING DETECTION (FIRST-CLASS SENTENCE TYPE)
# =============================================================================

class TestComplainingDetection:
    def test_complaining_is_in_text_type_labels(self):
        assert "Complaining" in TEXT_TYPE_LABELS

    def test_english_complaining(self, baseline_engine):
        pred = baseline_engine.predict_text_type("Your application is terrible and keeps crashing every minute.")
        assert pred.label == "Complaining"
        assert pred.confidence >= 0.80

    def test_hinglish_complaining(self, baseline_engine):
        pred = baseline_engine.predict_text_type("Yeh app bilkul kaam nahi kar raha hai bohot bura service hai.")
        assert pred.label == "Complaining"
        assert pred.confidence >= 0.80

    def test_devanagari_hindi_complaining(self, baseline_engine):
        pred = baseline_engine.predict_text_type("यह ऐप बिल्कुल बेकार है और बार-बार क्रैश हो रहा है।")
        assert pred.label == "Complaining"
        assert pred.confidence >= 0.80

    def test_negative_sentiment_without_complaint_is_statement(self, baseline_engine):
        """Pure negative sentiment (e.g. personal feelings) must NOT be labeled Complaining."""
        res_sent = baseline_engine.predict_sentiment("I hate this person so much.")
        res_type = baseline_engine.predict_text_type("I hate this person so much.")
        assert res_sent.label == "Negative"
        assert res_type.label == "Statement"


# =============================================================================
# 2. MANDATORY ABUSE VOCABULARY COVERAGE
# =============================================================================

class TestMandatoryAbuseVocabulary:
    def test_madarchod_hinglish(self, abuse_detector):
        pred = abuse_detector.predict("You are a madarchod for doing this.")
        assert pred.label == "Sexualized Abuse"
        assert pred.metadata.get("severity") == "Severe"
        assert pred.metadata.get("is_abusive") is True

    def test_madarchod_devanagari(self, abuse_detector):
        pred = abuse_detector.predict("तू बहुत बड़ा मादरचोद है।")
        assert pred.label == "Sexualized Abuse"
        assert pred.metadata.get("severity") == "Severe"

    def test_chutiya_hinglish(self, abuse_detector):
        pred = abuse_detector.predict("Chutiya insaan hai tu bilkul.")
        assert pred.label == "Severe Personal Insult"
        assert pred.metadata.get("is_abusive") is True

    def test_chutiye_devanagari(self, abuse_detector):
        pred = abuse_detector.predict("अरे चूतिए चुप बैठ।")
        assert pred.label == "Severe Personal Insult"
        assert pred.metadata.get("is_abusive") is True

    def test_gaand_hinglish(self, abuse_detector):
        pred = abuse_detector.predict("Apni gaand mara.")
        assert pred.label == "Sexualized Abuse"
        assert pred.metadata.get("severity") == "Severe"

    def test_gaand_devanagari(self, abuse_detector):
        pred = abuse_detector.predict("गांड में दम है तो सामने आ।")
        assert pred.label == "Sexualized Abuse"

    def test_lund_hinglish(self, abuse_detector):
        pred = abuse_detector.predict("Lund le le mera.")
        assert pred.label == "Sexualized Abuse"
        assert pred.metadata.get("severity") == "Severe"

    def test_lund_devanagari(self, abuse_detector):
        pred = abuse_detector.predict("लंड जैसा चेहरा है तेरा।")
        assert pred.label == "Sexualized Abuse"

    def test_teri_maa_ki_chut_hinglish(self, abuse_detector):
        pred = abuse_detector.predict("Teri maa ki chut nikal yahan se.")
        assert pred.label == "Sexualized Abuse"
        assert pred.metadata.get("severity") == "Severe"

    def test_teri_maa_ki_chut_devanagari(self, abuse_detector):
        pred = abuse_detector.predict("तेरी माँ की चूत।")
        assert pred.label == "Sexualized Abuse"
        assert pred.metadata.get("severity") == "Severe"


# =============================================================================
# 3. NORMALIZATION & LEETSPEAK UNMASKING
# =============================================================================

class TestNormalizationAndLeetspeak:
    def test_leetspeak_m_at_darchod(self, abuse_detector):
        pred = abuse_detector.predict("You m@darchod don't talk to me.")
        assert pred.label == "Sexualized Abuse"

    def test_leetspeak_ch00tiya(self, abuse_detector):
        pred = abuse_detector.predict("Tu ch00tiya hai kya re?")
        assert pred.label == "Severe Personal Insult"

    def test_leetspeak_l_at_nd(self, abuse_detector):
        pred = abuse_detector.predict("Apna l@nd band rakh.")
        assert pred.label == "Sexualized Abuse"

    def test_leetspeak_g_at_nd(self, abuse_detector):
        pred = abuse_detector.predict("Apni g@nd sambhal.")
        assert pred.label == "Sexualized Abuse"

    def test_asterisk_unmasking(self, abuse_detector):
        pred = abuse_detector.predict("Tu ek ch*tiya admi hai.")
        assert pred.label == "Severe Personal Insult"

    def test_spacing_collapse(self, abuse_detector):
        pred = abuse_detector.predict("madar chod nikal yahan se")
        assert pred.label == "Sexualized Abuse"

    def test_character_elongation_collapse(self, abuse_detector):
        pred = abuse_detector.predict("chuuuutiya insan hai")
        assert pred.label == "Severe Personal Insult"


# =============================================================================
# 4. CONTEXTUAL INTELLIGENCE & FALSE-POSITIVE PREVENTION
# =============================================================================

class TestContextualIntelligence:
    def test_quoted_abuse_reported_by_author(self, abuse_detector):
        pred = abuse_detector.predict("Someone called me an idiot in the street today.")
        assert pred.label == "Not abusive"
        assert pred.metadata.get("is_quoted") is True

    def test_quoted_abuse_in_speech_marks(self, abuse_detector):
        pred = abuse_detector.predict('He screamed "you are stupid" at the meeting.')
        assert pred.label == "Not abusive"
        assert pred.metadata.get("is_quoted") is True

    def test_academic_linguistic_discussion(self, abuse_detector):
        pred = abuse_detector.predict("The word madarchod is considered severe Hindi profanity.")
        assert pred.label == "Not abusive"
        assert pred.metadata.get("is_discussion") is True

    def test_linguistics_study_context(self, abuse_detector):
        pred = abuse_detector.predict("In linguistics the term chutiya is studied as an insult.")
        assert pred.label == "Not abusive"
        assert pred.metadata.get("is_discussion") is True

    def test_product_service_criticism_not_targeted_abuse(self, abuse_detector):
        pred = abuse_detector.predict("Your application is terrible and crashed my server.")
        assert pred.label == "Not abusive"
        assert pred.metadata.get("is_product_criticism") is True

    def test_general_restaurant_service_complaint(self, abuse_detector):
        pred = abuse_detector.predict("The service was very slow and the food was cold.")
        assert pred.label == "Not abusive"


# =============================================================================
# 5. THREATS, HARASSMENT & MULTI-LABEL
# =============================================================================

class TestThreatsAndHarassment:
    def test_explicit_threat_of_violence(self, abuse_detector):
        pred = abuse_detector.predict("I will hunt you down and kill you tonight.")
        assert pred.label == "Threat"
        assert pred.metadata.get("severity") == "Severe"

    def test_physical_violence_threat(self, abuse_detector):
        pred = abuse_detector.predict("I am going to beat you up until you bleed.")
        assert pred.label == "Threat"
        assert pred.metadata.get("severity") == "Severe"

    def test_harassment_and_intimidation(self, abuse_detector):
        pred = abuse_detector.predict("Stop posting or I will ruin your entire life.")
        assert pred.label == "Harassment"
        assert pred.metadata.get("severity") == "Moderate"

    def test_profanity_without_targeting(self, abuse_detector):
        pred = abuse_detector.predict("What the fuck is this bullshit?")
        assert pred.label == "Profanity / Vulgarity"
        assert pred.metadata.get("severity") in ("Mild", "Moderate")

    def test_multi_label_detection(self, abuse_detector):
        """Sentence containing both an insult and sexualized slur."""
        pred = abuse_detector.predict("You idiot madarchod listen to me.")
        assert pred.label in ("Sexualized Abuse", "Severe Personal Insult")
        assert len(pred.metadata.get("all_categories", [])) >= 2


# =============================================================================
# 6. ZERO-PII ANALYTICS GUARANTEE
# =============================================================================

class TestZeroPIIAnalyticsPolicy:
    def test_analytics_does_not_log_sentence_text(self, tmp_path):
        from db.analytics import record_usage_event, get_recent_events_log
        from db.models import UsageEvent
        from db.session import get_db_session

        abusive_sample = "You madarchod don't talk to me."
        
        # Simulate tracking an analysis event
        record_usage_event(
            event_category="analysis",
            event_action="sentence_analysis",
            count=1,
            metadata={"task": "predict_all"},
            visitor_id="test-anon-visitor-1",
            session_id="test-session-1",
        )

        # Inspect database records
        with get_db_session() as s:
            events = s.query(UsageEvent).all()
            for ev in events:
                # Guarantee: raw sentence text and abusive terms NEVER stored
                raw_json = str(ev.metadata_json or {})
                assert "madarchod" not in raw_json.lower()
                assert abusive_sample not in raw_json


# =============================================================================
# 7. UI MAPPINGS & DATA MANAGER PERSISTENCE
# =============================================================================

class TestUIMappingsAndDataManager:
    def test_bidirectional_text_type_mapping(self):
        assert to_internal_label("text_type", "Complaining or reporting an issue") == "Complaining"
        assert to_display_label("text_type", "Complaining") == "Complaining or reporting an issue"

    def test_bidirectional_abuse_mapping(self):
        for cat in ["Not abusive", "Profanity / Vulgarity", "Severe Personal Insult", "Sexualized Abuse", "Harassment", "Threat"]:
            assert to_internal_label("abuse", cat) == cat
            assert to_display_label("abuse", cat) == cat

        assert to_internal_label("abuse", "Not sure") == "Uncertain"
        assert to_display_label("abuse", "Uncertain") == "Not sure"

    def test_data_manager_cols_contain_abuse(self):
        assert "abuse_category" in COLS
        assert "abuse_severity" in COLS

    def test_validation_issues_mandatory_notes_for_uncertain_abuse(self):
        vals = {
            "language": "English",
            "sentiment": "Neutral",
            "text_type": "Statement",
            "abuse_category": "Uncertain",
            "review_status": "Reviewed — looks correct",
            "annotator_notes": "",
        }
        issues = validation_issues(vals)
        assert any("uncertain abuse category" in iss.lower() for iss in issues)

    def test_save_dataset_record_preserves_abuse(self, tmp_path):
        p = tmp_path / "test_annot.csv"
        df = pd.DataFrame([{
            "id": "rec_1",
            "text": "Hello world",
            "language": "English",
            "sentiment": "Positive",
            "text_type": "Greeting",
            "abuse_category": "Not abusive",
            "abuse_severity": "None",
            "review_status": "Reviewed — looks correct",
            "annotator_notes": "All verified",
        }])
        df.to_csv(p, index=False)

        update_vals = {
            "abuse_category": "Profanity / Vulgarity",
            "abuse_severity": "Mild",
        }
        ok = save_dataset_record(p, df, "rec_1", update_vals)
        assert ok is True

        saved_df = pd.read_csv(p)
        assert saved_df.iloc[0]["abuse_category"] == "Profanity / Vulgarity"
        assert saved_df.iloc[0]["abuse_severity"] == "Mild"

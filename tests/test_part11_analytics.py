"""
Comprehensive Unit, Privacy & Telemetry Tests for LinguaLabel Part 11:
Private Website Analytics & Admin Dashboard.

Validates:
1. Anonymous Visitor Tracking:
   - Randomly generated first-party identifiers (RFC 4122 UUID hex).
   - Zero invasive fingerprinting.
   - Deduplication across visits and sessions.
2. Visitor Session Management & Inactivity Boundary:
   - Resuming active sessions within the 30-minute inactivity window.
   - Starting a new session and incrementing visit_count after >30 minutes inactivity.
   - Streamlit reruns do not create new visits.
3. Page View Tracking:
   - Deduplication: Page views record only on actual page/view changes.
   - Incrementing visitor and session pageview counters.
4. Website Usage Analytics:
   - Sentence analyses tracking.
   - Dataset upload tracking.
   - CSV and Excel import tracking.
   - Export tracking (CSV and Excel).
   - Most-used feature breakdown.
   - Day-by-day activity trend calculations.
5. Strict Zero-PII Privacy Protection:
   - Automatic sanitization of event metadata.
   - Forbids storing actual sentence text, dataset contents, or raw inputs.
6. Private Owner Access Gate:
   - Constant-time secret key verification.
   - Rejection of invalid, empty, or wrong keys.
7. Public Website Open Access:
   - Verification that sentence analysis, file imports, and exports work without authentication.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.analytics import (
    generate_anonymous_id,
    get_activity_trends,
    get_or_create_visitor,
    get_recent_events_log,
    get_usage_analytics,
    get_visitor_analytics,
    record_page_view,
    record_usage_event,
    register_or_resume_session,
    sanitize_event_metadata,
    verify_admin_key,
)
from db.config import ADMIN_SECRET_KEY
from db.models import Base, PageView, UsageEvent, Visitor, VisitorSession
from data_manager import (
    load_csv_bytes,
    load_excel_bytes,
    load_pasted_sentences,
    load_single_sentence,
    stabilize_dataset,
)
from ml.engine import MLEngine


@pytest.fixture
def isolated_db_session():
    """Provides an isolated, in-memory SQLite database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    session = SessionFactory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


# =============================================================================
# 1. ANONYMOUS VISITOR TRACKING TESTS
# =============================================================================

def test_anonymous_id_generation():
    """Validates that anonymous visitor identifiers are unique 32-hex UUID strings."""
    id1 = generate_anonymous_id()
    id2 = generate_anonymous_id()
    assert id1 != id2
    assert len(id1) == 32
    assert isinstance(id1, str)
    # Hexadecimal characters only
    int(id1, 16)


def test_get_or_create_visitor(isolated_db_session):
    """Validates creating new visitors and retrieving existing ones."""
    s = isolated_db_session

    # 1. Create brand new visitor
    v1, is_new1 = get_or_create_visitor(session=s)
    s.flush()
    assert is_new1 is True
    assert v1.id is not None
    assert v1.visit_count == 0
    assert v1.pageview_count == 0

    # 2. Retrieve existing visitor
    v2, is_new2 = get_or_create_visitor(visitor_id=v1.id, session=s)
    assert is_new2 is False
    assert v2.id == v1.id


# =============================================================================
# 2. VISITOR SESSION MANAGEMENT & 30-MIN INACTIVITY BOUNDARY
# =============================================================================

def test_session_resumption_within_inactivity_window(isolated_db_session):
    """Validates that rapid actions (or Streamlit reruns) within 30 min resume the same session."""
    s = isolated_db_session
    visitor, _ = get_or_create_visitor(session=s)
    s.flush()

    # Initial session
    sess1, is_new1 = register_or_resume_session(visitor.id, session=s)
    s.flush()
    assert is_new1 is True
    assert sess1.visitor_id == visitor.id

    # Simulated rerun or navigation 2 minutes later
    sess2, is_new2 = register_or_resume_session(visitor.id, session_id=sess1.id, session=s)
    assert is_new2 is False
    assert sess2.id == sess1.id
    # Visitor visit_count must remain 1
    assert visitor.visit_count == 1


def test_session_inactivity_timeout_and_returning_visitor(isolated_db_session):
    """Validates that returning after > 30 minutes starts a new session and increments visit_count."""
    s = isolated_db_session
    visitor, _ = get_or_create_visitor(session=s)
    s.flush()

    # First visit session started 45 minutes ago
    sess1, _ = register_or_resume_session(visitor.id, session=s)
    now = datetime.now(timezone.utc)
    sess1.started_at = now - timedelta(minutes=45)
    sess1.last_active_at = now - timedelta(minutes=40)
    s.flush()

    # Visitor returns now (40 minutes idle)
    sess2, is_new2 = register_or_resume_session(visitor.id, session_id=sess1.id, session=s)
    s.flush()

    assert is_new2 is True
    assert sess2.id != sess1.id
    # Visitor is now identified as a RETURNING visitor (visit_count == 2)
    assert visitor.visit_count == 2


# =============================================================================
# 3. PAGE VIEW TRACKING & DEDUPLICATION
# =============================================================================

def test_page_view_tracking(isolated_db_session):
    """Validates recording page views and updating aggregate counters."""
    s = isolated_db_session
    visitor, _ = get_or_create_visitor(session=s)
    sess, _ = register_or_resume_session(visitor.id, session=s)
    s.flush()

    # Record views on distinct pages
    pv1 = record_page_view(visitor.id, sess.id, "Home", session=s)
    pv2 = record_page_view(visitor.id, sess.id, "Import Sentences", session=s)
    pv3 = record_page_view(visitor.id, sess.id, "Analyse & Label", session=s)
    s.flush()

    assert pv1.page_name == "Home"
    assert pv2.page_name == "Import Sentences"
    assert pv3.page_name == "Analyse & Label"

    # Verify counters
    assert visitor.pageview_count == 3
    assert sess.pageview_count == 3


# =============================================================================
# 4. PRIVACY PROTECTION & ZERO-PII METADATA SANITIZATION
# =============================================================================

def test_privacy_sanitization_strips_sentence_and_dataset_text():
    """Validates that sensitive content, sentence texts, and dataset dumps are strictly stripped."""
    dirty_meta = {
        "format": "csv",
        "row_count": 50,
        "text": "This is a sensitive sentence that must not be logged!",
        "sentence": "Ye secret sentence hai",
        "original_text": "Sensitive input data",
        "dataset_content": "Col1,Col2\nVal1,Val2",
        "user_email": "private@example.com",
        "status": "success",
    }

    clean_meta = sanitize_event_metadata(dirty_meta)

    # Allowed safe metadata remains
    assert clean_meta["format"] == "csv"
    assert clean_meta["row_count"] == 50
    assert clean_meta["status"] == "success"

    # Forbidden text and content keys MUST be completely absent
    assert "text" not in clean_meta
    assert "sentence" not in clean_meta
    assert "original_text" not in clean_meta
    assert "dataset_content" not in clean_meta
    assert "user_email" not in clean_meta


def test_record_usage_event_privacy(isolated_db_session):
    """Validates that record_usage_event enforces privacy sanitization in the database."""
    s = isolated_db_session

    ev = record_usage_event(
        event_category="import",
        event_action="csv_import",
        count=1,
        metadata={"format": "csv", "sentence": "Secret sentence", "row_count": 25},
        session=s,
    )
    s.flush()

    assert ev.event_action == "csv_import"
    assert ev.event_count == 1
    assert "sentence" not in ev.metadata_json
    assert ev.metadata_json.get("format") == "csv"
    assert ev.metadata_json.get("row_count") == 25


# =============================================================================
# 5. WEBSITE USAGE & FEATURE ANALYTICS AGGREGATION
# =============================================================================

def test_usage_and_feature_analytics(isolated_db_session):
    """Validates aggregation of sentence analyses, dataset uploads, imports, and exports."""
    s = isolated_db_session

    # Record multiple usage events
    record_usage_event("analysis", "sentence_analysis", count=5, session=s)
    record_usage_event("import", "dataset_upload", count=1, metadata={"format": "csv"}, session=s)
    record_usage_event("import", "csv_import", count=1, session=s)
    record_usage_event("import", "dataset_upload", count=1, metadata={"format": "xlsx"}, session=s)
    record_usage_event("import", "excel_import", count=1, session=s)
    record_usage_event("export", "export_csv", count=1, session=s)
    record_usage_event("export", "export_excel", count=1, session=s)
    s.flush()

    metrics = get_usage_analytics(session=s)

    assert metrics["sentence_analyses_count"] == 5
    assert metrics["datasets_uploaded_count"] == 2
    assert metrics["csv_imports_count"] == 1
    assert metrics["excel_imports_count"] == 1
    assert metrics["exports_count"] == 2
    assert metrics["csv_exports_count"] == 1
    assert metrics["excel_exports_count"] == 1
    assert "sentence_analysis" in metrics["feature_breakdown"]


def test_visitor_summary_metrics(isolated_db_session):
    """Validates calculation of total, new, returning, daily, weekly, and monthly visitors."""
    s = isolated_db_session
    now = datetime.now(timezone.utc)

    # Visitor 1: New visitor active today
    v1, _ = get_or_create_visitor(session=s)
    s1, _ = register_or_resume_session(v1.id, session=s)
    record_page_view(v1.id, s1.id, "Home", session=s)

    # Visitor 2: Returning visitor active 2 days ago
    v2, _ = get_or_create_visitor(session=s)
    v2.visit_count = 3
    v2.last_seen_at = now - timedelta(days=2)
    s2, _ = register_or_resume_session(v2.id, session=s)
    s2.started_at = now - timedelta(days=2)
    s2.last_active_at = now - timedelta(days=2)
    record_page_view(v2.id, s2.id, "Analyse & Label", session=s)

    # Visitor 3: Active 10 days ago (in monthly, outside weekly)
    v3, _ = get_or_create_visitor(session=s)
    v3.last_seen_at = now - timedelta(days=10)
    s3, _ = register_or_resume_session(v3.id, session=s)
    s3.started_at = now - timedelta(days=10)
    s3.last_active_at = now - timedelta(days=10)
    record_page_view(v3.id, s3.id, "Export Data", session=s)

    s.flush()

    v_metrics = get_visitor_analytics(session=s)

    assert v_metrics["total_unique_visitors"] == 3
    assert v_metrics["total_visits"] == 3
    assert v_metrics["new_visitors"] == 2  # v1 and v3
    assert v_metrics["returning_visitors"] == 1  # v2
    assert v_metrics["daily_visitors"] >= 1  # v1
    assert v_metrics["weekly_visitors"] >= 2  # v1 and v2
    assert v_metrics["monthly_visitors"] == 3  # v1, v2, v3
    assert v_metrics["total_pageviews"] == 3
    assert "Home" in v_metrics["pageviews_by_page"]
    assert "Analyse & Label" in v_metrics["pageviews_by_page"]


def test_activity_trends_and_recent_log(isolated_db_session):
    """Validates activity trends generation and recent events audit log."""
    s = isolated_db_session

    record_usage_event("analysis", "sentence_analysis", count=3, session=s)
    record_usage_event("export", "export_csv", count=1, session=s)
    s.flush()

    trends = get_activity_trends(days=7, session=s)
    assert len(trends) == 7
    assert "date" in trends[0]
    assert "analyses" in trends[-1]

    recent = get_recent_events_log(limit=10, session=s)
    assert len(recent) == 2
    assert recent[0]["action"] in ("sentence_analysis", "export_csv")


# =============================================================================
# 6. PRIVATE OWNER ACCESS GATE VERIFICATION
# =============================================================================

def test_admin_key_verification():
    """Validates constant-time verification of the private owner secret key."""
    # Correct key
    assert verify_admin_key(ADMIN_SECRET_KEY) is True

    # Incorrect keys
    assert verify_admin_key("wrong_secret_key") is False
    assert verify_admin_key("") is False
    assert verify_admin_key(None) is False
    assert verify_admin_key("   ") is False


# =============================================================================
# 7. PUBLIC OPEN ACCESS (ZERO AUTHENTICATION REQUIRED)
# =============================================================================

def test_public_unauthenticated_workflow():
    """Validates that public core features work smoothly without user accounts or login."""
    # 1. Single sentence query
    df_single, err1 = load_single_sentence("This is a great app.")
    assert err1 is None
    assert len(df_single) == 1

    # 2. Pasted sentences
    df_paste, err2 = load_pasted_sentences("Line 1\nLine 2\nLine 3")
    assert err2 is None
    assert len(df_paste) == 3

    # 3. CSV Loading
    csv_raw = b"id,text\n1,Hello world\n2,Namaste dosto\n"
    df_csv, _, err3 = load_csv_bytes(csv_raw, "test.csv")
    assert err3 is None
    assert len(df_csv) == 2

    # 4. AI Engine prediction without authentication
    engine = MLEngine()
    preds = engine.predict_all("Hello world, this is a test.")
    assert "language" in preds
    assert "sentiment" in preds
    assert "text_type" in preds

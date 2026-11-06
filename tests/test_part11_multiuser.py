"""
Comprehensive Unit, Security & Isolation Tests for LinguaLabel Part 11:
User Accounts, Personal Workspaces & Multi-User Data Isolation.

Validates:
1. Authentication & Security:
   - Argon2id password hashing and constant-time verification.
   - User registration with input validation (email, username, password complexity).
   - Duplicate email and username prevention.
   - Authentication with generic failure messages.
   - Account deactivation handling.
   - HMAC-SHA256 session token generation, signature validation, and expiration.
2. Multi-User Data Isolation (Strict Zero-Trust Boundary):
   - Cross-user workspace isolation (User B cannot access or list User A's workspaces).
   - Cross-user dataset isolation (User B cannot view, modify, or delete User A's datasets).
   - Cross-user sentence annotation isolation (User B cannot alter User A's annotations).
   - Cross-user feedback isolation (User B cannot inspect User A's model feedback).
   - Cross-user export blocking (User B cannot export User A's dataset).
3. Workspace & Dataset Lifecycle:
   - Creating, renaming, duplicating, and safely deleting workspaces and datasets.
   - Preservation of sentences and sequential indexing.
4. Legacy Storage Migration:
   - Previewing and migrating legacy files to a specific user and workspace.
   - Repeat-safe (idempotent) execution and non-destructive backups.
"""

import time
from pathlib import Path
import tempfile
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base, Dataset, Feedback, Sentence, User, Workspace
from db.auth import (
    authenticate_user,
    create_session_token,
    hash_password,
    register_user,
    validate_registration_inputs,
    verify_password,
    verify_session_token,
)
from db.service import (
    create_dataset_with_sentences,
    create_workspace,
    delete_dataset,
    delete_workspace,
    duplicate_dataset,
    export_user_dataset_df,
    get_dataset,
    get_dataset_sentences,
    get_user_dashboard_stats,
    get_workspace,
    list_datasets,
    list_user_feedback,
    list_workspaces,
    record_feedback,
    rename_dataset,
    rename_workspace,
    update_sentence_annotation,
)
from db.migration_tool import migrate_legacy_data_to_user, preview_legacy_data


@pytest.fixture
def isolated_db_session():
    """Provides an isolated, in-memory SQLite database session with enforced foreign keys."""
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
# 1. AUTHENTICATION & PASSWORD HASHING TESTS
# =============================================================================

def test_argon2_password_hashing():
    """Validates Argon2id hashing and verification."""
    raw_pass = "SecurePass123!"
    p_hash = hash_password(raw_pass)

    assert p_hash.startswith("$argon2id$")
    assert verify_password(raw_pass, p_hash) is True
    assert verify_password("WrongPassword123!", p_hash) is False
    assert verify_password("", p_hash) is False


def test_registration_validation():
    """Validates input validation for registration."""
    # Invalid email
    ok, err = validate_registration_inputs("invalid-email", "validuser", "ValidPass1")
    assert ok is False
    assert "valid email" in err.lower()

    # Short username
    ok, err = validate_registration_inputs("user@example.com", "ab", "ValidPass1")
    assert ok is False
    assert "3-30 characters" in err.lower()

    # Weak password
    ok, err = validate_registration_inputs("user@example.com", "validuser", "short1")
    assert ok is False
    assert "at least 8 characters" in err.lower()

    # Password without numbers
    ok, err = validate_registration_inputs("user@example.com", "validuser", "NoNumbersHere")
    assert ok is False
    assert "at least one letter and one number" in err.lower()

    # Valid inputs
    ok, err = validate_registration_inputs("user@example.com", "valid_user-1", "ValidPass123")
    assert ok is True
    assert err is None


def test_user_registration_and_duplicate_prevention(isolated_db_session):
    """Validates user creation and prevention of duplicate emails/usernames."""
    s = isolated_db_session

    user, err = register_user("alice@domain.com", "alice", "AlicePassword1", session=s)
    assert err is None
    assert user is not None
    assert user.email == "alice@domain.com"
    assert user.username == "alice"
    assert user.is_active is True

    # User should have an automatic default workspace
    workspaces = list_workspaces(user.id, session=s)
    assert len(workspaces) == 1
    assert workspaces[0].name == "Personal Workspace"

    # Duplicate email check (case-insensitive)
    dup_user, dup_err = register_user("ALICE@domain.com", "alice_other", "Pass12345", session=s)
    assert dup_user is None
    assert "already exists" in dup_err.lower()

    # Duplicate username check (case-insensitive)
    dup_user2, dup_err2 = register_user("other@domain.com", "ALICE", "Pass12345", session=s)
    assert dup_user2 is None
    assert "already taken" in dup_err2.lower()


def test_authentication_flow(isolated_db_session):
    """Validates login with username or email, generic errors, and account deactivation."""
    s = isolated_db_session
    user, _ = register_user("bob@domain.com", "bob_dev", "BobPassword123", session=s)

    # Login by email
    auth_user, err = authenticate_user("bob@domain.com", "BobPassword123", session=s)
    assert err is None
    assert auth_user.id == user.id

    # Login by username
    auth_user2, err2 = authenticate_user("bob_dev", "BobPassword123", session=s)
    assert err2 is None
    assert auth_user2.id == user.id

    # Incorrect password -> Generic error
    bad_user, bad_err = authenticate_user("bob_dev", "WrongPass123", session=s)
    assert bad_user is None
    assert bad_err == "Invalid email or password."

    # Non-existent user -> Generic error
    ghost_user, ghost_err = authenticate_user("ghost@domain.com", "AnyPass123", session=s)
    assert ghost_user is None
    assert ghost_err == "Invalid email or password."

    # Deactivated account
    user.is_active = False
    s.flush()
    deact_user, deact_err = authenticate_user("bob@domain.com", "BobPassword123", session=s)
    assert deact_user is None
    assert "deactivated" in deact_err.lower()


def test_hmac_session_tokens():
    """Validates HMAC session token creation, verification, and expiration."""
    user_id = "test-user-uuid-12345"
    token = create_session_token(user_id)
    assert token is not None

    # Verify valid token
    extracted_id = verify_session_token(token)
    assert extracted_id == user_id

    # Tampered token
    tampered = token[:-4] + "abcd"
    assert verify_session_token(tampered) is None

    # Malformed token
    assert verify_session_token("not.enough.parts") is None
    assert verify_session_token("") is None


# =============================================================================
# 2. MULTI-USER DATA ISOLATION (ZERO-TRUST TESTS)
# =============================================================================

def test_cross_user_isolation(isolated_db_session):
    """
    Rigorously validates that User B cannot access, read, alter, or export
    any workspace, dataset, sentence, or feedback belonging to User A.
    """
    s = isolated_db_session

    user_a, _ = register_user("alice@isolate.com", "alice_iso", "AliceSecret1", session=s)
    user_b, _ = register_user("bob@isolate.com", "bob_iso", "BobSecret1", session=s)

    # User A creates a workspace and dataset
    ws_a, _ = create_workspace(user_a.id, "Alice NLP Projects", "Confidential", session=s)
    records = [
        {"id": "sent_1", "text": "Confidential customer statement.", "language": "English", "sentiment": "Positive", "text_type": "Statement", "review_status": "Reviewed — looks correct"},
        {"id": "sent_2", "text": "Another private item.", "language": "English", "sentiment": "Neutral", "text_type": "Statement", "review_status": "Pending review"},
    ]
    ds_a, _ = create_dataset_with_sentences(user_a.id, ws_a.id, "Alice Dataset", records, session=s)

    # User A records feedback
    fb_a, _ = record_feedback(user_a.id, ds_a.id, "sent_1", "sentiment", {"prob": 0.8}, "Positive", session=s)

    # -------------------------------------------------------------
    # 1. Workspace Isolation
    # -------------------------------------------------------------
    # User B cannot see Alice's workspace in list
    b_workspaces = list_workspaces(user_b.id, session=s)
    assert ws_a.id not in [w.id for w in b_workspaces]

    # User B cannot get Alice's workspace by ID
    assert get_workspace(user_b.id, ws_a.id, session=s) is None

    # User B cannot rename Alice's workspace
    ok, msg = rename_workspace(user_b.id, ws_a.id, "Hacked Workspace", session=s)
    assert ok is False
    assert "unauthorized" in msg.lower() or "not found" in msg.lower()

    # User B cannot delete Alice's workspace
    ok_del, msg_del = delete_workspace(user_b.id, ws_a.id, session=s)
    assert ok_del is False

    # -------------------------------------------------------------
    # 2. Dataset Isolation
    # -------------------------------------------------------------
    # User B cannot list Alice's dataset
    b_datasets = list_datasets(user_b.id, session=s)
    assert ds_a.id not in [d.id for d in b_datasets]

    # User B cannot fetch Alice's dataset
    assert get_dataset(user_b.id, ds_a.id, session=s) is None

    # User B cannot rename Alice's dataset
    ok_ren, msg_ren = rename_dataset(user_b.id, ds_a.id, "Bob Stolen Dataset", session=s)
    assert ok_ren is False

    # User B cannot duplicate Alice's dataset
    dup_res, dup_err = duplicate_dataset(user_b.id, ds_a.id, session=s)
    assert dup_res is None

    # User B cannot delete Alice's dataset
    del_res, _ = delete_dataset(user_b.id, ds_a.id, session=s)
    assert del_res is False

    # -------------------------------------------------------------
    # 3. Sentence Annotation Isolation
    # -------------------------------------------------------------
    # User B cannot view Alice's sentences
    sentences_b, err_b = get_dataset_sentences(user_b.id, ds_a.id, session=s)
    assert sentences_b is None
    assert "unauthorized" in err_b.lower() or "not found" in err_b.lower()

    # User B cannot modify Alice's sentence annotation
    hacked_vals = {
        "language": "Hindi",
        "sentiment": "Negative",
        "text_type": "Asking a question",
        "review_status": "Reviewed — looks correct",
        "annotator_notes": "Maliciously modified by Bob",
    }
    upd_ok, upd_msg = update_sentence_annotation(user_b.id, ds_a.id, "sent_1", hacked_vals, session=s)
    assert upd_ok is False
    assert "unauthorized" in upd_msg.lower() or "not found" in upd_msg.lower()

    # Verify Alice's sentence was NOT changed
    sentences_a, _ = get_dataset_sentences(user_a.id, ds_a.id, session=s)
    assert sentences_a[0].language == "English"
    assert sentences_a[0].sentiment == "Positive"

    # -------------------------------------------------------------
    # 4. Feedback Isolation
    # -------------------------------------------------------------
    b_feedbacks = list_user_feedback(user_b.id, session=s)
    assert len(b_feedbacks) == 0

    # User B cannot record feedback on Alice's dataset
    bad_fb, fb_err = record_feedback(user_b.id, ds_a.id, "sent_1", "sentiment", {}, "Negative", session=s)
    assert bad_fb is None
    assert "unauthorized" in fb_err.lower() or "not found" in fb_err.lower()

    # -------------------------------------------------------------
    # 5. Export Isolation
    # -------------------------------------------------------------
    export_df, exp_err = export_user_dataset_df(user_b.id, ds_a.id, session=s)
    assert export_df is None
    assert "unauthorized" in exp_err.lower() or "not found" in exp_err.lower()


def test_user_dashboard_stats_isolation(isolated_db_session):
    """Verifies that personal dashboard metrics strictly reflect only the user's data."""
    s = isolated_db_session

    user_1, _ = register_user("user1@stats.com", "user1", "Pass123456", session=s)
    user_2, _ = register_user("user2@stats.com", "user2", "Pass123456", session=s)

    ws_1 = list_workspaces(user_1.id, session=s)[0]
    records_1 = [
        {"id": "u1_1", "text": "Item 1", "language": "English", "sentiment": "Positive", "text_type": "Statement", "review_status": "Reviewed — looks correct"},
        {"id": "u1_2", "text": "Item 2", "language": "English", "sentiment": "Neutral", "text_type": "Statement", "review_status": "Pending review"},
    ]
    create_dataset_with_sentences(user_1.id, ws_1.id, "User 1 Dataset", records_1, session=s)

    # Check stats for User 1
    stats_1 = get_user_dashboard_stats(user_1.id, session=s)
    assert stats_1["total_workspaces"] == 1
    assert stats_1["total_datasets"] == 1
    assert stats_1["total_sentences"] == 2
    assert stats_1["reviewed_sentences"] == 1
    assert stats_1["pending_reviews"] == 1

    # Check stats for User 2 (must be 0 datasets and 0 sentences)
    stats_2 = get_user_dashboard_stats(user_2.id, session=s)
    assert stats_2["total_workspaces"] == 1  # Only their default personal workspace
    assert stats_2["total_datasets"] == 0
    assert stats_2["total_sentences"] == 0
    assert stats_2["reviewed_sentences"] == 0
    assert stats_2["pending_reviews"] == 0


# =============================================================================
# 3. WORKSPACE & DATASET LIFECYCLE TESTS
# =============================================================================

def test_workspace_lifecycle(isolated_db_session):
    """Tests creating, renaming, and deleting workspaces safely."""
    s = isolated_db_session
    user, _ = register_user("ws_user@test.com", "ws_user", "ValidPassword1", session=s)

    # Initially has 1 default workspace
    ws_list = list_workspaces(user.id, session=s)
    assert len(ws_list) == 1
    default_ws = ws_list[0]

    # Cannot delete only workspace
    can_del, msg = delete_workspace(user.id, default_ws.id, session=s)
    assert can_del is False
    assert "only workspace" in msg.lower()

    # Create second workspace
    ws_2, err = create_workspace(user.id, "Second Workspace", "Description", session=s)
    assert err is None
    assert ws_2 is not None

    # Rename second workspace
    ok_ren, _ = rename_workspace(user.id, ws_2.id, "Renamed Second Workspace", "New Desc", session=s)
    assert ok_ren is True
    updated_ws = get_workspace(user.id, ws_2.id, session=s)
    assert updated_ws.name == "Renamed Second Workspace"

    # Now can delete second workspace
    can_del2, _ = delete_workspace(user.id, ws_2.id, session=s)
    assert can_del2 is True
    assert len(list_workspaces(user.id, session=s)) == 1


def test_dataset_duplication_and_export(isolated_db_session):
    """Tests dataset creation, duplication, and authorized export."""
    s = isolated_db_session
    user, _ = register_user("dup_user@test.com", "dup_user", "ValidPassword1", session=s)
    ws = list_workspaces(user.id, session=s)[0]

    records = [
        {"id": "item_1", "text": "Sentence 1", "language": "English", "sentiment": "Positive", "text_type": "Statement", "review_status": "Reviewed — looks correct", "annotator_notes": "Note 1"},
        {"id": "item_2", "text": "Sentence 2", "language": "Hindi", "sentiment": "Neutral", "text_type": "Statement", "review_status": "Pending review"},
    ]
    ds, err = create_dataset_with_sentences(user.id, ws.id, "Original Dataset", records, session=s)
    assert err is None
    assert ds.total_sentences == 2

    # Duplicate dataset
    dup_ds, dup_err = duplicate_dataset(user.id, ds.id, new_name="Cloned Dataset", session=s)
    assert dup_err is None
    assert dup_ds.name == "Cloned Dataset"
    assert dup_ds.total_sentences == 2

    dup_sentences, _ = get_dataset_sentences(user.id, dup_ds.id, session=s)
    assert len(dup_sentences) == 2
    assert dup_sentences[0].original_text == "Sentence 1"
    assert dup_sentences[0].annotator_notes == "Note 1"

    # Authorized Export
    df, exp_err = export_user_dataset_df(user.id, ds.id, session=s)
    assert exp_err is None
    assert df is not None
    assert len(df) == 2
    assert "text" in df.columns
    assert df.iloc[0]["annotator_notes"] == "Note 1"


# =============================================================================
# 4. LEGACY STORAGE MIGRATION TESTS
# =============================================================================

def test_legacy_data_migration(isolated_db_session):
    """
    Validates preview and migration of legacy files into a user's workspace
    without deleting local files.
    """
    s = isolated_db_session
    user, _ = register_user("migrator@test.com", "migrator", "ValidPassword1", session=s)
    ws = list_workspaces(user.id, session=s)[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        sessions_dir = root / "saved_sessions"
        feedback_dir = root / "active_learning_feedback"
        sessions_dir.mkdir(parents=True)
        feedback_dir.mkdir(parents=True)

        # Create dummy session CSV
        csv_path = sessions_dir / "annotated_legacy_batch1.csv"
        csv_df = pd.DataFrame([
            {"id": "leg_1", "text": "First legacy sentence.", "language": "English", "sentiment": "Positive", "text_type": "Statement", "review_status": "Reviewed — looks correct", "annotator_notes": "Legacy note"},
            {"id": "leg_2", "text": "Second legacy sentence.", "language": "Hindi", "sentiment": "Neutral", "text_type": "Statement", "review_status": "Pending review", "annotator_notes": ""},
        ])
        csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # Create dummy feedback JSONL
        fb_path = feedback_dir / "feedback_legacy.jsonl"
        with open(fb_path, "w", encoding="utf-8") as fp:
            fp.write('{"sentence_id": "leg_1", "task": "sentiment", "corrected_label": "Positive", "annotator_notes": "Human verified"}\n')

        # Preview check
        preview = preview_legacy_data(root)
        assert preview["session_files_count"] == 1
        assert preview["total_session_sentences"] == 2
        assert preview["total_feedback_records"] == 1

        # Run migration
        result = migrate_legacy_data_to_user(user.id, ws.id, root, session=s)
        assert result["status"] == "success"
        assert result["datasets_imported"] == 1
        assert result["sentences_imported"] == 2
        assert result["feedback_imported"] == 1

        # Original files must NOT be deleted
        assert csv_path.exists()
        assert fb_path.exists()

        # Backup folder must exist
        backup_dir = Path(result["backup_dir"])
        assert backup_dir.exists()
        assert (backup_dir / csv_path.name).exists()

        # Database verification
        user_datasets = list_datasets(user.id, session=s)
        assert len(user_datasets) == 1
        assert "Legacy Batch1" in user_datasets[0].name

        imported_sentences, _ = get_dataset_sentences(user.id, user_datasets[0].id, session=s)
        assert len(imported_sentences) == 2
        assert imported_sentences[0].id == "leg_1"
        assert imported_sentences[0].annotator_notes == "Legacy note"

        # Repeat-safe check (re-running migration skips already imported datasets)
        res_repeat = migrate_legacy_data_to_user(user.id, ws.id, root, session=s)
        assert res_repeat["datasets_imported"] == 0
        assert res_repeat["datasets_skipped"] == 1


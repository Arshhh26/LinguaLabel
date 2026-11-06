"""
LinguaLabel Database & Multi-User Architecture Package (Part 11)
"""

from db.config import APPLICATION_ENV, AUTH_SECRET_KEY, DATABASE_URL, SESSION_EXPIRATION_MINUTES, ADMIN_SECRET_KEY
from db.models import Base, Dataset, Feedback, Sentence, User, Workspace, Visitor, VisitorSession, PageView, UsageEvent
from db.session import get_db_session, get_engine, get_session_factory, init_db
from db.analytics import (
    generate_anonymous_id,
    verify_admin_key,
    get_or_create_visitor,
    register_or_resume_session,
    record_page_view,
    record_usage_event,
    get_visitor_analytics,
    get_usage_analytics,
    get_activity_trends,
    get_recent_events_log,
    sanitize_event_metadata,
)

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

__all__ = [
    "APPLICATION_ENV",
    "AUTH_SECRET_KEY",
    "DATABASE_URL",
    "SESSION_EXPIRATION_MINUTES",
    "Base",
    "User",
    "Workspace",
    "Dataset",
    "Sentence",
    "Feedback",
    "get_db_session",
    "get_engine",
    "get_session_factory",
    "init_db",
    "hash_password",
    "verify_password",
    "validate_registration_inputs",
    "create_session_token",
    "verify_session_token",
    "register_user",
    "authenticate_user",
    "list_workspaces",
    "get_workspace",
    "create_workspace",
    "rename_workspace",
    "delete_workspace",
    "list_datasets",
    "get_dataset",
    "create_dataset_with_sentences",
    "rename_dataset",
    "duplicate_dataset",
    "delete_dataset",
    "get_dataset_sentences",
    "update_sentence_annotation",
    "record_feedback",
    "list_user_feedback",
    "get_user_dashboard_stats",
    "export_user_dataset_df",
    "preview_legacy_data",
    "migrate_legacy_data_to_user",
]


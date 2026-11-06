"""
LinguaLabel — Multi-User Service Layer & Authorization Enforcement (Part 11)

Strictly enforces multi-user data isolation at the service/storage layer.
Every query, insert, update, delete, and export operation requires an authenticated
user_id and verifies ownership before executing.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from data_manager import validation_issues
from db.models import Dataset, Feedback, Sentence, User, Workspace
from db.session import get_db_session


class AuthorizationError(Exception):
    """Raised when an operation attempts to access or mutate another user's data."""
    pass


# =============================================================================
# WORKSPACE MANAGEMENT SERVICE
# =============================================================================

def list_workspaces(user_id: str, session: Optional[Session] = None) -> List[Workspace]:
    """Returns all workspaces owned by the authenticated user."""
    def _run(s: Session):
        return s.query(Workspace).filter(Workspace.owner_id == user_id).order_by(Workspace.created_at.asc()).all()

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def get_workspace(user_id: str, workspace_id: str, session: Optional[Session] = None) -> Optional[Workspace]:
    """Retrieves a workspace only if owned by the user."""
    def _run(s: Session):
        return s.query(Workspace).filter(
            Workspace.id == workspace_id,
            Workspace.owner_id == user_id,
        ).first()

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def create_workspace(
    user_id: str,
    name: str,
    description: Optional[str] = "",
    session: Optional[Session] = None,
) -> Tuple[Optional[Workspace], Optional[str]]:
    """Creates a new workspace for the authenticated user."""
    clean_name = (name or "").strip()
    if not clean_name:
        return None, "Workspace name cannot be empty."

    def _run(s: Session):
        # Check duplicate name for same user
        existing = s.query(Workspace).filter(
            Workspace.owner_id == user_id,
            func.lower(Workspace.name) == clean_name.lower(),
        ).first()
        if existing:
            return None, f"You already have a workspace named '{clean_name}'."

        ws = Workspace(
            owner_id=user_id,
            name=clean_name,
            description=description or "",
        )
        s.add(ws)
        s.flush()
        return ws, None

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def rename_workspace(
    user_id: str,
    workspace_id: str,
    new_name: str,
    new_description: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tuple[bool, str]:
    """Renames an existing workspace owned by the user."""
    clean_name = (new_name or "").strip()
    if not clean_name:
        return False, "Workspace name cannot be empty."

    def _run(s: Session):
        ws = s.query(Workspace).filter(
            Workspace.id == workspace_id,
            Workspace.owner_id == user_id,
        ).first()
        if not ws:
            return False, "Workspace not found or unauthorized access."

        # Check duplicate name
        existing = s.query(Workspace).filter(
            Workspace.owner_id == user_id,
            Workspace.id != workspace_id,
            func.lower(Workspace.name) == clean_name.lower(),
        ).first()
        if existing:
            return False, f"Another workspace is already named '{clean_name}'."

        ws.name = clean_name
        if new_description is not None:
            ws.description = new_description
        ws.updated_at = datetime.now(timezone.utc)
        return True, "Workspace updated successfully."

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def delete_workspace(
    user_id: str,
    workspace_id: str,
    session: Optional[Session] = None,
) -> Tuple[bool, str]:
    """
    Deletes a workspace and cascades its datasets/sentences.
    Ensures user always maintains at least one active workspace.
    """
    def _run(s: Session):
        ws_count = s.query(Workspace).filter(Workspace.owner_id == user_id).count()
        if ws_count <= 1:
            return False, "Cannot delete your only workspace. Create another workspace first."

        ws = s.query(Workspace).filter(
            Workspace.id == workspace_id,
            Workspace.owner_id == user_id,
        ).first()
        if not ws:
            return False, "Workspace not found or unauthorized access."

        s.delete(ws)
        return True, f"Workspace '{ws.name}' and its contents deleted successfully."

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


# =============================================================================
# DATASET MANAGEMENT SERVICE
# =============================================================================

def list_datasets(
    user_id: str,
    workspace_id: Optional[str] = None,
    session: Optional[Session] = None,
) -> List[Dataset]:
    """Lists datasets owned by user, optionally filtered by workspace."""
    def _run(s: Session):
        query = s.query(Dataset).filter(Dataset.owner_id == user_id)
        if workspace_id:
            query = query.filter(Dataset.workspace_id == workspace_id)
        return query.order_by(Dataset.created_at.desc()).all()

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def get_dataset(
    user_id: str,
    dataset_id: str,
    session: Optional[Session] = None,
) -> Optional[Dataset]:
    """Retrieves dataset strictly verifying ownership."""
    def _run(s: Session):
        return s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def create_dataset_with_sentences(
    user_id: str,
    workspace_id: str,
    name: str,
    records: List[Dict[str, Any]],
    original_filename: Optional[str] = None,
    dataset_key: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tuple[Optional[Dataset], Optional[str]]:
    """
    Creates a dataset and its sentences inside the user's workspace.
    Verifies workspace ownership.
    """
    clean_name = (name or "").strip()
    if not clean_name:
        return None, "Dataset name cannot be empty."

    if not records:
        return None, "Dataset contains no records."

    def _run(s: Session):
        ws = s.query(Workspace).filter(
            Workspace.id == workspace_id,
            Workspace.owner_id == user_id,
        ).first()
        if not ws:
            return None, "Target workspace not found or unauthorized access."

        ds = Dataset(
            workspace_id=workspace_id,
            owner_id=user_id,
            name=clean_name,
            original_filename=original_filename,
            dataset_key=dataset_key,
            total_sentences=len(records),
            processing_status="ready",
        )
        s.add(ds)
        s.flush()

        for idx, rec in enumerate(records):
            sid = str(rec.get("id", "")).strip() or f"{ds.id}_{idx+1}"
            raw_text = str(rec.get("text", "")).strip()
            if not raw_text:
                continue

            sent = Sentence(
                id=sid,
                dataset_id=ds.id,
                sequence_number=idx + 1,
                original_text=raw_text,
                language=rec.get("language") or None,
                sentiment=rec.get("sentiment") or None,
                sentence_type=rec.get("text_type") or rec.get("sentence_type") or None,
                abuse_category=rec.get("abuse_category") or None,
                abuse_severity=rec.get("abuse_severity") or None,
                review_status=rec.get("review_status") or "Pending review",
                annotator_notes=rec.get("annotator_notes") or None,
                annotation_metadata=rec.get("metadata") or None,
            )
            s.add(sent)

        s.flush()
        return ds, None

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def rename_dataset(
    user_id: str,
    dataset_id: str,
    new_name: str,
    session: Optional[Session] = None,
) -> Tuple[bool, str]:
    """Renames an owned dataset."""
    clean_name = (new_name or "").strip()
    if not clean_name:
        return False, "Dataset name cannot be empty."

    def _run(s: Session):
        ds = s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()
        if not ds:
            return False, "Dataset not found or unauthorized access."

        ds.name = clean_name
        ds.updated_at = datetime.now(timezone.utc)
        return True, "Dataset renamed successfully."

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def duplicate_dataset(
    user_id: str,
    dataset_id: str,
    new_name: Optional[str] = None,
    target_workspace_id: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tuple[Optional[Dataset], Optional[str]]:
    """Duplicates a dataset and all of its sentences."""
    def _run(s: Session):
        src_ds = s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()
        if not src_ds:
            return None, "Source dataset not found or unauthorized access."

        dest_ws_id = target_workspace_id or src_ds.workspace_id
        ws = s.query(Workspace).filter(
            Workspace.id == dest_ws_id,
            Workspace.owner_id == user_id,
        ).first()
        if not ws:
            return None, "Target workspace not found or unauthorized."

        dup_name = (new_name or f"Copy of {src_ds.name}").strip()
        new_ds = Dataset(
            workspace_id=dest_ws_id,
            owner_id=user_id,
            name=dup_name,
            original_filename=src_ds.original_filename,
            dataset_key=src_ds.dataset_key,
            total_sentences=src_ds.total_sentences,
            processing_status="ready",
        )
        s.add(new_ds)
        s.flush()

        # Copy sentences
        sentences = s.query(Sentence).filter(Sentence.dataset_id == src_ds.id).order_by(Sentence.sequence_number.asc()).all()
        for sent in sentences:
            new_sid = f"{new_ds.id}_{sent.sequence_number}"
            new_sent = Sentence(
                id=new_sid,
                dataset_id=new_ds.id,
                sequence_number=sent.sequence_number,
                original_text=sent.original_text,
                language=sent.language,
                sentiment=sent.sentiment,
                sentence_type=sent.sentence_type,
                review_status=sent.review_status,
                annotator_notes=sent.annotator_notes,
                annotation_metadata=sent.annotation_metadata,
            )
            s.add(new_sent)

        s.flush()
        return new_ds, None

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def delete_dataset(
    user_id: str,
    dataset_id: str,
    session: Optional[Session] = None,
) -> Tuple[bool, str]:
    """Deletes an owned dataset and its sentences/feedback."""
    def _run(s: Session):
        ds = s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()
        if not ds:
            return False, "Dataset not found or unauthorized access."

        s.delete(ds)
        return True, f"Dataset '{ds.name}' deleted successfully."

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


# =============================================================================
# SENTENCE ANNOTATION SERVICE
# =============================================================================

def get_dataset_sentences(
    user_id: str,
    dataset_id: str,
    session: Optional[Session] = None,
) -> Tuple[Optional[List[Sentence]], Optional[str]]:
    """Retrieves all sentences for a dataset, strictly verifying user ownership."""
    def _run(s: Session):
        ds = s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()
        if not ds:
            return None, "Dataset not found or unauthorized access."

        sentences = s.query(Sentence).filter(
            Sentence.dataset_id == dataset_id
        ).order_by(Sentence.sequence_number.asc()).all()
        return sentences, None

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def update_sentence_annotation(
    user_id: str,
    dataset_id: str,
    sentence_id: str,
    values: Dict[str, Any],
    session: Optional[Session] = None,
) -> Tuple[bool, str]:
    """
    Updates annotation for a sentence, verifying dataset and sentence ownership.
    Enforces LinguaLabel validation rules.
    """
    issues = validation_issues(values)
    if issues:
        return False, "; ".join(issues)

    def _run(s: Session):
        ds = s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()
        if not ds:
            return False, "Dataset not found or unauthorized access."

        sent = s.query(Sentence).filter(
            Sentence.id == sentence_id,
            Sentence.dataset_id == dataset_id,
        ).first()
        if not sent:
            return False, "Sentence not found in your dataset."

        sent.language = str(values.get("language", ""))
        sent.sentiment = str(values.get("sentiment", ""))
        sent.sentence_type = str(values.get("text_type", values.get("sentence_type", "")))
        if "abuse_category" in values:
            sent.abuse_category = str(values.get("abuse_category", ""))
        if "abuse_severity" in values:
            sent.abuse_severity = str(values.get("abuse_severity", ""))
        sent.review_status = str(values.get("review_status", ""))
        sent.annotator_notes = str(values.get("annotator_notes", ""))
        sent.updated_at = datetime.now(timezone.utc)
        ds.updated_at = datetime.now(timezone.utc)
        return True, "Annotation saved successfully."

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


# =============================================================================
# FEEDBACK & ACTIVE LEARNING SERVICE
# =============================================================================

def record_feedback(
    user_id: str,
    dataset_id: str,
    sentence_id: str,
    task: str,
    original_prediction: Any,
    corrected_label: str,
    annotator_notes: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tuple[Optional[Feedback], Optional[str]]:
    """Records human active-learning feedback strictly scoped to the user."""
    def _run(s: Session):
        # Verify dataset ownership
        ds = s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()
        if not ds:
            return None, "Dataset not found or unauthorized access."

        fb = Feedback(
            owner_id=user_id,
            dataset_id=dataset_id,
            sentence_id=sentence_id,
            task=task,
            original_prediction=original_prediction if isinstance(original_prediction, dict) else {"raw": str(original_prediction)},
            corrected_label=corrected_label,
            annotator_notes=annotator_notes or "",
        )
        s.add(fb)
        s.flush()
        return fb, None

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


def list_user_feedback(user_id: str, session: Optional[Session] = None) -> List[Feedback]:
    """Returns feedback records owned exclusively by user_id."""
    def _run(s: Session):
        return s.query(Feedback).filter(Feedback.owner_id == user_id).order_by(Feedback.created_at.desc()).all()

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


# =============================================================================
# PERSONAL DASHBOARD ANALYTICS SERVICE
# =============================================================================

def get_user_dashboard_stats(user_id: str, session: Optional[Session] = None) -> Dict[str, Any]:
    """
    Computes summary metrics strictly isolated to the authenticated user:
    - Total workspaces
    - Total datasets
    - Total sentences
    - Pending reviews
    - Completed datasets
    - Recent activity
    """
    def _run(s: Session):
        total_workspaces = s.query(Workspace).filter(Workspace.owner_id == user_id).count()
        total_datasets = s.query(Dataset).filter(Dataset.owner_id == user_id).count()

        # Sentence metrics via join on owned datasets
        sentence_stats = s.query(
            func.count(Sentence.id).label("total"),
            func.count(Sentence.id).filter(Sentence.review_status == "Reviewed — looks correct").label("reviewed"),
        ).join(Dataset, Sentence.dataset_id == Dataset.id).filter(Dataset.owner_id == user_id).one()

        tot_sentences = sentence_stats.total or 0
        tot_reviewed = sentence_stats.reviewed or 0
        pending_reviews = tot_sentences - tot_reviewed

        # Completed datasets: datasets where all sentences are reviewed and count > 0
        datasets = s.query(Dataset).filter(Dataset.owner_id == user_id).all()
        completed_datasets = 0
        for ds in datasets:
            if ds.total_sentences > 0:
                rev_in_ds = s.query(Sentence).filter(
                    Sentence.dataset_id == ds.id,
                    Sentence.review_status == "Reviewed — looks correct",
                ).count()
                if rev_in_ds >= ds.total_sentences:
                    completed_datasets += 1

        # Recent activity (latest 5 datasets updated)
        recent_datasets = s.query(Dataset).filter(
            Dataset.owner_id == user_id
        ).order_by(Dataset.updated_at.desc()).limit(5).all()

        recent_activity = [
            {
                "id": d.id,
                "name": d.name,
                "workspace": d.workspace.name if d.workspace else "General",
                "sentences": d.total_sentences,
                "updated_at": d.updated_at.strftime("%b %d, %Y %H:%M") if d.updated_at else "",
            }
            for d in recent_datasets
        ]

        return {
            "total_workspaces": total_workspaces,
            "total_datasets": total_datasets,
            "total_sentences": tot_sentences,
            "reviewed_sentences": tot_reviewed,
            "pending_reviews": pending_reviews,
            "completed_datasets": completed_datasets,
            "recent_activity": recent_activity,
        }

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


# =============================================================================
# EXPORT SERVICE WITH STRICT AUTHORIZATION
# =============================================================================

def export_user_dataset_df(
    user_id: str,
    dataset_id: str,
    session: Optional[Session] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Exports a dataset as a Pandas DataFrame.
    Strictly verifies ownership: if user_id is not the owner, returns (None, 'Unauthorized access').
    """
    def _run(s: Session):
        ds = s.query(Dataset).filter(
            Dataset.id == dataset_id,
            Dataset.owner_id == user_id,
        ).first()
        if not ds:
            return None, "Dataset not found or unauthorized access."

        sentences = s.query(Sentence).filter(
            Sentence.dataset_id == dataset_id
        ).order_by(Sentence.sequence_number.asc()).all()

        rows = []
        for sent in sentences:
            rows.append({
                "id": sent.id,
                "text": sent.original_text,
                "language": sent.language or "",
                "sentiment": sent.sentiment or "",
                "text_type": sent.sentence_type or "",
                "abuse_category": sent.abuse_category or "",
                "abuse_severity": sent.abuse_severity or "",
                "review_status": sent.review_status or "",
                "annotator_notes": sent.annotator_notes or "",
            })

        df = pd.DataFrame(rows)
        return df, None

    if session is not None:
        return _run(session)
    with get_db_session() as s:
        return _run(s)


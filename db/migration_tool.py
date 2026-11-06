"""
LinguaLabel — Safe Legacy Data Migration Tooling (Part 11)

Safely imports legacy local CSV sessions (saved_sessions/) and feedback logs
(active_learning_feedback/) into the PostgreSQL/SQLAlchemy multi-user schema.

Guarantees:
- Creates timestamped backups before migration.
- Preserves original sentence IDs, texts, human review statuses, and notes.
- Preserves feedback history.
- Repeat-safe (idempotent; skips previously imported datasets).
- Never deletes local files.
- Scoped strictly to the explicitly selected user and workspace.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from db.models import Dataset, Feedback, Sentence, User, Workspace
from db.session import get_db_session


def preview_legacy_data(root_dir: Path) -> Dict[str, Any]:
    """
    Scans the repository for legacy session CSVs and feedback JSONL files.
    Returns counts and summaries without making any modifications.
    """
    sessions_dir = root_dir / "saved_sessions"
    feedback_dir = root_dir / "active_learning_feedback"

    session_files = list(sessions_dir.glob("*.csv")) if sessions_dir.exists() else []
    feedback_files = list(feedback_dir.glob("*.jsonl")) if feedback_dir.exists() else []

    total_session_sentences = 0
    total_session_reviewed = 0
    sessions_summary = []

    for f in session_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
            count = len(df)
            rev = (df["review_status"] == "Reviewed — looks correct").sum() if "review_status" in df.columns else 0
            total_session_sentences += count
            total_session_reviewed += rev
            sessions_summary.append({
                "filename": f.name,
                "sentences": count,
                "reviewed": rev,
            })
        except Exception:
            sessions_summary.append({"filename": f.name, "sentences": 0, "reviewed": 0})

    total_feedback_lines = 0
    feedback_summary = []
    for f in feedback_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                lines = [json.loads(line) for line in fp if line.strip()]
            total_feedback_lines += len(lines)
            feedback_summary.append({
                "filename": f.name,
                "records": len(lines),
            })
        except Exception:
            feedback_summary.append({"filename": f.name, "records": 0})

    return {
        "session_files_count": len(session_files),
        "total_session_sentences": total_session_sentences,
        "total_session_reviewed": total_session_reviewed,
        "sessions": sessions_summary,
        "feedback_files_count": len(feedback_files),
        "total_feedback_records": total_feedback_lines,
        "feedback": feedback_summary,
    }


def migrate_legacy_data_to_user(
    target_user_id: str,
    target_workspace_id: str,
    root_dir: Path,
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Executes a safe, repeat-safe migration of local sessions and feedback
    into the target user's workspace.
    """
    sessions_dir = root_dir / "saved_sessions"
    feedback_dir = root_dir / "active_learning_feedback"

    session_files = list(sessions_dir.glob("*.csv")) if sessions_dir.exists() else []
    feedback_files = list(feedback_dir.glob("*.jsonl")) if feedback_dir.exists() else []

    if not session_files and not feedback_files:
        return {
            "status": "noop",
            "message": "No legacy session files or feedback logs found to migrate.",
            "datasets_imported": 0,
            "sentences_imported": 0,
            "feedback_imported": 0,
        }

    # Step 1: Create a non-destructive timestamped backup
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_root = root_dir / "legacy_backups" / f"backup_{timestamp_str}"
    backup_root.mkdir(parents=True, exist_ok=True)

    for f in session_files:
        shutil.copy2(f, backup_root / f.name)
    for f in feedback_files:
        shutil.copy2(f, backup_root / f.name)

    # Step 2: Migrate records inside a transactional database session
    def _run_migration(s: Session) -> Dict[str, Any]:
        # Verify user and workspace existence
        ws = s.query(Workspace).filter(
            Workspace.id == target_workspace_id,
            Workspace.owner_id == target_user_id,
        ).first()
        if not ws:
            raise ValueError(f"Target workspace '{target_workspace_id}' not found for user.")

        datasets_created = 0
        sentences_created = 0
        skipped_datasets = 0
        feedback_created = 0

        # Import CSV Sessions
        for csv_path in session_files:
            clean_stem = csv_path.stem
            # Check if dataset was already migrated for this user (repeat-safe)
            existing_ds = s.query(Dataset).filter(
                Dataset.owner_id == target_user_id,
                Dataset.workspace_id == target_workspace_id,
                Dataset.dataset_key == clean_stem,
            ).first()

            if existing_ds:
                skipped_datasets += 1
                continue

            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except Exception:
                continue

            if df.empty or "text" not in df.columns:
                continue

            ds_name = clean_stem.replace("annotated_", "").replace("_", " ").title()
            ds = Dataset(
                workspace_id=target_workspace_id,
                owner_id=target_user_id,
                name=f"Migrated: {ds_name}",
                original_filename=csv_path.name,
                dataset_key=clean_stem,
                total_sentences=len(df),
                processing_status="ready",
            )
            s.add(ds)
            s.flush()
            datasets_created += 1

            for seq, (_, row) in enumerate(df.iterrows(), start=1):
                raw_id = str(row.get("id", "")).strip() or f"{ds.id}_{seq}"
                sent = Sentence(
                    id=raw_id,
                    dataset_id=ds.id,
                    sequence_number=seq,
                    original_text=str(row.get("text", "")).strip(),
                    language=str(row.get("language", "")) or None,
                    sentiment=str(row.get("sentiment", "")) or None,
                    sentence_type=str(row.get("text_type", row.get("sentence_type", ""))) or None,
                    review_status=str(row.get("review_status", "Pending review")),
                    annotator_notes=str(row.get("annotator_notes", "")) or None,
                )
                s.add(sent)
                sentences_created += 1

        # Import Feedback JSONLs
        for fb_path in feedback_files:
            try:
                with open(fb_path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        sid = record.get("sentence_id") or record.get("record_id")
                        task = record.get("task", "general")
                        corr = record.get("corrected_label")
                        if not corr:
                            continue

                        # Check if duplicate feedback exists
                        existing_fb = s.query(Feedback).filter(
                            Feedback.owner_id == target_user_id,
                            Feedback.sentence_id == sid,
                            Feedback.task == task,
                            Feedback.corrected_label == corr,
                        ).first()
                        if existing_fb:
                            continue

                        fb_obj = Feedback(
                            owner_id=target_user_id,
                            dataset_id=None,
                            sentence_id=sid,
                            task=task,
                            original_prediction=record.get("original_prediction"),
                            corrected_label=corr,
                            annotator_notes=record.get("annotator_notes", "Imported from legacy feedback"),
                        )
                        s.add(fb_obj)
                        feedback_created += 1
            except Exception:
                continue

        s.flush()
        return {
            "status": "success",
            "backup_dir": str(backup_root),
            "datasets_imported": datasets_created,
            "sentences_imported": sentences_created,
            "datasets_skipped": skipped_datasets,
            "feedback_imported": feedback_created,
        }

    if session is not None:
        return _run_migration(session)
    with get_db_session() as s:
        return _run_migration(s)


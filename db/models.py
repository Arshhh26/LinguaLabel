"""
LinguaLabel — SQLAlchemy Models for Multi-User Architecture (Part 11)

Implements relational schema for Users, Workspaces, Datasets, Sentences, and Feedback
with strict foreign-key integrity, indexes, and UUID identifiers.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def generate_uuid() -> str:
    """Generates a canonical RFC 4122 UUID v4 string."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    workspaces: Mapped[List["Workspace"]] = relationship(
        "Workspace", back_populates="owner", cascade="all, delete-orphan"
    )
    datasets: Mapped[List["Dataset"]] = relationship(
        "Dataset", back_populates="owner", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="owner", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="workspaces")
    datasets: Mapped[List["Dataset"]] = relationship(
        "Dataset", back_populates="workspace", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_workspaces_owner_name", "owner_id", "name"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "name": self.name,
            "description": self.description or "",
            "dataset_count": len(self.datasets) if self.datasets else 0,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("workspaces.id", ondelete="CASCADE"), index=True, nullable=False
    )
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    original_filename: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    dataset_key: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    total_sentences: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processing_status: Mapped[str] = mapped_column(String(50), default="ready", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="datasets")
    owner: Mapped["User"] = relationship("User", back_populates="datasets")
    sentences: Mapped[List["Sentence"]] = relationship(
        "Sentence", back_populates="dataset", cascade="all, delete-orphan", order_by="Sentence.sequence_number"
    )
    feedbacks: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="dataset", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_datasets_owner_workspace", "owner_id", "workspace_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "owner_id": self.owner_id,
            "name": self.name,
            "original_filename": self.original_filename or "",
            "dataset_key": self.dataset_key or "",
            "total_sentences": self.total_sentences,
            "processing_status": self.processing_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Sentence(Base):
    __tablename__ = "sentences"

    # id can be standard UUID or stable deterministic ID (e.g. hash)
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    sequence_number: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sentence_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    review_status: Mapped[str] = mapped_column(String(50), default="Pending review", nullable=False)
    annotator_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    annotation_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="sentences")
    feedbacks: Mapped[List["Feedback"]] = relationship(
        "Feedback", back_populates="sentence", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_sentences_dataset_seq", "dataset_id", "sequence_number"),
        Index("ix_sentences_dataset_status", "dataset_id", "review_status"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "sequence_number": self.sequence_number,
            "original_text": self.original_text,
            "language": self.language or "",
            "sentiment": self.sentiment or "",
            "sentence_type": self.sentence_type or "",
            "review_status": self.review_status,
            "annotator_notes": self.annotator_notes or "",
            "annotation_metadata": self.annotation_metadata or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    dataset_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=True
    )
    sentence_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("sentences.id", ondelete="CASCADE"), index=True, nullable=True
    )
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    original_prediction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    corrected_label: Mapped[str] = mapped_column(String(100), nullable=False)
    annotator_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="feedbacks")
    dataset: Mapped[Optional["Dataset"]] = relationship("Dataset", back_populates="feedbacks")
    sentence: Mapped[Optional["Sentence"]] = relationship("Sentence", back_populates="feedbacks")

    __table_args__ = (
        Index("ix_feedbacks_owner_task", "owner_id", "task"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "dataset_id": self.dataset_id,
            "sentence_id": self.sentence_id,
            "task": self.task,
            "original_prediction": self.original_prediction,
            "corrected_label": self.corrected_label,
            "annotator_notes": self.annotator_notes or "",
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# ANONYMOUS VISITOR & USAGE ANALYTICS MODELS (Part 11)
# =============================================================================

class Visitor(Base):
    """
    Anonymous visitor record identified by a randomly generated first-party UUID.
    Zero invasive fingerprinting. Privacy-first tracking.
    """
    __tablename__ = "visitors"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    visit_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    pageview_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    sessions: Mapped[List["VisitorSession"]] = relationship(
        "VisitorSession", back_populates="visitor", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "first_seen_at": self.first_seen_at.isoformat() if self.first_seen_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "visit_count": self.visit_count,
            "pageview_count": self.pageview_count,
        }


class VisitorSession(Base):
    """
    Distinct visitor session with 30-minute inactivity boundary.
    Deduplicates repeated Streamlit script executions.
    """
    __tablename__ = "visitor_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    visitor_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("visitors.id", ondelete="CASCADE"), index=True, nullable=False
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    pageview_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    visitor: Mapped["Visitor"] = relationship("Visitor", back_populates="sessions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "pageview_count": self.pageview_count,
            "duration_seconds": self.duration_seconds,
        }


class PageView(Base):
    """
    Individual page view entry. Triggered strictly on view navigation.
    Streamlit component reruns on the same view are excluded.
    """
    __tablename__ = "page_views"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    visitor_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    page_name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "session_id": self.session_id,
            "page_name": self.page_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class UsageEvent(Base):
    """
    Privacy-conscious aggregate usage events.
    Strictly forbids storing actual sentence texts or uploaded file contents.
    """
    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    visitor_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    event_category: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    event_action: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    event_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True, nullable=False
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "visitor_id": self.visitor_id,
            "session_id": self.session_id,
            "event_category": self.event_category,
            "event_action": self.event_action,
            "event_count": self.event_count,
            "metadata_json": self.metadata_json or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }



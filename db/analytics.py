"""
LinguaLabel — Anonymous Visitor & Website Usage Analytics (Part 11)

A lightweight, secure, privacy-conscious analytics engine providing:
1. Anonymous First-Party Visitor Tracking:
   - Randomly generated UUIDs (no invasive fingerprinting).
   - Session deduplication (30-minute inactivity boundary; Streamlit reruns do not inflate visits).
   - Distinguishes Page Views vs. Visits/Sessions vs. Unique Visitors.
   - Computes Total Visits, Unique Visitors, New vs. Returning Visitors, Daily/Weekly/Monthly Active Visitors.
2. Aggregate Usage Analytics:
   - Sentence analyses count.
   - Datasets uploaded count.
   - CSV and Excel import counts.
   - Export counts.
   - Most-used feature breakdown.
   - Daily, weekly, and monthly activity trends.
3. Strict Zero-PII Privacy Protection:
   - Strictly forbids storing actual sentence texts or uploaded file contents.
   - Automatically sanitizes and redacts all event metadata.
4. Private Website Owner Access Verification:
   - Secure verification of the owner admin key for dashboard access.
"""

from datetime import datetime, timedelta, timezone
import hmac
from typing import Any, Dict, List, Optional, Tuple
import uuid

from sqlalchemy import desc, func, or_
from sqlalchemy.orm import Session

from db.config import ADMIN_SECRET_KEY
from db.models import PageView, UsageEvent, Visitor, VisitorSession
from db.session import get_db_session


SESSION_INACTIVITY_TIMEOUT_SECONDS = 1800  # 30 minutes


def generate_anonymous_id() -> str:
    """Generates an RFC 4122 compliant 32-hex anonymous identifier."""
    return uuid.uuid4().hex


def verify_admin_key(entered_key: str) -> bool:
    """
    Validates entered owner key against ADMIN_SECRET_KEY using constant-time comparison.
    """
    if not entered_key or not isinstance(entered_key, str):
        return False
    return hmac.compare_digest(entered_key.strip(), ADMIN_SECRET_KEY)


# =============================================================================
# PRIVACY ENFORCEMENT & METADATA SANITIZATION
# =============================================================================

# Forbidden patterns in event metadata to guarantee zero sentence text or dataset content is stored
FORBIDDEN_META_SUBSTRINGS = (
    "text",
    "sentence",
    "content",
    "data",
    "row",
    "payload",
    "input",
    "body",
    "sample",
    "df",
    "word",
    "prompt",
    "token",
    "password",
    "secret",
    "email",
    "dataset_content",
    "original",
)

ALLOWED_META_KEYS = {
    "format",
    "source",
    "row_count",
    "column_count",
    "task",
    "status",
    "category",
    "scope",
    "mode",
    "view",
    "count",
    "batch_size",
    "duration_ms",
    "file_type",
}


def sanitize_event_metadata(raw_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Strictly filters metadata dictionary against an explicit whitelist of safe fields
    to guarantee zero PII, user sentences, or dataset text rows are stored.
    """
    if not raw_metadata or not isinstance(raw_metadata, dict):
        return {}

    sanitized: Dict[str, Any] = {}
    for k, v in raw_metadata.items():
        k_clean = str(k).strip().lower()
        # Whitelist safe keys only
        if k_clean not in ALLOWED_META_KEYS:
            continue
        # Whitelist safe scalar types
        if isinstance(v, (int, float, bool)):
            sanitized[k_clean] = v
        elif isinstance(v, str):
            val_clean = v.strip()
            if len(val_clean) <= 100 and "\n" not in val_clean:
                sanitized[k_clean] = val_clean
        elif v is None:
            sanitized[k_clean] = None

    return sanitized


# =============================================================================
# VISITOR & SESSION TRACKING
# =============================================================================

def get_or_create_visitor(
    visitor_id: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tuple[Visitor, bool]:
    """
    Retrieves existing anonymous visitor by ID, or registers a new visitor.
    Returns: (visitor_record, is_new_visitor)
    """
    def _do(s: Session) -> Tuple[Visitor, bool]:
        now = datetime.now(timezone.utc)
        if visitor_id:
            vis = s.query(Visitor).filter(Visitor.id == visitor_id).first()
            if vis:
                vis.last_seen_at = now
                return vis, False

        # Create new anonymous visitor (visit_count starts at 0, incremented upon session start)
        new_id = visitor_id or generate_anonymous_id()
        new_vis = Visitor(
            id=new_id,
            first_seen_at=now,
            last_seen_at=now,
            visit_count=0,
            pageview_count=0,
        )
        s.add(new_vis)
        return new_vis, True

    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)


def register_or_resume_session(
    visitor_id: str,
    session_id: Optional[str] = None,
    session: Optional[Session] = None,
) -> Tuple[VisitorSession, bool]:
    """
    Maintains a visitor session with a 30-minute inactivity boundary.
    - If an active session exists (last activity <= 30 mins ago), it is updated and resumed.
    - If the session timed out (> 30 mins ago) or does not exist, a new session is started
      and the visitor's visit_count is incremented (marking them as a returning visitor if >1).
    Returns: (VisitorSession, is_new_session)
    """
    def _do(s: Session) -> Tuple[VisitorSession, bool]:
        now = datetime.now(timezone.utc)
        visitor = s.query(Visitor).filter(Visitor.id == visitor_id).first()
        if not visitor:
            visitor, _ = get_or_create_visitor(visitor_id, session=s)

        # 1. Attempt to resume specified session_id
        if session_id:
            curr_sess = s.query(VisitorSession).filter(
                VisitorSession.id == session_id,
                VisitorSession.visitor_id == visitor_id,
            ).first()

            if curr_sess:
                last_act = curr_sess.last_active_at
                if last_act.tzinfo is None:
                    last_act = last_act.replace(tzinfo=timezone.utc)
                idle_seconds = (now - last_act).total_seconds()

                if idle_seconds <= SESSION_INACTIVITY_TIMEOUT_SECONDS:
                    # Active session — touch last activity and duration
                    curr_sess.last_active_at = now
                    started = curr_sess.started_at
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    curr_sess.duration_seconds = max(0, int((now - started).total_seconds()))
                    visitor.last_seen_at = now
                    return curr_sess, False

        # 2. Check for any recent active session for this visitor within 30 minutes
        cutoff = now - timedelta(seconds=SESSION_INACTIVITY_TIMEOUT_SECONDS)
        recent_sess = s.query(VisitorSession).filter(
            VisitorSession.visitor_id == visitor_id,
            VisitorSession.last_active_at >= cutoff,
        ).order_by(desc(VisitorSession.last_active_at)).first()

        if recent_sess:
            recent_sess.last_active_at = now
            started = recent_sess.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            recent_sess.duration_seconds = max(0, int((now - started).total_seconds()))
            visitor.last_seen_at = now
            return recent_sess, False

        # 3. Inactive or first visit — initialize a NEW session
        new_sess_id = generate_anonymous_id()
        new_sess = VisitorSession(
            id=new_sess_id,
            visitor_id=visitor_id,
            started_at=now,
            last_active_at=now,
            pageview_count=0,
            duration_seconds=0,
        )
        s.add(new_sess)

        # Increment visitor's session count on each distinct new session
        visitor.visit_count += 1
        visitor.last_seen_at = now

        return new_sess, True


    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)


def record_page_view(
    visitor_id: str,
    session_id: str,
    page_name: str,
    session: Optional[Session] = None,
) -> PageView:
    """
    Records an individual page view. Triggered only when the active view changes.
    Reruns on the same view do not trigger page view increments.
    """
    def _do(s: Session) -> PageView:
        now = datetime.now(timezone.utc)
        clean_page = (page_name or "Home").strip()

        # Update visitor counts
        vis = s.query(Visitor).filter(Visitor.id == visitor_id).first()
        if vis:
            vis.pageview_count += 1
            vis.last_seen_at = now

        # Update session counts
        v_sess = s.query(VisitorSession).filter(VisitorSession.id == session_id).first()
        if v_sess:
            v_sess.pageview_count += 1
            v_sess.last_active_at = now
            started = v_sess.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            v_sess.duration_seconds = max(0, int((now - started).total_seconds()))

        # Insert PageView entry
        pv = PageView(
            visitor_id=visitor_id,
            session_id=session_id,
            page_name=clean_page,
            timestamp=now,
        )
        s.add(pv)
        return pv

    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)


def record_usage_event(
    event_category: str,
    event_action: str,
    count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    visitor_id: Optional[str] = None,
    session_id: Optional[str] = None,
    session: Optional[Session] = None,
) -> UsageEvent:
    """
    Records an aggregate usage event (analyses, imports, exports, reviews).
    Strictly sanitizes metadata to avoid recording raw sentence text or file data.
    """
    def _do(s: Session) -> UsageEvent:
        now = datetime.now(timezone.utc)
        safe_meta = sanitize_event_metadata(metadata)

        ue = UsageEvent(
            visitor_id=visitor_id,
            session_id=session_id,
            event_category=str(event_category).strip().lower(),
            event_action=str(event_action).strip().lower(),
            event_count=max(1, int(count)),
            metadata_json=safe_meta,
            timestamp=now,
        )
        s.add(ue)
        return ue

    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)


# =============================================================================
# WEBSITE OWNER ANALYTICS QUERIES & REPORTING
# =============================================================================

def get_visitor_analytics(
    time_window: str = "all",
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Aggregates visitor metrics:
    - Total visits (distinct sessions)
    - Total unique visitors
    - New visitors (visit_count == 1)
    - Returning visitors (visit_count > 1)
    - Daily visitors (last 24 hours)
    - Weekly visitors (last 7 days)
    - Monthly visitors (last 30 days)
    - Total pageviews
    - Last visit timestamp
    - Pageviews breakdown by page name
    """
    def _do(s: Session) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)

        # Base queries
        total_unique_visitors = s.query(func.count(Visitor.id)).scalar() or 0
        total_visits = s.query(func.count(VisitorSession.id)).scalar() or 0
        total_pageviews = s.query(func.count(PageView.id)).scalar() or 0

        # New vs. Returning Visitors
        new_visitors = s.query(func.count(Visitor.id)).filter(Visitor.visit_count == 1).scalar() or 0
        returning_visitors = s.query(func.count(Visitor.id)).filter(Visitor.visit_count > 1).scalar() or 0

        # Time-bounded active visitors
        cutoff_24h = now - timedelta(hours=24)
        cutoff_7d = now - timedelta(days=7)
        cutoff_30d = now - timedelta(days=30)

        daily_visitors = s.query(func.count(Visitor.id)).filter(Visitor.last_seen_at >= cutoff_24h).scalar() or 0
        weekly_visitors = s.query(func.count(Visitor.id)).filter(Visitor.last_seen_at >= cutoff_7d).scalar() or 0
        monthly_visitors = s.query(func.count(Visitor.id)).filter(Visitor.last_seen_at >= cutoff_30d).scalar() or 0

        # Most recent visit timestamp
        latest_visit = s.query(func.max(Visitor.last_seen_at)).scalar()

        # Average session duration in seconds
        avg_duration = s.query(func.avg(VisitorSession.duration_seconds)).scalar() or 0

        # Pageviews by page
        pv_rows = s.query(
            PageView.page_name,
            func.count(PageView.id).label("pv_count")
        ).group_by(PageView.page_name).order_by(desc("pv_count")).all()

        pageviews_by_page = {row.page_name: int(row.pv_count) for row in pv_rows}

        returning_ratio = round((returning_visitors / total_unique_visitors * 100), 1) if total_unique_visitors > 0 else 0.0

        return {
            "total_visits": total_visits,
            "total_unique_visitors": total_unique_visitors,
            "new_visitors": new_visitors,
            "returning_visitors": returning_visitors,
            "returning_ratio": returning_ratio,
            "daily_visitors": daily_visitors,
            "weekly_visitors": weekly_visitors,
            "monthly_visitors": monthly_visitors,
            "total_pageviews": total_pageviews,
            "last_visit_timestamp": latest_visit.isoformat() if latest_visit else None,
            "avg_session_duration_seconds": round(float(avg_duration), 1),
            "pageviews_by_page": pageviews_by_page,
        }

    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)


def get_usage_analytics(
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Aggregates feature and website usage events:
    - Number of sentence analyses
    - Number of datasets uploaded
    - Number of CSV imports
    - Number of Excel imports
    - Number of exports (CSV & Excel)
    - Most-used features breakdown
    """
    def _do(s: Session) -> Dict[str, Any]:
        # Helper to sum counts for an event_action
        def _sum_action(action: str) -> int:
            val = s.query(func.sum(UsageEvent.event_count)).filter(
                UsageEvent.event_action == action
            ).scalar()
            return int(val) if val else 0

        sentence_analyses_count = _sum_action("sentence_analysis")
        datasets_uploaded_count = _sum_action("dataset_upload")
        csv_imports_count = _sum_action("csv_import")
        excel_imports_count = _sum_action("excel_import")
        paste_imports_count = _sum_action("paste_import")
        csv_exports_count = _sum_action("export_csv")
        excel_exports_count = _sum_action("export_excel")
        total_exports = csv_exports_count + excel_exports_count

        # Most-used features breakdown
        feature_rows = s.query(
            UsageEvent.event_action,
            func.sum(UsageEvent.event_count).label("total_actions")
        ).group_by(UsageEvent.event_action).order_by(desc("total_actions")).limit(10).all()

        feature_breakdown = {row.event_action: int(row.total_actions) for row in feature_rows}

        return {
            "sentence_analyses_count": sentence_analyses_count,
            "datasets_uploaded_count": datasets_uploaded_count,
            "csv_imports_count": csv_imports_count,
            "excel_imports_count": excel_imports_count,
            "paste_imports_count": paste_imports_count,
            "exports_count": total_exports,
            "csv_exports_count": csv_exports_count,
            "excel_exports_count": excel_exports_count,
            "feature_breakdown": feature_breakdown,
        }

    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)


def get_activity_trends(
    days: int = 14,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """
    Returns day-by-day metrics for trend visualization:
    - Date
    - Daily Visits (Sessions)
    - Daily Unique Visitors
    - Page Views
    - Analyses
    - Imports
    - Exports
    """
    def _do(s: Session) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        trends = []

        for d in range(days - 1, -1, -1):
            day_start = (now - timedelta(days=d)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            date_label = day_start.strftime("%Y-%m-%d")

            # Unique visitors active on that day
            day_visitors = s.query(func.count(func.distinct(PageView.visitor_id))).filter(
                PageView.timestamp >= day_start,
                PageView.timestamp < day_end,
            ).scalar() or 0

            # Visits (sessions started on that day)
            day_visits = s.query(func.count(VisitorSession.id)).filter(
                VisitorSession.started_at >= day_start,
                VisitorSession.started_at < day_end,
            ).scalar() or 0

            # Pageviews on that day
            day_pvs = s.query(func.count(PageView.id)).filter(
                PageView.timestamp >= day_start,
                PageView.timestamp < day_end,
            ).scalar() or 0

            # Sentence analyses on that day
            day_analyses = s.query(func.sum(UsageEvent.event_count)).filter(
                UsageEvent.event_action == "sentence_analysis",
                UsageEvent.timestamp >= day_start,
                UsageEvent.timestamp < day_end,
            ).scalar() or 0

            # Imports on that day
            day_imports = s.query(func.sum(UsageEvent.event_count)).filter(
                UsageEvent.event_action == "dataset_upload",
                UsageEvent.timestamp >= day_start,
                UsageEvent.timestamp < day_end,
            ).scalar() or 0

            # Exports on that day
            day_exports = s.query(func.sum(UsageEvent.event_count)).filter(
                or_(
                    UsageEvent.event_action == "export_csv",
                    UsageEvent.event_action == "export_excel",
                ),
                UsageEvent.timestamp >= day_start,
                UsageEvent.timestamp < day_end,
            ).scalar() or 0

            trends.append({
                "date": date_label,
                "visitors": int(day_visitors),
                "visits": int(day_visits),
                "pageviews": int(day_pvs),
                "analyses": int(day_analyses),
                "imports": int(day_imports),
                "exports": int(day_exports),
            })

        return trends

    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)


def get_recent_events_log(
    limit: int = 50,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """
    Returns the most recent aggregate usage events for live audit log inspection.
    """
    def _do(s: Session) -> List[Dict[str, Any]]:
        events = s.query(UsageEvent).order_by(desc(UsageEvent.timestamp)).limit(limit).all()
        return [
            {
                "id": e.id,
                "timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if e.timestamp else "",
                "category": e.event_category,
                "action": e.event_action,
                "count": e.event_count,
                "metadata": e.metadata_json or {},
            }
            for e in events
        ]

    if session is not None:
        return _do(session)
    with get_db_session() as s:
        return _do(s)

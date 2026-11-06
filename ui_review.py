"""
LinguaLabel — Review Suggestions Page (Part 10)

Beginner-friendly prioritization and triage interface (powered by Active Learning).
Surfaces sentences where AI requires human judgment:
- AI is unsure about the sentiment or intent.
- Two analysis methods (rules vs transformer) gave conflicting suggestions.
- Labels are incomplete or missing.
- Sentence was previously flagged for another review.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from ml.active_learning import ActiveLearningQueue
from ml.engine import MLEngine
from ui_mappings import to_display_label


def map_reason_to_plain_language(primary_reason: str, prec: Any) -> str:
    """Translates technical active-learning reasons into friendly plain English."""
    pr = str(primary_reason).lower()
    if "disagreement" in pr or "conflict" in pr or (prec and prec.disagreement_score > 0):
        return "Two analysis methods gave different answers."
    elif "uncertain" in pr or (prec and prec.uncertainty_score > 0.40):
        return "AI is unsure about this sentence."
    elif "missing" in pr or (prec and len(prec.missing_fields) > 0):
        return "Some labels are missing."
    elif "flagged" in pr or "ambiguous" in pr or (prec and prec.is_ambiguous):
        return "This sentence was marked for another review."
    elif prec and prec.is_reviewed:
        return "Already verified by human annotator."
    return "Needs a quick human check."


def render_review_suggestions_page(
    root_dir: Path,
    df: pd.DataFrame,
    ml_engine: MLEngine,
    dataset_key: str,
):
    """Renders the friendly Review Suggestions interface."""
    st.markdown("""
    <div class="header-container" style="margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
            <span class="header-badge" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3);">
                ⚡ SMART REVIEW QUEUE
            </span>
        </div>
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.02em;">
            Review Suggestions
        </h1>
        <div class="header-subtitle" style="color: #94a3b8; font-size: 0.9rem;">
            LinguaLabel automatically prioritizes sentences where the AI is uncertain, conflicting, or where labels are incomplete.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No active dataset loaded. Please import sentences first.")
        return

    # Cache queue computation in session
    cache_key = f"friendly_review_queue_{dataset_key}_{len(df)}"
    if cache_key not in st.session_state or st.button("🔄 Refresh Suggestions", key="btn_refresh_friendly_queue"):
        with st.spinner("Analyzing sentences and sorting by review priority..."):
            try:
                from db.analytics import record_usage_event
                v_id = st.session_state.get("_visitor_id")
                s_id = st.session_state.get("_session_id")
                record_usage_event("review", "active_learning_review", count=1, metadata={"row_count": len(df)}, visitor_id=v_id, session_id=s_id)
            except Exception:
                pass

            queue_obj = ActiveLearningQueue(
                df=df,
                ml_engine=ml_engine,
                strategy="lowest_confidence",
                run_transformer=True,
            )
            st.session_state[cache_key] = queue_obj
    else:
        queue_obj = st.session_state[cache_key]

    summary = queue_obj.summary

    # Summary Cards
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.9rem 1rem;">
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Total Sentences</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; margin-top: 0.2rem;">{summary.total_records}</div>
            <div style="font-size: 0.7rem; color: #64748b;">In current dataset</div>
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 0.9rem 1rem;">
            <div style="font-size: 0.72rem; color: #f87171; text-transform: uppercase; font-weight: 600;">Needs Attention</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f87171; margin-top: 0.2rem;">{summary.high_priority_count}</div>
            <div style="font-size: 0.7rem; color: #64748b;">High priority to review</div>
        </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 0.9rem 1rem;">
            <div style="font-size: 0.72rem; color: #fbbf24; text-transform: uppercase; font-weight: 600;">AI is Unsure</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #fbbf24; margin-top: 0.2rem;">{summary.uncertain_count}</div>
            <div style="font-size: 0.7rem; color: #64748b;">Low model confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(168, 85, 247, 0.25); border-radius: 8px; padding: 0.9rem 1rem;">
            <div style="font-size: 0.72rem; color: #c084fc; text-transform: uppercase; font-weight: 600;">Differing AI Answers</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #c084fc; margin-top: 0.2rem;">{summary.disagreement_count}</div>
            <div style="font-size: 0.7rem; color: #64748b;">Requires human decision</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    # Simplified Filters
    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2.5])
    with f_col1:
        priority_filter = st.selectbox(
            "Priority",
            ["All Sentences", "Needs Attention (High Priority)", "Medium Priority", "Low Priority"],
            key="friendly_p_filter",
        )
    with f_col2:
        status_filter = st.selectbox(
            "Review Status",
            ["All", "Pending Review Only", "Reviewed & Approved", "Needs Second Review"],
            key="friendly_s_filter",
        )
    with f_col3:
        search_query = st.text_input(
            "Search Text",
            placeholder="Search sentence keywords...",
            key="friendly_search_q",
        )

    # Map friendly filter selections to internal queue query
    int_p_filter = "All"
    if "High" in priority_filter:
        int_p_filter = "High"
    elif "Medium" in priority_filter:
        int_p_filter = "Medium"
    elif "Low" in priority_filter:
        int_p_filter = "Low"

    int_s_filter = "All"
    if "Pending" in status_filter:
        int_s_filter = "Pending"
    elif "Reviewed" in status_filter:
        int_s_filter = "Reviewed"
    elif "Second" in status_filter:
        int_s_filter = "Ambiguous / Second Review"

    filtered_records = queue_obj.get_filtered_queue(
        priority_filter=int_p_filter,
        status_filter=int_s_filter,
        search_query=search_query,
    )

    st.caption(f"Showing **{len(filtered_records)}** sentence(s) needing attention.")

    if not filtered_records:
        st.success("🎉 No sentences match this filter. Everything is looking good!")
    else:
        for prec in filtered_records[:20]:
            plain_reason = map_reason_to_plain_language(prec.primary_reason, prec)
            
            # Find original index in df
            id_matches = df.index[df["id"].astype(str) == prec.record_id].tolist()
            target_idx = id_matches[0] if id_matches else 0

            with st.container(border=True):
                c_card1, c_card2 = st.columns([4, 1.3])
                with c_card1:
                    # Status badges
                    if prec.priority_level == "High":
                        badge_html = '<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.72rem;">NEEDS ATTENTION</span>'
                    elif prec.priority_level == "Medium":
                        badge_html = '<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem;">MEDIUM PRIORITY</span>'
                    else:
                        badge_html = '<span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem;">LOW PRIORITY</span>'

                    st.markdown(f"""
                    <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.35rem;">
                        <strong style="color: #38bdf8; font-size: 0.85rem;">Sentence #{target_idx + 1}</strong>
                        {badge_html}
                        <span style="font-size: 0.75rem; color: #94a3b8;">• Status: <strong>{prec.human_status or 'Pending Review'}</strong></span>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 500; color: #ffffff; line-height: 1.4; margin-bottom: 0.4rem;">
                        "{prec.text}"
                    </div>
                    <div style="font-size: 0.82rem; color: #fbbf24;">
                        <strong>Why review:</strong> {plain_reason}
                    </div>
                    """, unsafe_allow_html=True)

                with c_card2:
                    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                    if st.button("✏️ Review Sentence", key=f"btn_review_item_{prec.record_id}", width="stretch", type="primary"):
                        st.session_state.idx = target_idx
                        st.query_params["view"] = "Analyse & Label"
                        st.rerun()


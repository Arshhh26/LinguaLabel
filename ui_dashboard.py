"""
LinguaLabel — Personal User Dashboard (Part 11)

Displays isolated statistics and recent activity belonging exclusively
to the authenticated user.
"""

from typing import Any, Dict
import streamlit as st

from db.service import get_user_dashboard_stats


def render_personal_dashboard(current_user: Dict[str, Any]) -> None:
    """Renders the personal dashboard for the logged-in user."""
    user_id = current_user.get("id")
    username = current_user.get("username", "Annotator")

    stats = get_user_dashboard_stats(user_id)

    # -------------------------------------------------------------
    # HEADER & WELCOME
    # -------------------------------------------------------------
    st.markdown(f"""
    <div style="margin-bottom: 1.75rem;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.25rem;">
            Personal Workspace Dashboard
        </div>
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; margin: 0;">
            Welcome back, {username}! 👋
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.35rem;">
            Here is a summary of your personal multilingual NLP projects and review progress.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 5-CARD METRIC ROW
    # -------------------------------------------------------------
    m1, m2, m3, m4, m5 = st.columns(5)

    def metric_card(col, label: str, value: Any, sublabel: str, color: str = "#38bdf8"):
        with col:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.15rem 1rem; text-align: center; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);">
                <div style="font-size: 0.75rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
                <div style="font-size: 2.1rem; font-weight: 800; color: {color}; margin: 0.3rem 0; line-height: 1;">{value}</div>
                <div style="font-size: 0.75rem; color: #64748b;">{sublabel}</div>
            </div>
            """, unsafe_allow_html=True)

    metric_card(m1, "Workspaces", stats["total_workspaces"], "Isolated projects", "#38bdf8")
    metric_card(m2, "Datasets", stats["total_datasets"], "Uploaded batches", "#818cf8")
    metric_card(m3, "Sentences", stats["total_sentences"], "Total text units", "#a78bfa")
    metric_card(m4, "Pending Reviews", stats["pending_reviews"], f"{stats['reviewed_sentences']} reviewed", "#f59e0b" if stats["pending_reviews"] > 0 else "#22c55e")
    metric_card(m5, "Completed", stats["completed_datasets"], "Fully reviewed batches", "#22c55e")

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # RECENT ACTIVITY & QUICK ACTIONS
    # -------------------------------------------------------------
    c_left, c_right = st.columns([2.5, 1.2])

    with c_left:
        st.markdown("### 🕒 Recent Activity")
        recent = stats.get("recent_activity", [])

        if not recent:
            st.info("No datasets yet. Start by importing sentences or creating a new dataset below!")
        else:
            for item in recent:
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 10px; padding: 0.85rem 1.15rem; margin-bottom: 0.65rem; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.98rem; font-weight: 700; color: #f1f5f9;">{item['name']}</div>
                        <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.15rem;">
                            Workspace: <strong style="color: #cbd5e1;">{item['workspace']}</strong> • {item['sentences']} sentences • Updated: {item['updated_at']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with c_right:
        st.markdown("### ⚡ Quick Actions")
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.25rem;">
        """, unsafe_allow_html=True)

        if st.button("📥 Import New Dataset", use_container_width=True, type="primary"):
            st.session_state.main_nav_radio = "Import Sentences"
            st.rerun()

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

        if st.button("📁 Manage Workspaces", use_container_width=True):
            st.session_state.main_nav_radio = "Workspaces & Datasets"
            st.rerun()

        st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

        if st.button("✍️ Continue Annotation", use_container_width=True):
            st.session_state.main_nav_radio = "Analyse & Label"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


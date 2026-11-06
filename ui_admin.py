"""
LinguaLabel — Private Website Analytics & Owner Dashboard (Part 11)

A dedicated, secure administrative dashboard designed exclusively for the website owner.
Public visitors never see authentication or login screens.

Features:
1. Owner Access Gate:
   - Secret key authentication (verified using constant-time comparison).
   - Session-locked authentication state.
2. Visitor Analytics:
   - Total unique visitors, total visits (sessions), new vs. returning visitors.
   - Daily (24h), weekly (7d), and monthly (30d) active visitors.
   - Total pageviews and page breakdown.
   - Clear distinction between Page Views, Visits/Sessions, and Unique Visitors.
   - Documented limitation for browser storage clearing and device switching.
3. Website Usage Analytics:
   - Number of sentence analyses.
   - Number of datasets uploaded.
   - Number of CSV and Excel imports.
   - Number of exports (CSV & Excel).
   - Most-used feature breakdown.
   - Daily and weekly activity trends.
4. Privacy & Zero-PII Policy:
   - Highlights that actual sentence text and dataset contents are never collected.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from db.analytics import (
    get_activity_trends,
    get_recent_events_log,
    get_usage_analytics,
    get_visitor_analytics,
    verify_admin_key,
)
from ui_training import render_training_center


def render_owner_access_gate():
    """
    Renders a secure, discreet access gate for the website owner.
    Visitors are informed that the public website requires no login.
    """
    st.markdown("""
    <div class="header-container" style="margin-bottom: 1.5rem; text-align: center;">
        <div style="display: inline-flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem;">
            <span class="header-badge" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); font-size: 0.85rem; padding: 4px 12px;">
                🔒 PRIVATE ACCESS
            </span>
        </div>
        <h1 style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin: 0 0 0.5rem 0; letter-spacing: -0.02em;">
            Website Owner Analytics Portal
        </h1>
        <div class="header-subtitle" style="color: #94a3b8; font-size: 0.95rem; max-width: 620px; margin: 0 auto;">
            This private dashboard is reserved for the LinguaLabel website owner to monitor traffic, performance, and usage trends.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 1.75rem; box-shadow: 0 10px 25px rgba(0,0,0,0.3); margin-bottom: 1.5rem;">
            <div style="font-size: 0.92rem; color: #cbd5e1; margin-bottom: 1.25rem; line-height: 1.5;">
                Enter your <strong>Private Owner Key</strong> to inspect real-time visitor statistics and feature usage metrics.
            </div>
        """, unsafe_allow_html=True)

        entered_key = st.text_input(
            "Owner Secret Key",
            type="password",
            placeholder="Enter administrative secret key...",
            key="admin_secret_key_input",
            help="Configured via LINGUALABEL_ADMIN_KEY environment variable.",
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        c_btn1, c_btn2 = st.columns([2, 1.2])
        with c_btn1:
            if st.button("Unlock Dashboard →", type="primary", width="stretch"):
                if not entered_key:
                    st.error("Please enter the Owner Secret Key.")
                elif verify_admin_key(entered_key):
                    st.session_state["is_owner_authenticated"] = True
                    st.success("Access granted! Loading analytics...")
                    st.rerun()
                else:
                    st.error("Invalid secret key. Access denied.")

        with c_btn2:
            if st.button("← Back to Website", width="stretch"):
                st.session_state["_pending_navigation"] = "Home"
                if "admin" in st.query_params:
                    del st.query_params["admin"]
                if "owner" in st.query_params:
                    del st.query_params["owner"]
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Public Visitor Reminder Box
        st.markdown("""
        <div style="background: rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.25); border-radius: 8px; padding: 0.9rem 1rem; font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
            <strong style="color: #38bdf8;">🌐 Notice for Visitors:</strong>
            LinguaLabel is completely open. You do not need to register, create an account, or log in to analyse sentences, upload files, or export data.
        </div>
        """, unsafe_allow_html=True)


def render_admin_dashboard(root_dir: Optional[Path] = None, ml_engine: Optional[Any] = None):
    """
    Renders the full Private Website Analytics & Owner Dashboard.
    """
    # 1. Check owner authentication
    if not st.session_state.get("is_owner_authenticated", False):
        render_owner_access_gate()
        return

    # Header Bar with Lock action
    h_col1, h_col2 = st.columns([4, 1.3])
    with h_col1:
        st.markdown("""
        <div class="header-container" style="margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
                <span class="header-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);">
                    🛡️ PRIVATE OWNER DASHBOARD
                </span>
                <span style="font-size: 0.78rem; color: #4ade80; background: rgba(74, 222, 128, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(74, 222, 128, 0.25);">
                    ● Live Telemetry
                </span>
            </div>
            <h1 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.02em;">
                Website Traffic & Usage Analytics
            </h1>
            <div class="header-subtitle" style="color: #94a3b8; font-size: 0.9rem;">
                Real-time anonymous visitor tracking, session analysis, and aggregate feature telemetry.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with h_col2:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        if st.button("🔒 Lock Dashboard", width="stretch", help="Sign out of the owner analytics session"):
            st.session_state["is_owner_authenticated"] = False
            st.session_state["_pending_navigation"] = "Home"
            if "admin" in st.query_params:
                del st.query_params["admin"]
            if "owner" in st.query_params:
                del st.query_params["owner"]
            st.rerun()

    # Privacy Guarantee & Architecture Alert
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
            <span style="font-size: 1.2rem;">🔒</span>
            <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.5;">
                <strong style="color: #38bdf8;">Privacy-First Architecture:</strong>
                LinguaLabel tracks visitors using anonymous, randomly generated first-party identifiers.
                Zero invasive fingerprinting or cross-site tracking.
                <strong>Strict Zero-PII Policy:</strong> User sentences and uploaded dataset contents are <em style="color: #f87171;">never</em> collected or stored in analytics.
                <br>
                <span style="color: #94a3b8; font-size: 0.78rem;">
                    * Limitation: Visitors who clear their browser data, browse in private/incognito mode, or switch devices are counted as new visitors.
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch real-time analytics data
    vis_data = get_visitor_analytics()
    usage_data = get_usage_analytics()
    trends_data = get_activity_trends(days=14)
    recent_events = get_recent_events_log(limit=40)

    # Dashboard Tabs
    tab_visitors, tab_usage, tab_trends, tab_events, tab_training = st.tabs([
        "👥 Visitor Analytics",
        "🚀 Feature & Usage Telemetry",
        "📈 Activity Trends (14 Days)",
        "📜 Live Event Stream",
        "🧠 Model Training Center",
    ])

    # =========================================================================
    # TAB 1: VISITOR ANALYTICS
    # =========================================================================
    with tab_visitors:
        st.markdown("#### Anonymous Visitor Statistics")
        st.caption("Deduplicated visitor metrics powered by first-party session tracking.")

        # Top 4 Primary Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Total Unique Visitors",
                f"{vis_data['total_unique_visitors']:,}",
                help="Distinct anonymous visitor UUIDs recorded in the database.",
            )
        with c2:
            st.metric(
                "Total Visits (Sessions)",
                f"{vis_data['total_visits']:,}",
                help="Distinct visitor sessions (30-minute inactivity boundary; Streamlit reruns do not inflate this count).",
            )
        with c3:
            st.metric(
                "New Visitors",
                f"{vis_data['new_visitors']:,}",
                help="Visitors who have visited LinguaLabel exactly once.",
            )
        with c4:
            st.metric(
                "Returning Visitors",
                f"{vis_data['returning_visitors']:,}",
                delta=f"{vis_data['returning_ratio']}% retention",
                help="Visitors who have returned for 2 or more distinct sessions.",
            )

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)

        # Secondary 4 Cards: Recency & Activity
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric(
                "Daily Visitors (24h)",
                f"{vis_data['daily_visitors']:,}",
                help="Unique visitors active in the past 24 hours.",
            )
        with s2:
            st.metric(
                "Weekly Visitors (7d)",
                f"{vis_data['weekly_visitors']:,}",
                help="Unique visitors active in the past 7 days.",
            )
        with s3:
            st.metric(
                "Monthly Visitors (30d)",
                f"{vis_data['monthly_visitors']:,}",
                help="Unique visitors active in the past 30 days.",
            )
        with s4:
            st.metric(
                "Total Page Views",
                f"{vis_data['total_pageviews']:,}",
                help="Total view navigations across all visitors.",
            )

        st.markdown("---")

        # Metric Definitions & Pageview Breakdown
        col_pv1, col_pv2 = st.columns([1.5, 1])

        with col_pv1:
            st.markdown("##### Page Views by Section")
            st.caption("Which features and views do visitors explore most?")
            pv_dict = vis_data.get("pageviews_by_page", {})
            if pv_dict:
                pv_df = pd.DataFrame(list(pv_dict.items()), columns=["Page / View", "Page Views"])
                st.bar_chart(pv_df.set_index("Page / View"), color="#38bdf8")
            else:
                st.info("No page views recorded yet.")

        with col_pv2:
            st.markdown("##### Tracking Concepts Explained")
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 1rem; font-size: 0.83rem; color: #cbd5e1; line-height: 1.6;">
                <div style="margin-bottom: 0.6rem;">
                    <strong style="color: #38bdf8;">1. Page Views:</strong>
                    Recorded whenever a visitor switches navigation tabs (e.g. Home → Analyse & Label). Streamlit component reruns on the same page do not increase this count.
                </div>
                <div style="margin-bottom: 0.6rem;">
                    <strong style="color: #4ade80;">2. Visits / Sessions:</strong>
                    A continuous interaction window. If a visitor is inactive for over 30 minutes and resumes, a new session is recorded.
                </div>
                <div>
                    <strong style="color: #c084fc;">3. Unique Visitors:</strong>
                    Deduplicated count of unique first-party visitor tokens.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if vis_data.get("last_visit_timestamp"):
                st.markdown(f"""
                <div style="margin-top: 1rem; font-size: 0.8rem; color: #94a3b8;">
                    <strong>Last Visit Detected:</strong> {vis_data['last_visit_timestamp']}
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: USAGE & FEATURE TELEMETRY
    # =========================================================================
    with tab_usage:
        st.markdown("#### Website Usage & Feature Activity")
        st.caption("High-level aggregate counts of core actions performed on LinguaLabel.")

        u1, u2, u3, u4, u5 = st.columns(5)
        with u1:
            st.metric(
                "Sentence Analyses",
                f"{usage_data['sentence_analyses_count']:,}",
                help="Total sentences evaluated by AI (rule baseline or multilingual transformer).",
            )
        with u2:
            st.metric(
                "Datasets Uploaded",
                f"{usage_data['datasets_uploaded_count']:,}",
                help="Total uploaded or pasted datasets ingested into the workspace.",
            )
        with u3:
            st.metric(
                "CSV Imports",
                f"{usage_data['csv_imports_count']:,}",
                help="Total CSV spreadsheets imported.",
            )
        with u4:
            st.metric(
                "Excel Imports",
                f"{usage_data['excel_imports_count']:,}",
                help="Total Excel (.xlsx / .xls) spreadsheets imported.",
            )
        with u5:
            st.metric(
                "Total Exports",
                f"{usage_data['exports_count']:,}",
                delta=f"{usage_data['csv_exports_count']} CSV / {usage_data['excel_exports_count']} Excel",
                help="Total exported annotation files downloaded by visitors.",
            )

        st.markdown("---")

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("##### Most-Used Features & Actions")
            features = usage_data.get("feature_breakdown", {})
            if features:
                f_df = pd.DataFrame(list(features.items()), columns=["Action", "Executions"])
                st.dataframe(f_df, width="stretch", hide_index=True)
            else:
                st.info("No feature actions recorded yet.")

        with f_col2:
            st.markdown("##### Ingestion & Export Breakdown")
            ingest_data = {
                "CSV Imports": usage_data["csv_imports_count"],
                "Excel Imports": usage_data["excel_imports_count"],
                "Pasted Text": usage_data.get("paste_imports_count", 0),
                "CSV Exports": usage_data["csv_exports_count"],
                "Excel Exports": usage_data["excel_exports_count"],
            }
            ingest_df = pd.DataFrame(list(ingest_data.items()), columns=["Channel", "Count"])
            st.bar_chart(ingest_df.set_index("Channel"), color="#818cf8")

    # =========================================================================
    # TAB 3: ACTIVITY TRENDS
    # =========================================================================
    with tab_trends:
        st.markdown("#### 14-Day Activity & Traffic Trends")
        st.caption("Day-by-day evolution of visitors, sessions, sentence analyses, and file downloads.")

        if trends_data:
            tr_df = pd.DataFrame(trends_data)

            # Chart 1: Visitors & Visits
            st.markdown("##### Daily Unique Visitors & Visits")
            chart_vis_df = tr_df[["date", "visitors", "visits", "pageviews"]].set_index("date")
            st.line_chart(chart_vis_df)

            st.markdown("##### Daily Sentence Analyses & Workflows")
            chart_work_df = tr_df[["date", "analyses", "imports", "exports"]].set_index("date")
            st.bar_chart(chart_work_df)

            st.markdown("##### Daily Trend Breakdown Table")
            st.dataframe(tr_df, width="stretch", hide_index=True)
        else:
            st.info("Insufficient historical activity to render trends.")

    # =========================================================================
    # TAB 4: LIVE EVENT STREAM
    # =========================================================================
    with tab_events:
        st.markdown("#### Recent Aggregate Event Log (Last 40 Events)")
        st.caption("Audit trail demonstrating real-time event capture with complete privacy sanitization.")

        if recent_events:
            event_rows = []
            for ev in recent_events:
                meta_str = ", ".join(f"{k}={v}" for k, v in ev["metadata"].items()) if ev["metadata"] else "—"
                event_rows.append({
                    "Timestamp (UTC)": ev["timestamp"],
                    "Category": ev["category"].upper(),
                    "Action": ev["action"],
                    "Count": ev["count"],
                    "Safe Metadata (No Text)": meta_str,
                })
            ev_df = pd.DataFrame(event_rows)
            st.dataframe(ev_df, width="stretch", hide_index=True)
        else:
            st.info("No events logged yet. Perform an action on the website to observe live telemetry.")

        st.markdown("""
        <div style="background: rgba(74, 222, 128, 0.08); border: 1px solid rgba(74, 222, 128, 0.25); border-radius: 6px; padding: 0.75rem 1rem; margin-top: 1rem; font-size: 0.8rem; color: #86efac;">
            ✓ Verified: Zero sentence text or uploaded dataset contents are stored in the telemetry database.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB 5: MODEL TRAINING & IMPROVEMENT CENTER
    # =========================================================================
    with tab_training:
        render_training_center(root_dir=root_dir, ml_engine=ml_engine)


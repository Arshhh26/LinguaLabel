"""
LinguaLabel — AI-Assisted Multilingual Text Analysis & Human Annotation (Part 11)

Private Website Analytics & Owner Dashboard:
- Public Website: Zero authentication requirement. Visitors can immediately paste sentences,
  upload CSV/Excel datasets, run AI predictions, review suggestions, and export data.
  No registration, no login, no passwords, and no user profiles.
- Anonymous Visitor Analytics:
  - First-party anonymous visitor tokens (UUIDs). Zero invasive fingerprinting.
  - Session deduplication (30-minute inactivity boundary; Streamlit reruns do not inflate visits).
  - Clear distinction between Page Views, Visits/Sessions, and Unique Visitors.
  - Real-time tracking of daily, weekly, and monthly active visitors.
- Aggregate Usage Telemetry:
  - Tracks sentence analyses, datasets uploaded, CSV/Excel imports, and exports.
  - Strict Zero-PII Policy: actual sentence text and dataset contents are never logged.
- Private Owner Analytics Dashboard:
  - Accessible only to the website owner via secret key verification.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from data_manager import (
    compute_dataset_key,
    get_dataset_storage_path,
    merge_saved_annotations,
    save_dataset_record,
    stabilize_dataset,
    validation_issues,
    COLS,
    SUPPORTED_ENCODINGS,
)
from db.analytics import (
    get_or_create_visitor,
    record_page_view,
    record_usage_event,
    register_or_resume_session,
)
from db.session import init_db
from ml.engine import MLEngine
from ml.model_config import MLBackend

# Modular UI views
from ui_home import render_home_page
from ui_import import render_import_page
from ui_analyse import render_analyse_page
from ui_review import render_review_suggestions_page
from ui_results import render_my_results_page
from ui_export import render_export_page
from ui_settings import render_settings_page
from ui_admin import render_admin_dashboard

# Core Paths & Constants preserved for backward compatibility
ROOT = Path(__file__).parent
DATA = ROOT / "sample_data.csv"
OUT = ROOT / "annotated_output.csv"
OPTIONS = {
    "language": ["— Select —", "English", "Hindi", "Hinglish", "Other", "Uncertain"],
    "sentiment": ["— Select —", "Positive", "Negative", "Neutral", "Not applicable", "Uncertain"],
    "text_type": ["— Select —", "Statement", "Question", "Request / Instruction", "Greeting", "Other", "Uncertain"],
    "review_status": ["— Select —", "Reviewed — looks correct", "Needs second review", "Ambiguous", "Skip"],
}

# Streamlit Page Config
st.set_page_config(
    page_title="LinguaLabel — Multilingual Annotation & Analytics",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
/* Modern, dark, high-contrast theme */
.stApp {
    background-color: #0b0f19;
    color: #f1f5f9;
}
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0.2rem 1.25rem 0.2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 1.25rem;
}
.brand-icon {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.brand-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
}
.brand-badge {
    font-size: 0.65rem;
    padding: 2px 7px;
    border-radius: 9999px;
    background: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    font-weight: 600;
    text-transform: uppercase;
    margin-left: 0.4rem;
}
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin: 1.25rem 0 0.5rem 0.2rem;
}
div[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
div[data-testid="stMetricValue"] {
    font-weight: 800;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

# Initialize database schema
init_db()

# Initialize anonymous visitor & session analytics tracking
if "_visitor_id" not in st.session_state:
    q_vid = st.query_params.get("_vid", "").strip()
    visitor, _ = get_or_create_visitor(visitor_id=q_vid if q_vid else None)
    st.session_state["_visitor_id"] = visitor.id
    st.query_params["_vid"] = visitor.id

visitor_id = st.session_state["_visitor_id"]
curr_sess_id = st.session_state.get("_session_id")
v_sess, _ = register_or_resume_session(visitor_id, session_id=curr_sess_id)
st.session_state["_session_id"] = v_sess.id
session_id = v_sess.id

# Engine initialization
if "ml_engine" not in st.session_state:
    st.session_state.ml_engine = MLEngine()
ml_engine = st.session_state.ml_engine

# Fallback initialization for default sample or session dataset
if "active_df" not in st.session_state or st.session_state.active_df is None:
    sample_path = ROOT / "sample_data.csv"
    out_path = ROOT / "annotated_output.csv"
    if sample_path.exists():
        raw = pd.read_csv(sample_path, dtype=str)
        st_df, _ = stabilize_dataset(raw, "text", "default_sample")
        if out_path.exists():
            st_df = merge_saved_annotations(st_df, out_path)
        st.session_state.active_df = st_df
        st.session_state.active_dataset_key = "default_sample"
        st.session_state.active_storage_path = out_path
    else:
        st.session_state.active_df = pd.DataFrame()
        st.session_state.active_dataset_key = "default_sample"
        st.session_state.active_storage_path = out_path

df = st.session_state.active_df
dataset_key = st.session_state.get("active_dataset_key", "default_sample")
storage_path = st.session_state.get("active_storage_path", ROOT / "annotated_output.csv")


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🌐</div>
        <div>
            <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span class="brand-title">LinguaLabel</span>
                <span class="brand-badge">Part 11</span>
            </div>
            <div style="font-size: 0.74rem; color: #94a3b8; margin-top: 1px;">Multilingual Text Annotation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)

    nav_options = [
        "Home",
        "Import Sentences",
        "Analyse & Label",
        "Review Suggestions",
        "My Results",
        "Export Data",
        "Settings",
    ]

    is_admin_mode = (
        st.query_params.get("admin") in ("true", "1")
        or st.query_params.get("owner") in ("true", "1")
        or st.session_state.get("main_nav_radio") == "Owner Analytics"
    )

    if is_admin_mode and "Owner Analytics" not in nav_options:
        nav_options.append("Owner Analytics")

    # Handle URL query parameter routing
    default_idx = 0
    if is_admin_mode and "Owner Analytics" in nav_options:
        default_idx = nav_options.index("Owner Analytics")
    elif "view" in st.query_params:
        qp = st.query_params.get("view", "").strip().lower()
        if qp in ("home", "welcome", "dashboard"):
            default_idx = 0
        elif qp in ("import sentences", "import", "upload"):
            default_idx = 1
        elif qp in ("analyse & label", "annotate", "label", "analysis"):
            default_idx = 2
        elif qp in ("review suggestions", "active learning", "review"):
            default_idx = 3
        elif qp in ("my results", "quality dashboard", "dataset overview", "results"):
            default_idx = 4
        elif qp in ("export data", "export", "download"):
            default_idx = 5
        elif qp in ("settings", "model performance", "research"):
            default_idx = 6
        elif qp in ("owner analytics", "admin", "owner"):
            if "Owner Analytics" not in nav_options:
                nav_options.append("Owner Analytics")
            default_idx = nav_options.index("Owner Analytics")

    # Apply navigation changes before the radio widget is instantiated.
    pending_view = st.session_state.pop("_pending_navigation", None)
    if pending_view in nav_options:
        st.session_state["main_nav_radio"] = pending_view

    view = st.radio(
        "Navigation",
        nav_options,
        index=default_idx,
        key="main_nav_radio",
        label_visibility="collapsed",
    )

    # Dataset status pill in sidebar
    st.markdown('<div class="sidebar-section-title">Active Dataset</div>', unsafe_allow_html=True)
    total_recs = len(df) if df is not None else 0
    rev_recs = int(df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct").sum()) if (df is not None and "review_status" in df.columns) else 0

    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.85rem 1rem;">
        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Current Session</div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin: 0.2rem 0;">{dataset_key}</div>
        <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94a3b8; margin-top: 0.35rem;">
            <span>Reviewed:</span>
            <strong style="color: #4ade80;">{rev_recs} / {total_recs} ({int(rev_recs/total_recs*100) if total_recs else 0}%)</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Discreet website owner analytics entry
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    if st.button("🔐 Owner Analytics", width="stretch", help="Private website analytics for the site owner"):
        st.query_params["owner"] = "true"
        st.session_state["_pending_navigation"] = "Owner Analytics"
        st.rerun()


# =============================================================================
# VIEW ROUTER & DEDUPLICATED PAGE VIEW TRACKING
# =============================================================================

# Deduplicated page view tracking: only records when active view changes
last_view = st.session_state.get("_last_tracked_view")
if view != last_view:
    try:
        record_page_view(visitor_id, session_id, view)
        st.session_state["_last_tracked_view"] = view
    except Exception:
        pass

# Temporary action messages
if "msg" in st.session_state:
    st.success(st.session_state.pop("msg"))

if view == "Home":
    render_home_page(ROOT, df)

elif view == "Import Sentences":
    render_import_page(ROOT)

elif view == "Analyse & Label":
    show_explanations = st.session_state.get("show_explanations", True)
    render_analyse_page(ROOT, df, ml_engine, dataset_key, storage_path, show_explanations)

elif view == "Review Suggestions":
    render_review_suggestions_page(ROOT, df, ml_engine, dataset_key)

elif view == "My Results":
    render_my_results_page(ROOT, df, dataset_key)

elif view == "Export Data":
    render_export_page(ROOT, df, dataset_key)

elif view == "Settings":
    render_settings_page(ROOT, df, ml_engine, dataset_key, storage_path)

elif view == "Owner Analytics":
    render_admin_dashboard(ROOT, ml_engine)

"""
LinguaLabel — AI-Assisted Multilingual Text Analysis & Human Annotation (Part 11)

Multi-User SaaS Architecture:
- User registration, login, logout with Argon2id password hashing.
- Cryptographically signed HMAC-SHA256 session tokens.
- Strict multi-user data isolation across workspaces, datasets, sentences, and feedback.
- Personal Dashboard with user-specific metrics and recent activity tracking.
- Workspace & Dataset management with creation, renaming, and safe deletion.
- Non-destructive legacy data migration tooling.
- Full preservation of ML engine, active learning queue, and annotation integrity.
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
from db.auth import create_session_token, verify_session_token
from db.service import (
    export_user_dataset_df,
    get_dataset,
    list_datasets,
    list_workspaces,
)
from db.session import init_db
from ml.engine import MLEngine
from ml.model_config import MLBackend

# Modular UI views
from ui_auth import logout_current_user, render_auth_page
from ui_dashboard import render_personal_dashboard
from ui_workspaces import render_workspaces_page
from ui_home import render_home_page
from ui_import import render_import_page
from ui_analyse import render_analyse_page
from ui_review import render_review_suggestions_page
from ui_results import render_my_results_page
from ui_export import render_export_page
from ui_settings import render_settings_page

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

# Streamlit Page Configuration
st.set_page_config(
    page_title="LinguaLabel — Multilingual Text Labelling",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Deep Dark SaaS Styling (Linear / Vercel aesthetic)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #e2e8f0;
    background-color: #080c14;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #0e172a 0%, #080c14 70%) !important;
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1360px !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1322 0%, #080c14 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-top: 1.25rem !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.25rem 0 1.25rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.brand-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.brand-badge {
    font-size: 0.65rem;
    font-weight: 600;
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 2px 6px;
    border-radius: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.25rem 0 0.5rem 0;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 0.35rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 0.65rem 0.9rem !important;
    border-radius: 8px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
    margin-bottom: 2px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(14, 165, 233, 0.15) 0%, rgba(14, 165, 233, 0.04) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    color: #38bdf8 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize database schema
init_db()

# Engine initialization
if "ml_engine" not in st.session_state:
    st.session_state.ml_engine = MLEngine()
ml_engine = st.session_state.ml_engine


# =============================================================================
# AUTHENTICATION GATEWAY
# =============================================================================

auth_token = st.session_state.get("auth_token")
current_user = st.session_state.get("current_user")
is_guest = st.session_state.get("is_guest", False)

# Validate token if present
if auth_token:
    user_id = verify_session_token(auth_token)
    if not user_id:
        st.session_state.auth_token = None
        st.session_state.current_user = None
        current_user = None

if not current_user and not is_guest:
    render_auth_page()
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
    c_guest1, c_guest2, c_guest3 = st.columns([1, 2.2, 1])
    with c_guest2:
        if st.button("Continue as Demo Guest (Quick Local Inspection) →", use_container_width=True):
            st.session_state.is_guest = True
            st.rerun()
    st.stop()


# =============================================================================
# WORKSPACE & DATASET CONTEXT
# =============================================================================

if current_user:
    user_id = current_user["id"]
    workspaces = list_workspaces(user_id)
    if not workspaces:
        from db.service import create_workspace
        ws_new, _ = create_workspace(user_id, "Personal Workspace", "Default workspace")
        workspaces = [ws_new] if ws_new else []

    if "active_workspace_id" not in st.session_state or not st.session_state.active_workspace_id:
        st.session_state.active_workspace_id = workspaces[0].id if workspaces else ""

    # Synchronize active dataset if an active_dataset_id is set
    active_ds_id = st.session_state.get("active_dataset_id")
    if active_ds_id and ("active_df" not in st.session_state or st.session_state.active_df is None or st.session_state.active_df.empty):
        df_loaded, _ = export_user_dataset_df(user_id, active_ds_id)
        if df_loaded is not None and not df_loaded.empty:
            st.session_state.active_df = df_loaded
            ds_obj = get_dataset(user_id, active_ds_id)
            if ds_obj:
                st.session_state.active_dataset_key = ds_obj.name

# Fallback initialization for guest or default sample
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
# SIDEBAR NAVIGATION & USER PROFILE
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
            <div style="font-size: 0.74rem; color: #94a3b8; margin-top: 1px;">Multi-User Annotation Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User Profile Pill
    if current_user:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.75rem 0.9rem; margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Signed In As</div>
                    <div style="font-size: 0.92rem; font-weight: 700; color: #f8fafc; margin-top: 0.1rem;">👤 {current_user['username']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True, help="Log out of your account"):
            logout_current_user()
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 0.65rem 0.85rem; margin-bottom: 0.75rem;">
            <div style="font-size: 0.75rem; color: #fbbf24; font-weight: 600;">Demo Guest Mode</div>
            <div style="font-size: 0.7rem; color: #cbd5e1; margin-top: 0.1rem;">Sign in to isolate personal workspaces.</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔐 Sign In / Create Account", use_container_width=True):
            st.session_state.is_guest = False
            st.rerun()

    # Workspace Switcher (if logged in)
    if current_user and workspaces:
        st.markdown('<div class="sidebar-section-title">Current Workspace</div>', unsafe_allow_html=True)
        ws_dict = {w.name: w.id for w in workspaces}
        curr_active_ws = st.session_state.get("active_workspace_id", workspaces[0].id)
        default_name = next((name for name, wid in ws_dict.items() if wid == curr_active_ws), list(ws_dict.keys())[0])

        sel_ws_sidebar = st.selectbox(
            "Sidebar Workspace Selector",
            list(ws_dict.keys()),
            index=list(ws_dict.keys()).index(default_name),
            label_visibility="collapsed",
            key="sb_ws_select",
        )
        new_ws_id = ws_dict.get(sel_ws_sidebar)
        if new_ws_id != st.session_state.get("active_workspace_id"):
            st.session_state.active_workspace_id = new_ws_id
            st.rerun()

    st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)

    nav_options = [
        "Personal Dashboard",
        "Workspaces & Datasets",
        "Import Sentences",
        "Analyse & Label",
        "Review Suggestions",
        "My Results",
        "Export Data",
        "Settings",
    ]

    # Handle URL query parameter routing
    default_idx = 0
    if "view" in st.query_params:
        qp = st.query_params.get("view", "").strip().lower()
        if qp in ("dashboard", "home", "welcome"):
            default_idx = 0
        elif qp in ("workspaces", "projects", "datasets"):
            default_idx = 1
        elif qp in ("import sentences", "import", "upload"):
            default_idx = 2
        elif qp in ("analyse & label", "annotate", "label", "analysis"):
            default_idx = 3
        elif qp in ("review suggestions", "active learning", "review"):
            default_idx = 4
        elif qp in ("my results", "quality dashboard", "dataset overview", "results"):
            default_idx = 5
        elif qp in ("export data", "export", "download"):
            default_idx = 6
        elif qp in ("settings", "model performance", "research"):
            default_idx = 7

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


# =============================================================================
# VIEW ROUTER
# =============================================================================

# Temporary action messages
if "msg" in st.session_state:
    st.success(st.session_state.pop("msg"))

if view == "Personal Dashboard":
    if current_user:
        render_personal_dashboard(current_user)
    else:
        render_home_page(ROOT, df)

elif view == "Workspaces & Datasets":
    if current_user:
        render_workspaces_page(ROOT, current_user)
    else:
        st.info("Please sign in or create an account to access workspaces and dataset management.")

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

import csv
from pathlib import Path
import pandas as pd
import streamlit as st
import altair as alt

from data_manager import (
    load_csv_bytes,
    detect_text_column,
    stabilize_dataset,
    compute_dataset_key,
    get_dataset_storage_path,
    merge_saved_annotations,
    save_dataset_record,
    validation_issues,
    COLS,
    SUPPORTED_ENCODINGS,
)
from ml.engine import MLEngine
from ml.model_config import MLBackend, DEFAULT_SENTIMENT_CHECKPOINT, DEFAULT_TEXT_TYPE_CHECKPOINT

if "ml_engine" not in st.session_state:
    st.session_state.ml_engine = MLEngine()
ml_engine = st.session_state.ml_engine

ROOT = Path(__file__).parent
DATA = ROOT / "sample_data.csv"
OUT = ROOT / "annotated_output.csv"
OPTIONS = {
    "language": ["— Select —", "English", "Hindi", "Other", "Uncertain"],
    "sentiment": ["— Select —", "Positive", "Negative", "Neutral", "Not applicable", "Uncertain"],
    "text_type": ["— Select —", "Statement", "Question", "Request / Instruction", "Greeting", "Other", "Uncertain"],
    "review_status": ["— Select —", "Reviewed — looks correct", "Needs second review", "Ambiguous", "Skip"],
}

st.set_page_config(
    page_title="LinguaLabel — Multilingual Annotation Studio",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Linear & Vercel inspired deep dark SaaS dashboard)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global resets & typography */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #e2e8f0;
    background-color: #080c14;
}

/* Background subtle glow */
.stApp {
    background: radial-gradient(circle at 50% 0%, #0e172a 0%, #080c14 70%) !important;
}

/* Main container constraints */
.block-container {
    padding-top: 1.75rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1380px !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1322 0%, #080c14 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding-top: 1.25rem !important;
}

/* Sidebar Branding Header */
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

.brand-desc {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin: 1.25rem 0 0.5rem 0;
}

/* Sidebar Radio Navigation */
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

/* Sidebar Card Component */
.sidebar-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.75rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.sidebar-card-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.sidebar-card-stat {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.6rem;
}

.sidebar-stat-number {
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
}

.sidebar-stat-sub {
    font-size: 0.8rem;
    color: #38bdf8;
    font-weight: 600;
}

.progress-pill-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.75rem;
}

.progress-pill {
    flex: 1;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    padding: 0.35rem 0.5rem;
    text-align: center;
}

.progress-pill-num {
    font-size: 0.85rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #f1f5f9;
}

.progress-pill-label {
    font-size: 0.62rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Page Header Component */
.header-container {
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #38bdf8;
    background: rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.2);
    padding: 3px 10px;
    border-radius: 9999px;
    margin-bottom: 0.6rem;
}

.header-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}

.header-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* Streamlit Container Border Wrapper Styling (Cards) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.65) 0%, rgba(11, 17, 30, 0.8) 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(56, 189, 248, 0.2) !important;
    box-shadow: 0 8px 24px -2px rgba(0, 0, 0, 0.45) !important;
}

/* Sentence Hero Card */
.sentence-hero {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.9) 0%, rgba(13, 20, 37, 0.95) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-top: 3px solid #0ea5e9;
    border-radius: 14px;
    padding: 1.4rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45), 0 0 25px rgba(14, 165, 233, 0.08);
    margin-bottom: 1rem;
}

.sentence-hero-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.sentence-id-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: #38bdf8;
    background: rgba(14, 165, 233, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.3);
    padding: 4px 10px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}

.sentence-tags {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.sentence-tag {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.05);
    color: #94a3b8;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.sentence-text-box {
    background: rgba(8, 12, 20, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: 0.75rem 0 0.5rem 0;
    font-size: 1.35rem;
    line-height: 1.65;
    font-weight: 500;
    color: #ffffff;
    letter-spacing: -0.01em;
    min-height: 90px;
    display: flex;
    align-items: center;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.75) 0%, rgba(11, 16, 28, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.25);
}

.metric-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
}

.metric-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}

.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}

.metric-footer {
    font-size: 0.78rem;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

/* Status Pill Helpers */
.status-pill-reviewed {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
}

.status-pill-pending {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
}

.status-pill-followup {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
}

/* Validation Check Status Containers */
.validation-pass {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.03) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.75rem 0 1rem 0;
}

.validation-pass-icon {
    font-size: 1.15rem;
    color: #34d399;
}

.validation-pass-text {
    font-size: 0.88rem;
    font-weight: 600;
    color: #34d399;
}

.validation-pass-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-left: 0.25rem;
}

.validation-alert {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.04) 100%);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem 0 1rem 0;
}

.validation-alert-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.88rem;
    font-weight: 700;
    color: #fbbf24;
    margin-bottom: 0.4rem;
}

.validation-issue-item {
    font-size: 0.82rem;
    color: #cbd5e1;
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    margin-top: 0.25rem;
}

.validation-issue-bullet {
    color: #f59e0b;
    font-size: 0.75rem;
}

/* Streamlit Button Styling */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1rem !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    background: rgba(15, 23, 42, 0.8) !important;
    color: #e2e8f0 !important;
}

.stButton > button:hover {
    background: rgba(30, 41, 59, 0.9) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
    color: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%) !important;
    border: 1px solid #38bdf8 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(14, 165, 233, 0.4) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
    box-shadow: 0 6px 18px rgba(14, 165, 233, 0.55) !important;
    transform: translateY(-1px);
}

.stDownloadButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 0.55rem 1rem !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    background: rgba(14, 165, 233, 0.08) !important;
    color: #38bdf8 !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: rgba(14, 165, 233, 0.18) !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}

/* Streamlit Input & Selectbox Styling */
[data-baseweb="select"] > div {
    background-color: #0d1527 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

[data-baseweb="select"]:hover > div {
    border-color: rgba(56, 189, 248, 0.4) !important;
}

[data-testid="stTextArea"] textarea {
    background-color: #0d1527 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #0ea5e9 !important;
    box-shadow: 0 0 0 1px #0ea5e9 !important;
}

[data-testid="stTextInput"] input {
    background-color: #0d1527 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
}

/* Progress bar styling */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%) !important;
    border-radius: 9999px !important;
}

.stProgress > div > div > div {
    background-color: rgba(255, 255, 255, 0.08) !important;
    border-radius: 9999px !important;
    height: 7px !important;
}

/* Guideline item */
.guideline-box {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.85rem;
    background: rgba(8, 12, 20, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

.guideline-item {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}

.guideline-dot {
    color: #38bdf8;
    font-size: 0.65rem;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080c14;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}

/* ML Advisory Assistance Card */
.ml-assistance-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(14, 165, 233, 0.05) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1rem;
    position: relative;
}

.ml-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.ml-card-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #e0e7ff;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.ml-advisory-badge {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 2px 7px;
    border-radius: 4px;
}

.ml-suggestion-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.ml-suggestion-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.ml-suggestion-field {
    font-size: 0.76rem;
    font-weight: 600;
    color: #94a3b8;
    min-width: 75px;
}

.ml-suggestion-val {
    font-size: 0.88rem;
    font-weight: 700;
    color: #ffffff;
}

.ml-badges {
    display: flex;
    align-items: center;
    gap: 0.35rem;
}

.ml-badge-conf {
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
}

.ml-conf-high {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.ml-conf-med {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.ml-conf-low {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.ml-badge-method {
    font-size: 0.65rem;
    font-weight: 500;
    color: #cbd5e1;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2px 6px;
    border-radius: 4px;
}

.ml-uncertain-alert {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 6px;
    padding: 0.45rem 0.65rem;
    font-size: 0.72rem;
    color: #fbbf24;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Data Persistence & Loading (Part 1 Multi-Source Architecture)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🌐</div>
        <div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="brand-title">LinguaLabel</span>
                <span class="brand-badge">STUDIO</span>
            </div>
            <div class="brand-desc">Multilingual Text Annotation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Navigation</div>', unsafe_allow_html=True)
    view_options = ["Annotate", "Quality dashboard", "Dataset overview"]
    default_idx = 0
    if "view" in st.query_params:
        qp_view = st.query_params.get("view")
        if qp_view in view_options:
            default_idx = view_options.index(qp_view)
    view = st.radio(
        "Workspace",
        view_options,
        index=default_idx,
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section-title">Dataset Source</div>', unsafe_allow_html=True)
    dataset_source_mode = st.radio(
        "Dataset Source Mode",
        ["Default Sample (20 sentences)", "Upload Custom CSV"],
        key="dataset_source_mode_radio",
        label_visibility="collapsed"
    )

    active_dataset_name = "sample_data.csv"
    active_dataset_key = "default_sample"
    text_column_selected = "text"
    raw_df = None
    upload_error = None
    id_info = {}

    if dataset_source_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            key="custom_csv_uploader",
            label_visibility="collapsed",
            help="Upload a CSV with text to annotate. Original columns will be preserved."
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            parsed_df, enc_used, parse_err = load_csv_bytes(file_bytes, uploaded_file.name)
            if parse_err:
                st.error(f"⚠️ {parse_err}")
                upload_error = parse_err
            elif parsed_df is not None and not parsed_df.empty:
                raw_df = parsed_df
                active_dataset_name = uploaded_file.name
                active_dataset_key = compute_dataset_key(file_bytes)

                candidate_col = detect_text_column(raw_df)
                col_options = list(raw_df.columns)
                default_cand_idx = col_options.index(candidate_col) if candidate_col in col_options else 0

                st.markdown("<div style='font-size: 0.72rem; color: #94a3b8; margin: 4px 0 2px 0;'>Select text column:</div>", unsafe_allow_html=True)
                text_column_selected = st.selectbox(
                    "Select text column",
                    options=col_options,
                    index=default_cand_idx,
                    key=f"text_col_select_{active_dataset_key}",
                    label_visibility="collapsed"
                )

                with st.expander(f"🔍 Dataset Preview ({len(raw_df)} rows)", expanded=False):
                    st.caption(f"Encoding: `{enc_used}` · Total Columns: {len(raw_df.columns)}")
                    preview_cols = [text_column_selected] + [c for c in raw_df.columns if c != text_column_selected][:3]
                    st.dataframe(raw_df[preview_cols].head(4), hide_index=True, use_container_width=True)
        else:
            st.info("Upload a CSV file above, or switch back to the Default Sample.")

    # Fallback to default sample if custom upload is not provided or had error
    if raw_df is None:
        if not DATA.exists():
            st.error("sample_data.csv must be in the same folder as app.py.")
            st.stop()
        raw_df = pd.read_csv(DATA, keep_default_na=False, dtype=str)
        active_dataset_key = "default_sample"
        active_dataset_name = "sample_data.csv"
        text_column_selected = "text"

    # Reset active item index if dataset changed
    if st.session_state.get("current_dataset_key") != active_dataset_key:
        st.session_state.current_dataset_key = active_dataset_key
        st.session_state.idx = 0

    # Stabilize dataset: normalize IDs, map text column, retain original columns
    stable_df, id_info = stabilize_dataset(raw_df, text_column=text_column_selected, dataset_key=active_dataset_key)
    if id_info.get("id_generated") and dataset_source_mode == "Upload Custom CSV":
        st.caption(f"ℹ️ Generated stable IDs ({id_info['id_reason']}).")
    if id_info.get("empty_text_rows", 0) > 0:
        st.caption(f"⚠️ {id_info['empty_text_rows']} rows have empty text.")

    # Designate isolated storage path
    OUT = get_dataset_storage_path(active_dataset_key, ROOT)

    # Load and merge saved annotations
    def get_data():
        return merge_saved_annotations(stable_df.copy(), OUT)

    df = get_data()
    if df.empty:
        st.warning("Dataset is empty.")
        st.stop()

    total = len(df)
    processed = int(df["review_status"].astype(str).str.strip().ne("").sum())
    remaining = total - processed
    completion_pct = (processed / total) if total else 0.0

    st.markdown('<div class="sidebar-section-title">Dataset Metrics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-card">
        <div class="sidebar-card-label">Annotation Progress</div>
        <div class="sidebar-card-stat">
            <span class="sidebar-stat-number">{processed} <span style="font-size: 0.9rem; color: #64748b; font-weight: normal;">/ {total}</span></span>
            <span class="sidebar-stat-sub">{completion_pct:.0%} Done</span>
        </div>
        <div class="progress-pill-row">
            <div class="progress-pill">
                <div class="progress-pill-num" style="color: #34d399;">{processed}</div>
                <div class="progress-pill-label">Reviewed</div>
            </div>
            <div class="progress-pill">
                <div class="progress-pill-num" style="color: #fbbf24;">{remaining}</div>
                <div class="progress-pill-label">Pending</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(completion_pct)

    st.markdown(f"""
    <div style="margin-top: 1.5rem; padding: 0.75rem; border-radius: 8px; background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); font-size: 0.72rem; color: #64748b;">
        <div style="font-weight: 600; color: #94a3b8; margin-bottom: 2px;">● Local Persistence</div>
        Saved annotations automatically write to <code>{OUT.name}</code>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">AI Assistance</div>', unsafe_allow_html=True)
    enable_ml_suggestions = st.checkbox(
        "Enable ML Suggestions",
        value=True,
        key="enable_ml_suggestions_toggle",
        help="Display advisory model suggestions, confidence scores, and provenance in the workspace."
    )

    if enable_ml_suggestions:
        backend_choice = st.radio(
            "Model Backend",
            ["Baseline (Rules & TF-IDF)", "Multilingual Transformer (XLM-R)"],
            index=0 if ml_engine.active_backend == MLBackend.BASELINE else 1,
            key="ml_backend_selector_radio",
            help="Switch between lightweight heuristic models and deep multilingual transformer checkpoints."
        )
        target_backend = MLBackend.BASELINE if "Baseline" in backend_choice else MLBackend.TRANSFORMER
        if ml_engine.active_backend != target_backend:
            ml_engine.set_backend(target_backend)

        backend_info = ml_engine.get_backend_info()

        if target_backend == MLBackend.TRANSFORMER:
            with st.expander("Transformer Details & Status", expanded=False):
                st.caption(f"**Device:** `{backend_info['device']}`")
                st.caption(f"**Sentiment Checkpoint:** `{backend_info['sentiment_checkpoint']}`")
                st.caption(f"**Text-Type Checkpoint:** `{backend_info['text_type_checkpoint']}`")

                if not backend_info["torch_available"] or not backend_info["transformers_available"]:
                    st.warning("⚠️ `torch` / `transformers` not installed. Gracefully using baseline fallback.")
                else:
                    is_cached = backend_info["sentiment_cached"] and backend_info["text_type_cached"]
                    if is_cached:
                        st.success("✅ Models loaded in memory cache")
                    else:
                        st.info("ℹ️ Models load on-demand. Click below to pre-warm cache.")
                        if st.button("Pre-warm Models", key="btn_prewarm_models"):
                            with st.spinner("Loading transformer models into cache..."):
                                ml_engine.predict_all("Warmup sentence")
                                st.rerun()

# Compute export filenames
safe_dataset_stem = Path(active_dataset_name).stem.replace(" ", "_")
download_annotated_name = "annotated_output.csv" if active_dataset_key == "default_sample" else f"annotated_{safe_dataset_stem}.csv"
download_filtered_name = "lingualabel_filtered.csv" if active_dataset_key == "default_sample" else f"filtered_{safe_dataset_stem}.csv"
download_dataset_name = "lingualabel_dataset.csv" if active_dataset_key == "default_sample" else f"dataset_{safe_dataset_stem}.csv"


# =============================================================
# VIEW 1: QUALITY DASHBOARD
# =============================================================
if view == "Quality dashboard":
    st.markdown("""
    <div class="header-container">
        <div class="header-badge">Quality Analytics</div>
        <h1 class="header-title">📊 Quality Dashboard</h1>
        <div class="header-subtitle">Live health summary, distribution insights, and review queue triage for all annotations.</div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Cards
    follow = df[df.review_status.isin(["Needs second review", "Ambiguous", "Skip"])]
    follow_count = len(follow)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card" style="border-top: 3px solid #38bdf8;">
            <div class="metric-top">
                <span class="metric-label">Total Sentences</span>
                <span class="metric-icon" style="background: rgba(14, 165, 233, 0.15); color: #38bdf8;">🌐</span>
            </div>
            <div class="metric-value">{total}</div>
            <div class="metric-footer">Dataset total sample size</div>
        </div>
        <div class="metric-card" style="border-top: 3px solid #34d399;">
            <div class="metric-top">
                <span class="metric-label">Reviewed / Status Set</span>
                <span class="metric-icon" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">✓</span>
            </div>
            <div class="metric-value">{processed}</div>
            <div class="metric-footer" style="color: #34d399;">{completion_pct:.1%} coverage completed</div>
        </div>
        <div class="metric-card" style="border-top: 3px solid #fbbf24;">
            <div class="metric-top">
                <span class="metric-label">Pending Review</span>
                <span class="metric-icon" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24;">⏳</span>
            </div>
            <div class="metric-value">{remaining}</div>
            <div class="metric-footer">Awaiting annotator action</div>
        </div>
        <div class="metric-card" style="border-top: 3px solid #f87171;">
            <div class="metric-top">
                <span class="metric-label">Review Queue Flags</span>
                <span class="metric-icon" style="background: rgba(239, 68, 68, 0.15); color: #f87171;">⚠️</span>
            </div>
            <div class="metric-value">{follow_count}</div>
            <div class="metric-footer">{('Action required' if follow_count > 0 else 'Queue clear')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Visualizations using Altair for tailored dark-theme aesthetics
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.25rem;">🌐 Language Distribution</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.75rem;">Breakdown of tagged languages</div>', unsafe_allow_html=True)
            
            lang_df = df[df.language.astype(str).str.strip().ne("")].language.value_counts().reset_index()
            lang_df.columns = ["Language", "Count"]
            
            if lang_df.empty:
                st.info("No language labels saved yet.")
            else:
                lang_chart = alt.Chart(lang_df).mark_bar(cornerRadiusEnd=6, height=26).encode(
                    x=alt.X("Count:Q", title="Count", axis=alt.Axis(format="d", tickMinStep=1, gridColor="rgba(255,255,255,0.06)", tickColor="transparent", labelColor="#94a3b8", titleColor="#64748b")),
                    y=alt.Y("Language:N", title=None, sort="-x", axis=alt.Axis(labelColor="#cbd5e1", labelFontSize=12)),
                    color=alt.Color("Language:N", scale=alt.Scale(range=["#0ea5e9", "#6366f1", "#10b981", "#f59e0b"]), legend=None),
                    tooltip=["Language", "Count"]
                ).properties(height=170).configure_view(strokeOpacity=0)
                st.altair_chart(lang_chart, use_container_width=True)

    with chart_col2:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.25rem;">🎭 Sentiment Labels</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.75rem;">Sentiment polarity across dataset</div>', unsafe_allow_html=True)
            
            sent_df = df[df.sentiment.astype(str).str.strip().ne("")].sentiment.value_counts().reset_index()
            sent_df.columns = ["Sentiment", "Count"]
            
            if sent_df.empty:
                st.info("No sentiment labels saved yet.")
            else:
                sent_chart = alt.Chart(sent_df).mark_bar(cornerRadiusEnd=6, height=26).encode(
                    x=alt.X("Count:Q", title="Count", axis=alt.Axis(format="d", tickMinStep=1, gridColor="rgba(255,255,255,0.06)", tickColor="transparent", labelColor="#94a3b8", titleColor="#64748b")),
                    y=alt.Y("Sentiment:N", title=None, sort="-x", axis=alt.Axis(labelColor="#cbd5e1", labelFontSize=12)),
                    color=alt.Color("Sentiment:N", scale=alt.Scale(range=["#34d399", "#38bdf8", "#818cf8", "#94a3b8", "#f87171"]), legend=None),
                    tooltip=["Sentiment", "Count"]
                ).properties(height=170).configure_view(strokeOpacity=0)
                st.altair_chart(sent_chart, use_container_width=True)

    # Text Types Card
    with st.container(border=True):
        st.markdown('<div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0.25rem;">📝 Text Type Classification</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.75rem;">Syntactic and pragmatic category distribution</div>', unsafe_allow_html=True)
        
        type_df = df[df.text_type.astype(str).str.strip().ne("")].text_type.value_counts().reset_index()
        type_df.columns = ["Text Type", "Count"]
        
        if type_df.empty:
            st.info("No text-type labels saved yet.")
        else:
            type_chart = alt.Chart(type_df).mark_bar(cornerRadiusEnd=6, height=26).encode(
                x=alt.X("Count:Q", title="Count", axis=alt.Axis(format="d", tickMinStep=1, gridColor="rgba(255,255,255,0.06)", tickColor="transparent", labelColor="#94a3b8", titleColor="#64748b")),
                y=alt.Y("Text Type:N", title=None, sort="-x", axis=alt.Axis(labelColor="#cbd5e1", labelFontSize=12)),
                color=alt.Color("Text Type:N", scale=alt.Scale(range=["#38bdf8", "#818cf8", "#a78bfa", "#f472b6", "#fbbf24"]), legend=None),
                tooltip=["Text Type", "Count"]
            ).properties(height=170).configure_view(strokeOpacity=0)
            st.altair_chart(type_chart, use_container_width=True)

    # Quality Review Queue Section
    st.markdown("""
    <div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
        <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">⚖️ Quality Review Queue</h3>
        <div style="font-size: 0.82rem; color: #94a3b8;">Items marked for second review, flagged as ambiguous, or skipped.</div>
    </div>
    """, unsafe_allow_html=True)

    if follow.empty:
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎉</div>
                <div style="font-weight: 700; font-size: 1.1rem; color: #34d399;">Quality Review Queue is Clear</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">No items currently require secondary review or disambiguation.</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.dataframe(
            follow[["id", "text"] + COLS],
            use_container_width=True,
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("ID", width="small"),
                "text": st.column_config.TextColumn("Sentence Text", width="large"),
                "language": st.column_config.TextColumn("Language", width="small"),
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "text_type": st.column_config.TextColumn("Text Type", width="medium"),
                "review_status": st.column_config.TextColumn("Status", width="medium"),
                "annotator_notes": st.column_config.TextColumn("Annotator Notes", width="large"),
            }
        )

    # Review Status Breakdown & Export Row
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    q_col1, q_col2 = st.columns([1.5, 1])
    
    with q_col1:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.75rem;">📋 Status Breakdown</div>', unsafe_allow_html=True)
            status_counts = df[df.review_status.astype(str).str.strip().ne("")].review_status.value_counts()
            if status_counts.empty:
                st.info("No review statuses have been set.")
            else:
                status_df = status_counts.rename_axis("Review Status").reset_index(name="Count")
                status_df["Percentage"] = (status_df["Count"] / total * 100).map("{:.1f}%".format)
                st.dataframe(status_df, hide_index=True, use_container_width=True)

    with q_col2:
        with st.container(border=True):
            st.markdown('<div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem;">💾 Export Annotated Dataset</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 1.25rem;">Download the full dataset including all saved labels and notes.</div>', unsafe_allow_html=True)
            st.download_button(
                "📥 Download Annotated CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                download_annotated_name,
                "text/csv",
                use_container_width=True
            )

    st.stop()


# =============================================================
# VIEW 2: DATASET OVERVIEW
# =============================================================
if view == "Dataset overview":
    st.markdown("""
    <div class="header-container">
        <div class="header-badge">Data Explorer</div>
        <h1 class="header-title">📋 Dataset Overview</h1>
        <div class="header-subtitle">Search, filter, and inspect records across the multilingual dataset.</div>
    </div>
    """, unsafe_allow_html=True)

    # Search & Filter Container
    with st.container(border=True):
        st.markdown('<div style="font-weight: 700; font-size: 0.95rem; color: #38bdf8; margin-bottom: 0.75rem;">🔍 Search & Filters</div>', unsafe_allow_html=True)
        
        search_text = st.text_input("Search sentence text", placeholder="Type a word, phrase, or keyword to filter sentences...", label_visibility="collapsed")
        
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            language_filter = st.selectbox("Language filter", ["All"] + OPTIONS["language"][1:])
        with f_col2:
            sentiment_filter = st.selectbox("Sentiment filter", ["All"] + OPTIONS["sentiment"][1:])
        with f_col3:
            status_filter = st.selectbox("Review status filter", ["All"] + OPTIONS["review_status"][1:])

        st.markdown('<div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #94a3b8; margin: 0.6rem 0 0.3rem 0;">Triage Queue Quick Filters</div>', unsafe_allow_html=True)
        quick_filter = st.radio(
            "Triage Queue Quick Filters",
            ["All Records", "Pending / Unreviewed", "Completed / Reviewed", "Follow-Up Queue (Needs review / Ambiguous / Skip)"],
            horizontal=True,
            key="dataset_overview_queue_filter",
            label_visibility="collapsed"
        )

    filtered = df.copy()
    if quick_filter == "Pending / Unreviewed":
        filtered = filtered[filtered["review_status"].astype(str).str.strip().eq("")]
    elif quick_filter == "Completed / Reviewed":
        filtered = filtered[filtered["review_status"].astype(str).str.strip().eq("Reviewed — looks correct")]
    elif quick_filter.startswith("Follow-Up Queue"):
        filtered = filtered[filtered["review_status"].astype(str).isin(["Needs second review", "Ambiguous", "Skip"])]

    if search_text.strip():
        filtered = filtered[filtered["text"].astype(str).str.contains(search_text.strip(), case=False, na=False, regex=False)]
    if language_filter != "All":
        filtered = filtered[filtered["language"].astype(str).eq(language_filter)]
    if sentiment_filter != "All":
        filtered = filtered[filtered["sentiment"].astype(str).eq(sentiment_filter)]
    if status_filter != "All":
        filtered = filtered[filtered["review_status"].astype(str).eq(status_filter)]

    # Result Header Chips
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin: 1.25rem 0 0.5rem 0;">
        <div style="font-size: 0.85rem; color: #94a3b8;">
            Showing <strong style="color: #ffffff;">{len(filtered)}</strong> of <strong style="color: #ffffff;">{len(df)}</strong> records
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Styled Table
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", width="small"),
            "text": st.column_config.TextColumn("Sentence Text", width="large"),
            "language": st.column_config.TextColumn("Language", width="small"),
            "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "text_type": st.column_config.TextColumn("Text Type", width="medium"),
            "review_status": st.column_config.TextColumn("Review Status", width="medium"),
            "annotator_notes": st.column_config.TextColumn("Annotator Notes", width="large"),
        }
    )

    # Export Controls Bar
    with st.container(border=True):
        st.markdown('<div style="font-weight: 700; font-size: 0.95rem; color: #ffffff; margin-bottom: 0.25rem;">Export Controls</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 0.75rem;">Export your current filtered view or the entire dataset in standard UTF-8 CSV format.</div>', unsafe_allow_html=True)

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                f"📥 Download Filtered CSV ({len(filtered)} records)",
                filtered.to_csv(index=False).encode("utf-8-sig"),
                download_filtered_name,
                "text/csv",
                use_container_width=True
            )
        with d_col2:
            st.download_button(
                f"💾 Download Full Dataset CSV ({len(df)} records)",
                df.to_csv(index=False).encode("utf-8-sig"),
                download_dataset_name,
                "text/csv",
                use_container_width=True
            )

    st.stop()


# =============================================================
# VIEW 3: ANNOTATE (MAIN WORKSPACE)
# =============================================================

# State Initialization
if "idx" not in st.session_state:
    if "item" in st.query_params and st.query_params["item"].isdigit():
        st.session_state.idx = max(0, min(int(st.query_params["item"]) - 1, total - 1))
    else:
        st.session_state.idx = 0
idx = max(0, min(st.session_state.idx, total - 1))
st.session_state.idx = idx

row = df.iloc[idx]
row_id = str(row["id"])
row_text = str(row["text"])
word_count = len(row_text.split())
char_count = len(row_text)

# Validation rules (Exact logic preserved)
def validation_issues(values):
    issues = []
    required = ["language", "sentiment", "text_type", "review_status"]
    for column in required:
        if not str(values.get(column, "")).strip() or values[column] == "— Select —":
            issues.append(f"Choose a value for {column.replace('_', ' ')}.")
    notes = str(values.get("annotator_notes", "")).strip()
    for column in ["language", "sentiment", "text_type"]:
        if values.get(column) == "Uncertain" and not notes:
            issues.append(f"Add an annotator note explaining the uncertain {column.replace('_', ' ')}.")
    if values.get("review_status") == "Reviewed — looks correct":
        uncertain = [c.replace("_", " ") for c in ["language", "sentiment", "text_type"]
                     if values.get(c) in ("Uncertain", "— Select —", "")]
        if uncertain:
            issues.append("Cannot mark as reviewed while labels are missing or uncertain: " + ", ".join(uncertain) + ".")
    if values.get("review_status") in ("Needs second review", "Ambiguous", "Skip") and not notes:
        issues.append("Add a note explaining this follow-up, ambiguous, or skipped item.")
    return issues

# Main Workspace Header
st.markdown("""
<div class="header-container">
    <div class="header-badge">Annotation Studio</div>
    <h1 class="header-title">🌐 Annotation Workspace</h1>
    <div class="header-subtitle">Evaluate multilingual sentences according to annotation standards. Labels are entered and verified by human review.</div>
</div>
""", unsafe_allow_html=True)

# Jump to Sentence Quick Selector Bar
sentence_labels = [
    f"Item #{r['id']} ({'✓ Reviewed' if str(r.get('review_status', '')).strip() else '⏳ Pending'}) — {str(r['text'])[:45]}..."
    for _, r in df.iterrows()
]

selector_col1, selector_col2 = st.columns([2.5, 1])
with selector_col1:
    selected_label = st.selectbox(
        "Jump directly to sentence",
        options=sentence_labels,
        index=idx,
        key=f"sentence_jumper_{idx}",
        label_visibility="collapsed"
    )
    chosen_idx = sentence_labels.index(selected_label)
    if chosen_idx != idx:
        curr_saved_status = str(row.get("review_status", "")).strip() or "— Select —"
        curr_widget_status = st.session_state.get(f"review_status_{row_id}", curr_saved_status)
        curr_vals = {
            col: st.session_state.get(f"{col}_{row_id}", str(row.get(col, "")).strip() or "— Select —")
            for col in ["language", "sentiment", "text_type", "review_status"]
        }
        curr_vals["annotator_notes"] = st.session_state.get(f"notes_{row_id}", str(row.get("annotator_notes", "")))
        curr_dirty = any(
            curr_vals[col] != (str(row.get(col, "")).strip() or "— Select —")
            for col in ["language", "sentiment", "text_type", "review_status"]
        ) or (curr_vals["annotator_notes"].strip() != str(row.get("annotator_notes", "")).strip())

        if curr_dirty:
            issues = validation_issues(curr_vals)
            if not issues:
                current = get_data()
                save_ok = save_dataset_record(storage_path=OUT, current_df=current, row_id=row_id, vals=curr_vals)
                if save_ok:
                    st.session_state.idx = chosen_idx
                    st.rerun()
                else:
                    st.session_state.nav_err = f"Failed to auto-save item #{row_id} to {OUT.name} before jumping."
                    st.rerun()
            else:
                st.session_state.nav_err = f"Cannot jump to sentence: Item #{row_id} has unsaved invalid changes. Please resolve validation requirements or revert your edits before jumping."
                st.rerun()
        else:
            st.session_state.idx = chosen_idx
            st.rerun()

with selector_col2:
    st.progress((idx + 1) / total)
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94a3b8; margin-top: -0.3rem;">
        <span>Progress</span>
        <strong style="color: #38bdf8;">Sentence {idx + 1} of {total}</strong>
    </div>
    """, unsafe_allow_html=True)

# Main Two-Column Layout: Sentence Hero & Guidelines (Left) + Form Controls (Right)
left_col, right_col = st.columns([1.35, 1.15], gap="large")

with left_col:
    # Status Tag Determination
    cur_status = str(row.get("review_status", "")).strip()
    if cur_status == "Reviewed — looks correct":
        status_badge_html = '<span class="status-pill-reviewed">✓ Reviewed</span>'
    elif cur_status in ["Needs second review", "Ambiguous", "Skip"]:
        status_badge_html = f'<span class="status-pill-followup">⚠️ {cur_status}</span>'
    else:
        status_badge_html = '<span class="status-pill-pending">⏳ Unreviewed</span>'

    # Prominent Hero Sentence Card
    st.markdown(f"""
    <div class="sentence-hero">
        <div class="sentence-hero-meta">
            <div class="sentence-id-badge">
                <span>ITEM #{row_id}</span>
            </div>
            <div class="sentence-tags">
                <span class="sentence-tag">{word_count} words</span>
                <span class="sentence-tag">{char_count} chars</span>
                {status_badge_html}
            </div>
        </div>
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: #38bdf8; font-weight: 700;">
            Text to Annotate
        </div>
        <div class="sentence-text-box">
            {row_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Guidelines Card inside a clean Streamlit container
    with st.container(border=True):
        st.markdown("""
        <div style="font-weight: 700; font-size: 0.95rem; color: #f1f5f9; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;">
            <span>📖</span> Annotation Guidelines
        </div>
        <div class="guideline-box">
            <div class="guideline-item">
                <span class="guideline-dot">●</span>
                <span><strong>Language:</strong> Determine based on the sentence script and vocabulary as written.</span>
            </div>
            <div class="guideline-item">
                <span class="guideline-dot">●</span>
                <span><strong>Sentiment:</strong> Assign polarity only when clear; use <em>Not applicable</em> for neutral factual statements.</span>
            </div>
            <div class="guideline-item">
                <span class="guideline-dot">●</span>
                <span><strong>Text Type:</strong> Classify into Statement, Question, Request/Instruction, or Greeting.</span>
            </div>
            <div class="guideline-item">
                <span class="guideline-dot">●</span>
                <span><strong>Uncertainty:</strong> Use <em>Uncertain</em> or <em>Ambiguous</em> rather than guessing; an explanatory note is required.</span>
            </div>
            <div class="guideline-item">
                <span class="guideline-dot">●</span>
                <span><strong>Review Status:</strong> Must meet all label requirements before marking as <em>Reviewed</em>.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    # -------------------------------------------------------------
    # ML Advisory Assistance Card (Human-in-the-Loop)
    # -------------------------------------------------------------
    if enable_ml_suggestions and ml_engine.is_available():
        preds = ml_engine.predict_all(row_text)
        has_uncertain = any(p.is_uncertain for p in preds.values())

        with st.container(border=True):
            backend_label = "Transformer (XLM-R)" if ml_engine.active_backend == MLBackend.TRANSFORMER else "Baseline Heuristics"
            backend_sub = "Multilingual transformer & zero-shot intent predictions." if ml_engine.active_backend == MLBackend.TRANSFORMER else "Statistical & rule-based suggestions."

            st.markdown(f"""
            <div class="ml-card-header">
                <div class="ml-card-title">
                    <span>✨</span> Advisory ML Suggestions
                </div>
                <div style="display: flex; gap: 0.4rem; align-items: center;">
                    <span class="status-pill-clean" style="font-size: 0.68rem;">{backend_label}</span>
                    <div class="ml-advisory-badge">Advisory • Non-Committing</div>
                </div>
            </div>
            <div style="font-size: 0.76rem; color: #94a3b8; margin-bottom: 0.65rem;">
                {backend_sub} Clicking populates the form without saving to disk.
            </div>
            """, unsafe_allow_html=True)

            # Helper for displaying honest confidence and backend badges
            def get_pred_badges(p, conf_cls):
                c_lbl = "Heuristic" if p.confidence_type == "heuristic" else ("Calibrated" if p.confidence_type == "calibrated_probability" else "Model Prob")
                fb = '<span class="ml-badge-method" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border-color: rgba(245, 158, 11, 0.3);">Fallback</span>' if p.actual_backend == "fallback_baseline" else ""
                return f"""{fb}
                        <span class="ml-badge-conf {conf_cls}" title="Confidence type: {p.confidence_type}">{int(p.confidence * 100)}% ({c_lbl})</span>
                        <span class="ml-badge-method">{p.method}</span>"""

            # Language Row
            pred_lang = preds["language"]
            conf_class_lang = "ml-conf-high" if pred_lang.confidence >= 0.80 else ("ml-conf-med" if pred_lang.confidence >= 0.60 else "ml-conf-low")
            c_l1, c_l2 = st.columns([3, 1.2])
            with c_l1:
                st.markdown(f"""
                <div class="ml-suggestion-row">
                    <div class="ml-suggestion-left">
                        <span class="ml-suggestion-field">Language:</span>
                        <span class="ml-suggestion-val">{pred_lang.label}</span>
                    </div>
                    <div class="ml-badges">
                        {get_pred_badges(pred_lang, conf_class_lang)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_l2:
                if st.button("Use", key=f"btn_use_lang_{row_id}", use_container_width=True, help=f"Use suggestion '{pred_lang.label}'"):
                    if pred_lang.label in OPTIONS["language"]:
                        st.session_state[f"language_{row_id}"] = pred_lang.label
                        st.rerun()

            # Sentiment Row
            pred_sent = preds["sentiment"]
            conf_class_sent = "ml-conf-high" if pred_sent.confidence >= 0.80 else ("ml-conf-med" if pred_sent.confidence >= 0.60 else "ml-conf-low")
            c_s1, c_s2 = st.columns([3, 1.2])
            with c_s1:
                st.markdown(f"""
                <div class="ml-suggestion-row">
                    <div class="ml-suggestion-left">
                        <span class="ml-suggestion-field">Sentiment:</span>
                        <span class="ml-suggestion-val">{pred_sent.label}</span>
                    </div>
                    <div class="ml-badges">
                        {get_pred_badges(pred_sent, conf_class_sent)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_s2:
                if st.button("Use", key=f"btn_use_sent_{row_id}", use_container_width=True, help=f"Use suggestion '{pred_sent.label}'"):
                    if pred_sent.label in OPTIONS["sentiment"]:
                        st.session_state[f"sentiment_{row_id}"] = pred_sent.label
                        st.rerun()

            # Text Type Row
            pred_tt = preds["text_type"]
            conf_class_tt = "ml-conf-high" if pred_tt.confidence >= 0.80 else ("ml-conf-med" if pred_tt.confidence >= 0.60 else "ml-conf-low")
            c_t1, c_t2 = st.columns([3, 1.2])
            with c_t1:
                st.markdown(f"""
                <div class="ml-suggestion-row">
                    <div class="ml-suggestion-left">
                        <span class="ml-suggestion-field">Text Type:</span>
                        <span class="ml-suggestion-val">{pred_tt.label}</span>
                    </div>
                    <div class="ml-badges">
                        {get_pred_badges(pred_tt, conf_class_tt)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_t2:
                if st.button("Use", key=f"btn_use_tt_{row_id}", use_container_width=True, help=f"Use suggestion '{pred_tt.label}'"):
                    if pred_tt.label in OPTIONS["text_type"]:
                        st.session_state[f"text_type_{row_id}"] = pred_tt.label
                        st.rerun()

            if has_uncertain:
                st.markdown("""
                <div class="ml-uncertain-alert">
                    <span>⚠️</span>
                    <span>Low confidence prediction detected. Annotator verification and rationale note required if used.</span>
                </div>
                """, unsafe_allow_html=True)

            if st.button("⚡ Accept All Suggestions", key=f"btn_accept_all_{row_id}", use_container_width=True, help="Populate Language, Sentiment, and Text Type fields with suggestions."):
                if pred_lang.label in OPTIONS["language"]:
                    st.session_state[f"language_{row_id}"] = pred_lang.label
                if pred_sent.label in OPTIONS["sentiment"]:
                    st.session_state[f"sentiment_{row_id}"] = pred_sent.label
                if pred_tt.label in OPTIONS["text_type"]:
                    st.session_state[f"text_type_{row_id}"] = pred_tt.label
                st.rerun()

    # Form Controls Card
    with st.container(border=True):
        st.markdown(f"""
        <div style="font-weight: 700; font-size: 1.05rem; color: #ffffff; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>🏷️</span> Annotation Form
        </div>
        <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 0.75rem;">
            Select labels for item #{row_id}. Required fields marked with <span style="color: #38bdf8;">*</span>
        </div>
        """, unsafe_allow_html=True)

        vals = {}
        field_labels = [
            ("language", "Language *"),
            ("sentiment", "Sentiment *"),
            ("text_type", "Text Type *"),
            ("review_status", "Quality Review Status *"),
        ]

        for col, label in field_labels:
            options = OPTIONS[col]
            saved_val = str(row.get(col, "")).strip()
            idx_in_options = options.index(saved_val) if saved_val in options else 0
            vals[col] = st.selectbox(
                label,
                options,
                index=idx_in_options,
                key=f"{col}_{row_id}",
            )

        vals["annotator_notes"] = st.text_area(
            "Annotator Notes",
            value=str(row.get("annotator_notes", "")),
            key=f"notes_{row_id}",
            placeholder="Required when flagging uncertain, ambiguous, or skipped items...",
            help="Provide rationale when selecting Uncertain, Ambiguous, Needs second review, or Skip.",
        )

# -------------------------------------------------------------
# Save Functionality & Live Validation Feedback
# -------------------------------------------------------------
is_dirty = any(
    vals[c] != (str(row.get(c, "")).strip() or "— Select —")
    for c in ["language", "sentiment", "text_type", "review_status"]
) or (
    vals["annotator_notes"].strip() != str(row.get("annotator_notes", "")).strip()
)

def save(show_errors=True):
    issues = validation_issues(vals)
    if issues:
        if show_errors:
            st.error("Please resolve the validation issues before saving:")
            for issue in issues:
                st.markdown(f"- {issue}")
        return False
    current = get_data()
    save_ok = save_dataset_record(storage_path=OUT, current_df=current, row_id=row_id, vals=vals)
    if save_ok:
        st.session_state.msg = f"Saved annotation for item {row_id} to {OUT.name}."
        return True
    else:
        st.error(f"Failed to write annotation to storage path: {OUT}. Please verify filesystem permissions.")
        return False

# Display any banner messages from previous action
if "msg" in st.session_state:
    st.success(st.session_state.pop("msg"))

if "nav_err" in st.session_state:
    st.warning(st.session_state.pop("nav_err"))

# Live Validation Feedback Banner
issues_now = validation_issues(vals)
if issues_now:
    issues_html = "".join([f'<div class="validation-issue-item"><span class="validation-issue-bullet">●</span><span>{issue}</span></div>' for issue in issues_now])
    st.markdown(f"""
    <div class="validation-alert">
        <div class="validation-alert-header">
            <span>⚠️</span> Validation Checklist ({len(issues_now)} remaining requirements)
        </div>
        {issues_html}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="validation-pass">
        <span class="validation-pass-icon">✓</span>
        <div>
            <span class="validation-pass-text">Validation Passed</span>
            <span class="validation-pass-sub">• All quality requirements met. Ready to save or submit.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Action Buttons Bar
# -------------------------------------------------------------
btn_p, btn_n, btn_save, btn_unrev = st.columns([1, 1, 1.4, 1.4])

with btn_p:
    if st.button("← Previous", disabled=(idx == 0), use_container_width=True):
        if is_dirty:
            if not validation_issues(vals):
                save(show_errors=False)
                st.session_state.idx = max(0, idx - 1)
                st.rerun()
            else:
                st.session_state.nav_err = f"Cannot navigate to Previous: Item #{row_id} has unsaved invalid changes. Please resolve validation requirements or revert changes before moving."
                st.rerun()
        else:
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

with btn_n:
    if st.button("Next →", disabled=(idx == total - 1), use_container_width=True):
        if is_dirty:
            if not validation_issues(vals):
                save(show_errors=False)
                st.session_state.idx = min(total - 1, idx + 1)
                st.rerun()
            else:
                st.session_state.nav_err = f"Cannot navigate to Next: Item #{row_id} has unsaved invalid changes. Please resolve validation requirements or revert changes before moving."
                st.rerun()
        else:
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()

with btn_save:
    if st.button("💾 Save Annotation", type="primary", use_container_width=True):
        if save(show_errors=True):
            st.rerun()

with btn_unrev:
    if st.button("⏩ Go to Next Unreviewed", use_container_width=True):
        if is_dirty:
            if not validation_issues(vals):
                save(show_errors=False)
            else:
                st.session_state.nav_err = f"Cannot navigate to Next Unreviewed: Item #{row_id} has unsaved invalid changes. Please resolve validation requirements or revert changes before moving."
                st.rerun()
        updated = get_data()
        pending = updated.index[updated.review_status.astype(str).str.strip().eq("")].tolist()
        candidates = [i for i in pending if i > idx] or [i for i in pending if i <= idx]
        if candidates:
            st.session_state.idx = candidates[0]
            st.rerun()
        else:
            st.success("🎉 Every sentence has a review status! Check the Quality Dashboard.")

# Bottom Export Link & Storage Details
st.markdown("<div style='margin-top: 1.25rem;'></div>", unsafe_allow_html=True)
foot_col1, foot_col2 = st.columns([2, 1])

with foot_col1:
    st.markdown(f"<div style='font-size: 0.78rem; color: #64748b; padding-top: 0.5rem;'>Annotations automatically persist to <code>{OUT.name}</code> with UTF-8 encoding.</div>", unsafe_allow_html=True)

with foot_col2:
    if OUT.exists():
        st.download_button(
            "📥 Download Annotated CSV",
            OUT.read_bytes(),
            download_annotated_name,
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("Save an annotation to create the output CSV.")

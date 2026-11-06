"""
LinguaLabel — Home Page (Part 10)

Beginner-friendly welcome page introducing LinguaLabel in plain language:
- Explains the 3-step workflow (Upload -> AI Analyse -> Review & Export).
- Displays real-time summary statistics across loaded and saved datasets.
- Lists recent datasets with one-click access.
- Clear call-to-action button to start a new analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st


def scan_recent_datasets(root_dir: Path) -> List[Dict[str, Any]]:
    """Discovers available datasets in root and saved_sessions."""
    datasets = []
    
    # 1. Default Sample & Annotated output
    default_annot = root_dir / "annotated_output.csv"
    sample_csv = root_dir / "sample_data.csv"
    if default_annot.exists():
        try:
            df = pd.read_csv(default_annot, dtype=str)
            rev_c = int(df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct").sum()) if "review_status" in df.columns else 0
            mtime = datetime.fromtimestamp(default_annot.stat().st_mtime).strftime("%b %d, %Y %H:%M")
            datasets.append({
                "name": "Default Sample (20 Sentences)",
                "key": "default_sample",
                "path": default_annot,
                "rows": len(df),
                "reviewed": rev_c,
                "modified": mtime,
                "is_completed": (rev_c == len(df) and len(df) > 0),
            })
        except Exception:
            pass
    elif sample_csv.exists():
        try:
            df = pd.read_csv(sample_csv, dtype=str)
            mtime = datetime.fromtimestamp(sample_csv.stat().st_mtime).strftime("%b %d, %Y %H:%M")
            datasets.append({
                "name": "Default Sample (20 Sentences)",
                "key": "default_sample",
                "path": sample_csv,
                "rows": len(df),
                "reviewed": 0,
                "modified": mtime,
                "is_completed": False,
            })
        except Exception:
            pass

    # 2. Saved Sessions
    sessions_dir = root_dir / "saved_sessions"
    if sessions_dir.exists():
        for f in sorted(sessions_dir.glob("annotated_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                df = pd.read_csv(f, dtype=str)
                k = f.stem.replace("annotated_", "")
                rev_c = int(df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct").sum()) if "review_status" in df.columns else 0
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%b %d, %Y %H:%M")
                clean_name = f.stem.replace("annotated_", "").replace("_", " ").capitalize()
                datasets.append({
                    "name": clean_name,
                    "key": k,
                    "path": f,
                    "rows": len(df),
                    "reviewed": rev_c,
                    "modified": mtime,
                    "is_completed": (rev_c == len(df) and len(df) > 0),
                })
            except Exception:
                continue

    return datasets


def render_home_page(root_dir: Path, active_df: Optional[pd.DataFrame] = None):
    """Renders the clean, welcoming Home Page."""
    
    # Hero Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 2rem 2.2rem; margin-bottom: 1.5rem;">
        <div style="display: inline-block; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 20px; padding: 4px 14px; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.75rem;">
            Multilingual Text Analysis & Human Annotation
        </div>
        <h1 style="font-size: 2.3rem; font-weight: 800; color: #ffffff; margin: 0 0 0.5rem 0; letter-spacing: -0.02em; line-height: 1.2;">
            Welcome to LinguaLabel
        </h1>
        <p style="font-size: 1.1rem; color: #cbd5e1; max-width: 780px; margin: 0 0 1.25rem 0; line-height: 1.5;">
            Understand, organize, and label sentences in multiple languages with help from AI.
            <br>
            <span style="font-size: 0.95rem; color: #94a3b8;">
                AI does the initial work. You make the final decisions.
            </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Compute genuine real-world statistics
    recent_datasets = scan_recent_datasets(root_dir)
    total_processed = sum(d["rows"] for d in recent_datasets)
    total_reviewed = sum(d["reviewed"] for d in recent_datasets)
    total_pending = max(0, total_processed - total_reviewed)
    completed_datasets = sum(1 for d in recent_datasets if d["is_completed"])

    # If an active dataframe is loaded in session, show its immediate stats
    if active_df is not None and not active_df.empty:
        curr_total = len(active_df)
        curr_rev = int(active_df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct").sum()) if "review_status" in active_df.columns else 0
        curr_pend = curr_total - curr_rev
    else:
        curr_total, curr_rev, curr_pend = total_processed, total_reviewed, total_pending

    # Overview Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 1rem 1.1rem;">
            <div style="font-size: 0.74rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Total Sentences</div>
            <div style="font-size: 1.75rem; font-weight: 800; color: #f8fafc; margin-top: 0.2rem;">{curr_total}</div>
            <div style="font-size: 0.72rem; color: #64748b;">In active session</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 1rem 1.1rem;">
            <div style="font-size: 0.74rem; color: #4ade80; text-transform: uppercase; font-weight: 600;">Reviewed & Approved</div>
            <div style="font-size: 1.75rem; font-weight: 800; color: #4ade80; margin-top: 0.2rem;">{curr_rev}</div>
            <div style="font-size: 0.72rem; color: #64748b;">Verified by human annotator</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 8px; padding: 1rem 1.1rem;">
            <div style="font-size: 0.74rem; color: #fbbf24; text-transform: uppercase; font-weight: 600;">Waiting for Review</div>
            <div style="font-size: 1.75rem; font-weight: 800; color: #fbbf24; margin-top: 0.2rem;">{curr_pend}</div>
            <div style="font-size: 0.72rem; color: #64748b;">Pending or incomplete</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px; padding: 1rem 1.1rem;">
            <div style="font-size: 0.74rem; color: #c084fc; text-transform: uppercase; font-weight: 600;">Completed Datasets</div>
            <div style="font-size: 1.75rem; font-weight: 800; color: #c084fc; margin-top: 0.2rem;">{completed_datasets}</div>
            <div style="font-size: 0.72rem; color: #64748b;">100% verified datasets</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)

    # 3-Step Simple Explanation Card
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 1rem;">
            How LinguaLabel Works in 3 Simple Steps
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.25rem;">
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem 1.2rem; border-radius: 8px; border-left: 3px solid #38bdf8;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-bottom: 0.3rem;">1. Upload Your Sentences</div>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                    Upload a CSV or Excel spreadsheet, paste sentences directly, or enter a single sentence to analyze.
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem 1.2rem; border-radius: 8px; border-left: 3px solid #fbbf24;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #fbbf24; margin-bottom: 0.3rem;">2. Let AI Analyse Them</div>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                    AI automatically suggests the language (Hindi, English, Hinglish), sentiment (positive, negative, neutral), and sentence type.
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); padding: 1rem 1.2rem; border-radius: 8px; border-left: 3px solid #4ade80;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #4ade80; margin-bottom: 0.3rem;">3. Review & Export</div>
                <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">
                    Accept suggestions with one click, easily correct difficult ones, and download clean results as CSV or Excel.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Actions & Recent Datasets Row
    c_act1, c_act2 = st.columns([1.5, 2.5])
    with c_act1:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 1.25rem; height: 100%;">
            <div style="font-size: 1rem; font-weight: 700; color: #ffffff; margin-bottom: 0.5rem;">
                🚀 Ready to start?
            </div>
            <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 1.2rem; line-height: 1.4;">
                Import your own text data or explore the pre-loaded 20 multilingual example sentences.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📥 Import New Sentences", type="primary", width="stretch", key="home_btn_import"):
            st.query_params["view"] = "Import Sentences"
            st.rerun()

        if st.button("✏️ Continue Labelling Active Data", width="stretch", key="home_btn_label"):
            st.query_params["view"] = "Analyse & Label"
            st.rerun()

    with c_act2:
        st.markdown("#### 📂 Available Datasets & Recent Sessions")
        if not recent_datasets:
            st.info("No saved datasets found yet. Click 'Import New Sentences' to load your data.")
        else:
            recent_rows = []
            for d in recent_datasets[:6]:
                recent_rows.append({
                    "Dataset Name": d["name"],
                    "Sentences": d["rows"],
                    "Reviewed": f"{d['reviewed']}/{d['rows']}",
                    "Last Modified": d["modified"],
                    "Status": "✅ Complete" if d["is_completed"] else "⏳ In Progress",
                })
            st.dataframe(pd.DataFrame(recent_rows), width="stretch")


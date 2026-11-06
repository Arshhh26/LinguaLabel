"""
LinguaLabel — My Results Page (Part 10)

Clear, beginner-friendly results dashboard:
- Summary cards: Total, Reviewed vs Pending, Sentiment counts.
- Intuitive dark-mode visual charts for Sentiment, Language, and Sentence Types.
- Interactive filtered table with distinct visual indicators for Human Verified ✓ vs Pending ⏳.
- Full filter bar by Language, Sentiment, Sentence Type, and Review Status.
"""

from pathlib import Path
from typing import Optional
import altair as alt
import pandas as pd
import streamlit as st

from ui_mappings import to_display_label


def render_my_results_page(root_dir: Path, df: pd.DataFrame, dataset_key: str):
    """Renders the simplified My Results dashboard."""
    st.markdown("""
    <div class="header-container" style="margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
            <span class="header-badge" style="background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3);">
                📊 DATASET RESULTS
            </span>
        </div>
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.02em;">
            My Results & Overview
        </h1>
        <div class="header-subtitle" style="color: #94a3b8; font-size: 0.9rem;">
            Explore summary statistics, language distributions, and verified annotations for the active dataset.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("No active dataset loaded. Please import sentences first.")
        return

    total = len(df)
    rev_mask = df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct")
    reviewed_count = int(rev_mask.sum())
    pending_count = total - reviewed_count

    # Sentiment distribution
    sent_series = df["sentiment"].astype(str).str.strip().replace("", "Unlabelled")
    pos_count = int((sent_series == "Positive").sum())
    neg_count = int((sent_series == "Negative").sum())
    neu_count = int((sent_series == "Neutral").sum())
    unc_count = int((sent_series == "Uncertain").sum())

    # Summary Cards Row
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    with rc1:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.85rem 1rem;">
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Total Sentences</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; margin-top: 0.2rem;">{total}</div>
            <div style="font-size: 0.7rem; color: #64748b;">Active session</div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 8px; padding: 0.85rem 1rem;">
            <div style="font-size: 0.72rem; color: #4ade80; text-transform: uppercase; font-weight: 600;">Human Verified</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #4ade80; margin-top: 0.2rem;">{reviewed_count}</div>
            <div style="font-size: 0.7rem; color: #64748b;">{int(reviewed_count/total*100) if total else 0}% complete</div>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 0.85rem 1rem;">
            <div style="font-size: 0.72rem; color: #38bdf8; text-transform: uppercase; font-weight: 600;">Positive Sentiment</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8; margin-top: 0.2rem;">{pos_count}</div>
            <div style="font-size: 0.7rem; color: #64748b;">Favorable sentences</div>
        </div>
        """, unsafe_allow_html=True)
    with rc4:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 0.85rem 1rem;">
            <div style="font-size: 0.72rem; color: #f87171; text-transform: uppercase; font-weight: 600;">Negative Sentiment</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #f87171; margin-top: 0.2rem;">{neg_count}</div>
            <div style="font-size: 0.7rem; color: #64748b;">Critical or complaints</div>
        </div>
        """, unsafe_allow_html=True)
    with rc5:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 8px; padding: 0.85rem 1rem;">
            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Neutral / Informative</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #cbd5e1; margin-top: 0.2rem;">{neu_count}</div>
            <div style="font-size: 0.7rem; color: #64748b;">Factual sentences</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Visual Charts: 2 Columns (Sentiment & Language)
    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        st.markdown("#### Sentiment Distribution")
        sent_counts = df["sentiment"].astype(str).str.strip().replace("", "Unlabelled").value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        
        sent_chart = alt.Chart(sent_counts).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("Sentiment:N", sort="-x", axis=alt.Axis(title=None, labelColor="#cbd5e1")),
            x=alt.X("Count:Q", axis=alt.Axis(title="Number of Sentences", labelColor="#94a3b8", gridColor="rgba(255,255,255,0.05)")),
            color=alt.Color("Sentiment:N", scale=alt.Scale(
                domain=["Positive", "Negative", "Neutral", "Uncertain", "Not applicable", "Unlabelled"],
                range=["#4ade80", "#f87171", "#94a3b8", "#fbbf24", "#818cf8", "#475569"]
            ), legend=None),
            tooltip=["Sentiment", "Count"]
        ).properties(height=200).configure_view(strokeOpacity=0)
        st.altair_chart(sent_chart, use_container_width=True)

    with chart_c2:
        st.markdown("#### Language Breakdown")
        lang_counts = df["language"].astype(str).str.strip().replace("", "Unlabelled").value_counts().reset_index()
        lang_counts.columns = ["Language", "Count"]
        
        lang_chart = alt.Chart(lang_counts).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y("Language:N", sort="-x", axis=alt.Axis(title=None, labelColor="#cbd5e1")),
            x=alt.X("Count:Q", axis=alt.Axis(title="Number of Sentences", labelColor="#94a3b8", gridColor="rgba(255,255,255,0.05)")),
            color=alt.Color("Language:N", scale=alt.Scale(
                domain=["English", "Hindi", "Other", "Uncertain", "Unlabelled"],
                range=["#38bdf8", "#fbbf24", "#a855f7", "#f43f5e", "#475569"]
            ), legend=None),
            tooltip=["Language", "Count"]
        ).properties(height=200).configure_view(strokeOpacity=0)
        st.altair_chart(lang_chart, use_container_width=True)

    st.markdown("---")

    # Interactive Table with Filters
    st.markdown("### 📋 Browse Sentence Annotations")
    st.caption("Human-approved annotations carry a green checkmark (✓). Sentences still waiting for review are marked with (⏳).")

    filt_c1, filt_c2, filt_c3, filt_c4 = st.columns(4)
    with filt_c1:
        f_lang = st.selectbox("Filter Language", ["All"] + sorted(list(set(df["language"].dropna().unique()))))
    with filt_c2:
        f_sent = st.selectbox("Filter Sentiment", ["All"] + sorted(list(set(df["sentiment"].dropna().unique()))))
    with filt_c3:
        f_tt = st.selectbox("Filter Sentence Type", ["All"] + sorted(list(set(df["text_type"].dropna().unique()))))
    with filt_c4:
        f_status = st.selectbox("Filter Review Status", ["All", "Reviewed & approved", "Pending / Incomplete"])

    # Apply filters
    filtered_df = df.copy()
    if f_lang != "All":
        filtered_df = filtered_df[filtered_df["language"] == f_lang]
    if f_sent != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == f_sent]
    if f_tt != "All":
        filtered_df = filtered_df[filtered_df["text_type"] == f_tt]
    if f_status == "Reviewed & approved":
        filtered_df = filtered_df[filtered_df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct")]
    elif f_status == "Pending / Incomplete":
        filtered_df = filtered_df[~filtered_df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct")]

    # Format table for friendly display
    display_rows = []
    for idx_val, row in filtered_df.iterrows():
        is_rev = str(row.get("review_status", "")).strip() == "Reviewed — looks correct"
        status_icon = "✅ Human Verified" if is_rev else "⏳ Pending Review"
        display_rows.append({
            "Sentence #": idx_val + 1,
            "Sentence Text": str(row.get("text", "")),
            "Language": to_display_label("language", str(row.get("language", ""))),
            "Sentiment": to_display_label("sentiment", str(row.get("sentiment", ""))),
            "Sentence Type": to_display_label("text_type", str(row.get("text_type", ""))),
            "Verification Status": status_icon,
            "Notes": str(row.get("annotator_notes", "")),
        })

    st.caption(f"Showing **{len(display_rows)}** matching sentence(s).")
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True)


"""
LinguaLabel — Export Data Page (Part 10)

Beginner-friendly data export interface supporting:
- Clean CSV download (UTF-8-SIG encoding ensuring flawless Devanagari Hindi and Excel rendering).
- Clean Excel workbook download (.xlsx) via openpyxl.
- Choice between Beginner-Friendly Export (clean core columns) and Advanced / Audit Export.
- Live export record count and preview.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from data_manager import export_excel_bytes
from ui_mappings import to_display_label


def render_export_page(root_dir: Path, df: pd.DataFrame, dataset_key: str):
    """Renders the Export Data page."""
    st.markdown("""
    <div class="header-container" style="margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
            <span class="header-badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3);">
                📤 EXPORT & DOWNLOAD
            </span>
        </div>
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.02em;">
            Export Your Annotations
        </h1>
        <div class="header-subtitle" style="color: #94a3b8; font-size: 0.9rem;">
            Download your labelled data as a clean CSV spreadsheet or Excel workbook.
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

    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <strong style="color: #f8fafc; font-size: 1.05rem;">Active Dataset:</strong>
                <span style="color: #38bdf8; font-weight: 600;"> {dataset_key}</span>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">
                Total sentences: <strong style="color: #f8fafc;">{total}</strong> | Human verified: <strong style="color: #4ade80;">{reviewed_count}</strong> | Pending review: <strong style="color: #fbbf24;">{pending_count}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Export Format Selection
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        export_mode = st.radio(
            "Choose Export Columns:",
            [
                "🟢 Beginner-Friendly Export (Clean columns for presentations, reports, and spreadsheets)",
                "🔬 Advanced / Complete Export (Includes record IDs, notes, and internal technical schema)",
            ],
            index=0,
        )
    with exp_c2:
        export_scope = st.radio(
            "Sentences to Include:",
            [
                "All Sentences (both verified and pending)",
                "Human Verified Only (sentences marked as 'Reviewed & approved')",
            ],
            index=0,
        )

    # Filter export rows
    target_df = df.copy()
    if export_scope == "Human Verified Only (sentences marked as 'Reviewed & approved')":
        target_df = target_df[rev_mask]

    if target_df.empty:
        st.warning("No sentences match the selected scope. Please adjust your filter or review more sentences.")
        return

    # Build clean export dataframe
    if "Beginner-Friendly" in export_mode:
        clean_rows = []
        for _, row in target_df.iterrows():
            is_rev = str(row.get("review_status", "")).strip() == "Reviewed — looks correct"
            clean_rows.append({
                "Sentence": str(row.get("text", "")),
                "Language": to_display_label("language", str(row.get("language", ""))),
                "Sentiment": to_display_label("sentiment", str(row.get("sentiment", ""))),
                "Sentence Type": to_display_label("text_type", str(row.get("text_type", ""))),
                "Human Verified": "Yes" if is_rev else "Pending",
                "Review Notes": str(row.get("annotator_notes", "")),
            })
        final_export_df = pd.DataFrame(clean_rows)
    else:
        final_export_df = target_df.copy()

    st.markdown("#### Preview Data to be Exported")
    st.caption(f"Ready to export **{len(final_export_df)}** sentence(s) with {len(final_export_df.columns)} columns.")
    st.dataframe(final_export_df.head(5), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Download Files")

    def cb_track_csv():
        try:
            from db.analytics import record_usage_event
            v_id = st.session_state.get("_visitor_id")
            s_id = st.session_state.get("_session_id")
            record_usage_event(
                "export",
                "export_csv",
                count=1,
                metadata={"format": "csv", "row_count": len(final_export_df)},
                visitor_id=v_id,
                session_id=s_id,
            )
        except Exception:
            pass

    def cb_track_excel():
        try:
            from db.analytics import record_usage_event
            v_id = st.session_state.get("_visitor_id")
            s_id = st.session_state.get("_session_id")
            record_usage_event(
                "export",
                "export_excel",
                count=1,
                metadata={"format": "xlsx", "row_count": len(final_export_df)},
                visitor_id=v_id,
                session_id=s_id,
            )
        except Exception:
            pass

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        csv_bytes = final_export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 Download as CSV (.csv)",
            data=csv_bytes,
            file_name=f"lingualabel_{dataset_key}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            on_click=cb_track_csv,
            help="Universal CSV format with UTF-8-SIG encoding. Opens cleanly in Microsoft Excel, Google Sheets, and Numbers without breaking Hindi text."
        )

    with d_col2:
        excel_bytes = export_excel_bytes(final_export_df)
        if excel_bytes:
            st.download_button(
                label="📊 Download as Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"lingualabel_{dataset_key}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True,
                on_click=cb_track_excel,
                help="Native Microsoft Excel workbook format with styled column headers."
            )
        else:
            st.button("📊 Excel Export Unavailable", disabled=True, use_container_width=True)


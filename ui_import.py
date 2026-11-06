"""
LinguaLabel — Import Sentences Page (Part 10 & 11)

Beginner-friendly data ingestion supporting:
A. Upload a CSV or Excel spreadsheet (.csv, .xlsx, .xls)
B. Paste multiple sentences (one per line)
C. Enter a single sentence manually to test immediately
D. Pre-loaded multilingual example sentences

Multi-User Integration:
- Allows selecting destination workspace.
- Stores dataset in the database under user ownership.
- Maintains local file persistence for backward compatibility.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import streamlit as st

from data_manager import (
    compute_dataset_key,
    detect_text_column,
    get_dataset_storage_path,
    load_csv_bytes,
    load_excel_bytes,
    load_pasted_sentences,
    load_single_sentence,
    merge_saved_annotations,
    stabilize_dataset,
)


def _save_to_user_db(
    current_user: Optional[Dict[str, Any]],
    target_ws_id: Optional[str],
    ds_name: str,
    stable_df: pd.DataFrame,
    orig_filename: Optional[str] = None,
    dataset_key: Optional[str] = None,
) -> None:
    """Helper to save imported records into the user's active workspace."""
    if not current_user or not target_ws_id or stable_df.empty:
        return
    try:
        from db.service import create_dataset_with_sentences
        records = [dict(r) for _, r in stable_df.iterrows()]
        created_ds, _ = create_dataset_with_sentences(
            user_id=current_user["id"],
            workspace_id=target_ws_id,
            name=ds_name,
            records=records,
            original_filename=orig_filename,
            dataset_key=dataset_key,
        )
        if created_ds:
            st.session_state.active_dataset_id = created_ds.id
    except Exception:
        pass


def render_import_page(root_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[Path]]:
    """Renders the Import Sentences interface."""
    st.markdown("""
    <div class="header-container" style="margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
            <span class="header-badge" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);">
                📥 DATA IMPORT
            </span>
        </div>
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.02em;">
            Import Sentences to Analyse
        </h1>
        <div class="header-subtitle" style="color: #94a3b8; font-size: 0.9rem;">
            Bring your sentences into LinguaLabel by uploading a spreadsheet, pasting text, or typing a sentence.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Multi-User Workspace Selector (if user is authenticated)
    current_user = st.session_state.get("current_user")
    target_ws_id: Optional[str] = None

    if current_user:
        from db.service import list_workspaces
        workspaces = list_workspaces(current_user["id"])
        if workspaces:
            ws_options = {w.name: w.id for w in workspaces}
            curr_active_ws = st.session_state.get("active_workspace_id", workspaces[0].id)
            default_ws_name = next((name for name, wid in ws_options.items() if wid == curr_active_ws), list(ws_options.keys())[0])

            c_ws_lbl, c_ws_sel = st.columns([1.5, 3.5])
            with c_ws_lbl:
                st.markdown("<div style='margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;'>Save into Workspace:</div>", unsafe_allow_html=True)
            with c_ws_sel:
                selected_ws_title = st.selectbox("Workspace", list(ws_options.keys()), index=list(ws_options.keys()).index(default_ws_name), label_visibility="collapsed")
                target_ws_id = ws_options.get(selected_ws_title)
                st.session_state.active_workspace_id = target_ws_id
            st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 0.75rem 0;'>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Choose how you want to add sentences:",
        [
            "📄 Upload a Spreadsheet (CSV or Excel)",
            "📋 Paste multiple sentences (one per line)",
            "✍️ Enter a single sentence manually",
            "✨ Use pre-loaded Example Sentences (20 multilingual samples)",
        ],
        horizontal=False,
    )

    imported_df: Optional[pd.DataFrame] = None
    dataset_key: Optional[str] = None
    storage_path: Optional[Path] = None

    # =========================================================================
    # OPTION A: UPLOAD SPREADSHEET (CSV or Excel)
    # =========================================================================
    if input_mode == "📄 Upload a Spreadsheet (CSV or Excel)":
        st.markdown("#### Upload your file")
        st.caption("Supports CSV (.csv) and Excel (.xlsx, .xls) files. Your data is isolated to your personal account.")

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name.lower()

            raw_df: Optional[pd.DataFrame] = None
            err_msg: Optional[str] = None

            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                raw_df, _, err_msg = load_excel_bytes(file_bytes, uploaded_file.name)
            else:
                raw_df, _, err_msg = load_csv_bytes(file_bytes, uploaded_file.name)

            if err_msg:
                st.error(f"Could not read the file: {err_msg}")
            elif raw_df is not None and not raw_df.empty:
                st.success(f"✓ Loaded '{uploaded_file.name}' with {len(raw_df)} rows and {len(raw_df.columns)} columns.")

                col_options = list(raw_df.columns)
                default_col = detect_text_column(raw_df)
                default_idx = col_options.index(default_col) if default_col in col_options else 0

                st.markdown("---")
                c_c1, c_c2 = st.columns(2)
                with c_c1:
                    selected_col = st.selectbox(
                        "1. Sentence Column",
                        col_options,
                        index=default_idx,
                        help="Select the column with the sentence text."
                    )
                with c_c2:
                    clean_ds_title = uploaded_file.name.rsplit(".", 1)[0].replace("_", " ").title()
                    ds_name_input = st.text_input("Dataset Display Name", value=clean_ds_title)

                st.markdown("#### 2. Preview your data")
                sample_preview = raw_df[[selected_col]].head(5).copy()
                sample_preview.columns = ["Sentence Preview (First 5 Rows)"]
                st.dataframe(sample_preview, use_container_width=True)

                d_key = compute_dataset_key(file_bytes)
                s_path = get_dataset_storage_path(d_key, root_dir)

                st.markdown("---")
                if st.button("🚀 Start Analysing This Dataset", type="primary", use_container_width=True):
                    with st.spinner("Preparing dataset and initializing session..."):
                        stable_df, _ = stabilize_dataset(raw_df, selected_col, d_key)
                        if s_path.exists():
                            stable_df = merge_saved_annotations(stable_df, s_path)
                        else:
                            stable_df.to_csv(s_path, index=False, encoding="utf-8-sig")

                        _save_to_user_db(current_user, target_ws_id, ds_name_input or clean_ds_title, stable_df, uploaded_file.name, d_key)

                        # Record anonymous aggregate usage event
                        try:
                            from db.analytics import record_usage_event
                            v_id = st.session_state.get("_visitor_id")
                            s_id = st.session_state.get("_session_id")
                            is_excel = filename.endswith(".xlsx") or filename.endswith(".xls")
                            fmt_label = "xlsx" if is_excel else "csv"
                            act_label = "excel_import" if is_excel else "csv_import"
                            record_usage_event("import", act_label, count=1, metadata={"format": fmt_label, "row_count": len(stable_df)}, visitor_id=v_id, session_id=s_id)
                            record_usage_event("import", "dataset_upload", count=1, metadata={"format": fmt_label, "row_count": len(stable_df)}, visitor_id=v_id, session_id=s_id)
                        except Exception:
                            pass

                        st.session_state.active_df = stable_df
                        st.session_state.active_dataset_key = ds_name_input or d_key
                        st.session_state.active_storage_path = s_path
                        st.session_state.idx = 0
                        st.session_state.main_nav_radio = "Analyse & Label"
                        st.rerun()

    # =========================================================================
    # OPTION B: PASTE MULTIPLE SENTENCES
    # =========================================================================
    elif input_mode == "📋 Paste multiple sentences (one per line)":
        st.markdown("#### Paste your sentences below")
        st.caption("Paste any text with one sentence per line. Hindi, English, and Hinglish are fully supported.")

        c_p1, c_p2 = st.columns([3, 1.5])
        with c_p2:
            p_name = st.text_input("Dataset Name", value="Pasted Sentences Batch")

        pasted_text = st.text_area(
            "Sentences Text Area",
            placeholder="Type or paste sentences here...\nSentence 1: I really like this app.\nSentence 2: यह बहुत अच्छा है।\nSentence 3: Ye phone thik nahi chal raha hai.",
            height=200,
            label_visibility="collapsed",
        )

        if pasted_text.strip():
            raw_df, err = load_pasted_sentences(pasted_text)
            if err:
                st.warning(err)
            elif raw_df is not None:
                st.info(f"Found **{len(raw_df)}** valid sentences.")
                st.dataframe(raw_df.head(5), use_container_width=True)

                if st.button("🚀 Import and Start Analysing", type="primary", use_container_width=True):
                    d_key = compute_dataset_key(pasted_text.encode("utf-8"))
                    s_path = get_dataset_storage_path(d_key, root_dir)
                    stable_df, _ = stabilize_dataset(raw_df, "text", d_key)
                    if s_path.exists():
                        stable_df = merge_saved_annotations(stable_df, s_path)
                    else:
                        stable_df.to_csv(s_path, index=False, encoding="utf-8-sig")

                    _save_to_user_db(current_user, target_ws_id, p_name or "Pasted Batch", stable_df, "pasted_text.txt", d_key)

                    try:
                        from db.analytics import record_usage_event
                        v_id = st.session_state.get("_visitor_id")
                        s_id = st.session_state.get("_session_id")
                        record_usage_event("import", "paste_import", count=1, metadata={"format": "text", "row_count": len(stable_df)}, visitor_id=v_id, session_id=s_id)
                        record_usage_event("import", "dataset_upload", count=1, metadata={"format": "text", "row_count": len(stable_df)}, visitor_id=v_id, session_id=s_id)
                    except Exception:
                        pass

                    st.session_state.active_df = stable_df
                    st.session_state.active_dataset_key = p_name or d_key
                    st.session_state.active_storage_path = s_path
                    st.session_state.idx = 0
                    st.session_state.main_nav_radio = "Analyse & Label"
                    st.rerun()

    # =========================================================================
    # OPTION C: SINGLE SENTENCE MANUAL ENTRY
    # =========================================================================
    elif input_mode == "✍️ Enter a single sentence manually":
        st.markdown("#### Enter one sentence to analyse immediately")
        st.caption("Perfect for a quick test or single-sentence query.")

        manual_sentence = st.text_input(
            "Sentence Input",
            placeholder="e.g. Kya yeh application Hindi support karta hai?",
            label_visibility="collapsed",
        )

        if manual_sentence.strip():
            raw_df, err = load_single_sentence(manual_sentence)
            if err:
                st.warning(err)
            elif raw_df is not None:
                if st.button("🔍 Analyse This Sentence", type="primary", use_container_width=True):
                    d_key = compute_dataset_key(manual_sentence.encode("utf-8"))
                    s_path = get_dataset_storage_path(d_key, root_dir)
                    stable_df, _ = stabilize_dataset(raw_df, "text", d_key)
                    if s_path.exists():
                        stable_df = merge_saved_annotations(stable_df, s_path)
                    else:
                        stable_df.to_csv(s_path, index=False, encoding="utf-8-sig")

                    _save_to_user_db(current_user, target_ws_id, "Single Sentence Query", stable_df, "single_sentence.txt", d_key)

                    try:
                        from db.analytics import record_usage_event
                        v_id = st.session_state.get("_visitor_id")
                        s_id = st.session_state.get("_session_id")
                        record_usage_event("analysis", "single_sentence_entry", count=1, metadata={"source": "manual_entry"}, visitor_id=v_id, session_id=s_id)
                    except Exception:
                        pass

                    st.session_state.active_df = stable_df
                    st.session_state.active_dataset_key = "Single Sentence"
                    st.session_state.active_storage_path = s_path
                    st.session_state.idx = 0
                    st.session_state.main_nav_radio = "Analyse & Label"
                    st.rerun()

    # =========================================================================
    # OPTION D: PRE-LOADED SAMPLE DATASET
    # =========================================================================
    else:
        sample_path = root_dir / "sample_data.csv"
        out_path = root_dir / "annotated_output.csv"

        st.markdown("#### Pre-Loaded Example Sentences (20 Sentences)")
        st.caption("A balanced collection of English, Hindi, and Hinglish sentences across positive, negative, and neutral sentiments.")

        if sample_path.exists():
            df_sample = pd.read_csv(sample_path, dtype=str)
            st.dataframe(df_sample[["id", "text"]].head(6), use_container_width=True)

            if st.button("🚀 Load Pre-Loaded Examples", type="primary", use_container_width=True):
                stable_df, _ = stabilize_dataset(df_sample, "text", "default_sample")
                if out_path.exists():
                    stable_df = merge_saved_annotations(stable_df, out_path)
                else:
                    stable_df.to_csv(out_path, index=False, encoding="utf-8-sig")

                _save_to_user_db(current_user, target_ws_id, "Pre-Loaded 20 Examples", stable_df, "sample_data.csv", "default_sample")

                try:
                    from db.analytics import record_usage_event
                    v_id = st.session_state.get("_visitor_id")
                    s_id = st.session_state.get("_session_id")
                    record_usage_event("import", "sample_dataset_load", count=1, metadata={"format": "sample_csv", "row_count": len(stable_df)}, visitor_id=v_id, session_id=s_id)
                except Exception:
                    pass

                st.session_state.active_df = stable_df
                st.session_state.active_dataset_key = "default_sample"
                st.session_state.active_storage_path = out_path
                st.session_state.idx = 0
                st.session_state.main_nav_radio = "Analyse & Label"
                st.rerun()

    return imported_df, dataset_key, storage_path

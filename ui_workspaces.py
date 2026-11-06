"""
LinguaLabel — Workspace & Dataset Management UI (Part 11)

Implements:
- Workspace creation, renaming, and safe deletion with confirmation.
- Dataset listing, opening, duplicating, renaming, and deletion.
- In-depth dataset progress tracking.
- Legacy storage migration into selected workspace.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from data_manager import export_excel_bytes
from db.migration_tool import migrate_legacy_data_to_user, preview_legacy_data
from db.service import (
    create_workspace,
    delete_dataset,
    delete_workspace,
    duplicate_dataset,
    export_user_dataset_df,
    get_dataset_sentences,
    get_workspace,
    list_datasets,
    list_workspaces,
    rename_dataset,
    rename_workspace,
)


def render_workspaces_page(root_dir: Path, current_user: Dict[str, Any]) -> None:
    """Renders workspace and dataset management interface."""
    user_id = current_user.get("id")

    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 2.1rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em; margin: 0;">
            📁 Workspaces & Datasets
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.35rem;">
            Organize your multilingual annotation projects into isolated personal workspaces.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Fetch user workspaces
    workspaces = list_workspaces(user_id)
    if not workspaces:
        # Fallback if no workspace found
        ws_new, _ = create_workspace(user_id, "Personal Workspace", "Default workspace")
        workspaces = [ws_new] if ws_new else []

    ws_map = {w.name: w for w in workspaces}
    ws_names = list(ws_map.keys())

    # Active workspace selection
    curr_ws_id = st.session_state.get("active_workspace_id", workspaces[0].id if workspaces else "")
    curr_ws = next((w for w in workspaces if w.id == curr_ws_id), workspaces[0] if workspaces else None)
    curr_ws_idx = ws_names.index(curr_ws.name) if (curr_ws and curr_ws.name in ws_names) else 0

    c_ws_select, c_ws_new = st.columns([3, 1.2])

    with c_ws_select:
        selected_ws_name = st.selectbox(
            "Select Workspace",
            ws_names,
            index=curr_ws_idx,
            key="workspace_selector_box",
            help="Choose a workspace to view and manage its datasets.",
        )
        selected_ws = ws_map.get(selected_ws_name)
        if selected_ws and selected_ws.id != st.session_state.get("active_workspace_id"):
            st.session_state.active_workspace_id = selected_ws.id

    with c_ws_new:
        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
        with st.popover("➕ Create Workspace", use_container_width=True):
            st.markdown("#### New Workspace")
            new_ws_name = st.text_input("Workspace Name", key="input_new_ws_name", placeholder="e.g. Hindi E-Commerce Reviews")
            new_ws_desc = st.text_area("Description (optional)", key="input_new_ws_desc", placeholder="Project goals and notes...")
            if st.button("Create Workspace", key="btn_confirm_create_ws", type="primary", use_container_width=True):
                if not new_ws_name.strip():
                    st.error("Please enter a workspace name.")
                else:
                    created_ws, err = create_workspace(user_id, new_ws_name.strip(), new_ws_desc.strip())
                    if err or not created_ws:
                        st.error(err or "Failed to create workspace.")
                    else:
                        st.session_state.active_workspace_id = created_ws.id
                        st.success(f"Workspace '{created_ws.name}' created!")
                        st.rerun()

    if not selected_ws:
        st.warning("No workspace selected.")
        return

    # Workspace Details & Management Expander
    with st.expander(f"⚙️ Workspace Settings: {selected_ws.name}", expanded=False):
        c_edit1, c_edit2 = st.columns(2)
        with c_edit1:
            st.markdown("##### Rename Workspace")
            ren_ws_name = st.text_input("New Name", value=selected_ws.name, key=f"ren_name_{selected_ws.id}")
            ren_ws_desc = st.text_area("New Description", value=selected_ws.description or "", key=f"ren_desc_{selected_ws.id}")
            if st.button("Save Changes", key=f"btn_save_ws_{selected_ws.id}"):
                ok, msg = rename_workspace(user_id, selected_ws.id, ren_ws_name, ren_ws_desc)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with c_edit2:
            st.markdown("##### Delete Workspace")
            st.warning("Deleting a workspace permanently deletes all datasets and annotations inside it.")
            confirm_del_ws = st.checkbox(
                f"I confirm I want to permanently delete '{selected_ws.name}'",
                key=f"chk_del_ws_{selected_ws.id}",
            )
            if st.button("🗑️ Delete Workspace Permanently", key=f"btn_del_ws_{selected_ws.id}", type="primary"):
                if not confirm_del_ws:
                    st.error("Please check the confirmation box above to proceed.")
                else:
                    ok_del, del_msg = delete_workspace(user_id, selected_ws.id)
                    if ok_del:
                        remaining = list_workspaces(user_id)
                        st.session_state.active_workspace_id = remaining[0].id if remaining else ""
                        st.success(del_msg)
                        st.rerun()
                    else:
                        st.error(del_msg)

    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # DATASETS IN WORKSPACE
    # -------------------------------------------------------------
    st.markdown(f"### 📦 Datasets in **{selected_ws.name}**")
    datasets = list_datasets(user_id, workspace_id=selected_ws.id)

    if not datasets:
        st.info("No datasets in this workspace yet. Click '📥 Import Sentences' in the navigation bar to add a dataset.")
    else:
        for ds in datasets:
            sentences, _ = get_dataset_sentences(user_id, ds.id)
            total_sents = len(sentences) if sentences else ds.total_sentences
            rev_count = sum(1 for s in sentences if s.review_status == "Reviewed — looks correct") if sentences else 0
            pct_done = int((rev_count / total_sents) * 100) if total_sents > 0 else 0

            is_active = (st.session_state.get("active_dataset_id") == ds.id)

            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid {'rgba(56, 189, 248, 0.5)' if is_active else 'rgba(255, 255, 255, 0.08)'}; border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.5rem;">
                            <strong style="font-size: 1.15rem; color: #f8fafc;">{ds.name}</strong>
                            {'<span style="font-size: 0.7rem; background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 7px; border-radius: 4px; font-weight: 600;">ACTIVE</span>' if is_active else ''}
                        </div>
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;">
                            {f"Original file: {ds.original_filename} • " if ds.original_filename else ""}
                            {total_sents} sentences • Created: {ds.created_at.strftime('%b %d, %Y')}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.95rem; font-weight: 700; color: #4ade80;">{rev_count} / {total_sents}</span>
                        <span style="font-size: 0.8rem; color: #94a3b8;"> ({pct_done}%)</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons for dataset
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1.5, 1.2, 1.2, 1.2, 1.5])

            with btn_col1:
                if st.button("📂 Open & Annotate", key=f"btn_open_ds_{ds.id}", use_container_width=True, type="primary"):
                    st.session_state.active_dataset_id = ds.id
                    st.session_state.active_workspace_id = selected_ws.id
                    # Load dataset into active_df
                    df_loaded, _ = export_user_dataset_df(user_id, ds.id)
                    if df_loaded is not None:
                        st.session_state.active_df = df_loaded
                        st.session_state.active_dataset_key = ds.name
                    st.session_state.main_nav_radio = "Analyse & Label"
                    st.rerun()

            with btn_col2:
                if st.button("📋 Duplicate", key=f"btn_dup_ds_{ds.id}", use_container_width=True):
                    dup_ds, err = duplicate_dataset(user_id, ds.id)
                    if dup_ds:
                        st.success(f"Duplicated as '{dup_ds.name}'")
                        st.rerun()
                    else:
                        st.error(err or "Failed to duplicate.")

            with btn_col3:
                with st.popover("✏️ Rename", use_container_width=True):
                    new_ds_title = st.text_input("New dataset name", value=ds.name, key=f"inp_ren_{ds.id}")
                    if st.button("Confirm Rename", key=f"btn_confirm_ren_{ds.id}", type="primary"):
                        ok_r, msg_r = rename_dataset(user_id, ds.id, new_ds_title)
                        if ok_r:
                            st.success(msg_r)
                            st.rerun()
                        else:
                            st.error(msg_r)

            with btn_col4:
                with st.popover("🗑️ Delete", use_container_width=True):
                    st.caption("Are you sure? This cannot be undone.")
                    if st.button("Confirm Delete", key=f"btn_confirm_del_{ds.id}", type="primary"):
                        ok_d, msg_d = delete_dataset(user_id, ds.id)
                        if ok_d:
                            if st.session_state.get("active_dataset_id") == ds.id:
                                st.session_state.active_dataset_id = ""
                            st.success(msg_d)
                            st.rerun()
                        else:
                            st.error(msg_d)

            with btn_col5:
                # Export options
                with st.popover("💾 Export Data", use_container_width=True):
                    df_exp, _ = export_user_dataset_df(user_id, ds.id)
                    if df_exp is not None and not df_exp.empty:
                        csv_data = df_exp.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 Download CSV",
                            data=csv_data,
                            file_name=f"{ds.name.replace(' ', '_').lower()}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key=f"dl_csv_{ds.id}",
                        )
                        excel_data = export_excel_bytes(df_exp)
                        if excel_data:
                            st.download_button(
                                "📥 Download Excel (.xlsx)",
                                data=excel_data,
                                file_name=f"{ds.name.replace(' ', '_').lower()}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key=f"dl_xl_{ds.id}",
                            )

    # -------------------------------------------------------------
    # LEGACY STORAGE MIGRATION SECTION
    # -------------------------------------------------------------
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("### 🔄 Legacy Data Migration Tool")
    st.caption("Import existing local session files (saved_sessions/) and feedback history into your current personal workspace.")

    preview = preview_legacy_data(root_dir)
    tot_legacy_sents = preview.get("total_session_sentences", 0)
    tot_legacy_fb = preview.get("total_feedback_records", 0)

    if tot_legacy_sents == 0 and tot_legacy_fb == 0:
        st.info("No unmigrated legacy session files or feedback logs found on disk.")
    else:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <strong>Legacy Files Found:</strong><br>
            • {preview['session_files_count']} session file(s) with {tot_legacy_sents} sentences ({preview['total_session_reviewed']} human-reviewed).<br>
            • {preview['feedback_files_count']} feedback file(s) with {tot_legacy_fb} records.<br>
            <em>All original local files are safely preserved and never deleted. A timestamped backup will be created automatically.</em>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"🚀 Import Legacy Data into '{selected_ws.name}'", type="primary"):
            with st.spinner("Creating backup and migrating records safely..."):
                res = migrate_legacy_data_to_user(user_id, selected_ws.id, root_dir)
                if res.get("status") == "success":
                    st.success(
                        f"Migration Complete! Imported {res['datasets_imported']} dataset(s) ({res['sentences_imported']} sentences) "
                        f"and {res['feedback_imported']} feedback items. Backup created in: {res['backup_dir']}"
                    )
                    st.rerun()
                else:
                    st.warning(res.get("message", "Migration completed."))


"""
LinguaLabel — Model Training & Continuous Improvement Center (Part 11.3)

Full-featured UI component for the website owner to:
1. View training overview and production models.
2. Upload and manage labelled datasets (CSV/Excel) with pre-training validation and class imbalance checks.
3. Curate and approve human feedback corrections (only approved corrections enter training).
4. Prepare unified, leakage-safe stratified training splits with provenance tracking.
5. Configure and execute reproducible model training runs (CPU & NVIDIA GPU auto-detection).
6. Evaluate and compare candidate models side-by-side with production models.
7. Explicitly promote candidate models to production or perform one-click rollbacks with full audit logging.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from ml.engine import MLEngine
from ml.training.data_preparation import UnifiedDataPreparationEngine
from ml.training.dataset_manager import DatasetManagementEngine
from ml.training.evaluator import ModelEvaluationEngine
from ml.training.feedback_collector import FeedbackCurationEngine
from ml.training.trainer import ModelTrainingEngine, TrainingJobConfig, TrainedModelPredictor
from ml.training.versioning import ModelVersionRegistry


def render_training_center(root_dir: Optional[Path] = None, ml_engine: Optional[MLEngine] = None):
    """
    Renders the Model Training & Continuous Improvement Center.
    Exclusively accessible to the website owner.
    """
    root = root_dir or Path(__file__).resolve().parent

    # Instantiate engines
    version_registry = ModelVersionRegistry(root)
    dataset_mgr = DatasetManagementEngine(root)
    feedback_curator = FeedbackCurationEngine(root)
    prep_engine = UnifiedDataPreparationEngine(root)
    trainer_engine = ModelTrainingEngine(root)
    evaluator_engine = ModelEvaluationEngine(root)

    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                    <span style="background: rgba(14, 165, 233, 0.2); color: #38bdf8; font-size: 0.78rem; font-weight: 700; padding: 2px 10px; border-radius: 9999px; border: 1px solid rgba(56, 189, 248, 0.4);">
                        🧠 CONTINUOUS IMPROVEMENT
                    </span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">
                        Two Improvement Channels: Owner Datasets + Approved Human Corrections
                    </span>
                </div>
                <h2 style="font-size: 1.6rem; font-weight: 800; color: #ffffff; margin: 0; letter-spacing: -0.02em;">
                    Model Training & Continuous Improvement Center
                </h2>
                <div style="font-size: 0.88rem; color: #94a3b8; margin-top: 0.25rem;">
                    Upload custom labelled data, curate approved annotator feedback, train candidate models, evaluate side-by-side, and control production deployment.
                </div>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.75rem; color: #4ade80; background: rgba(74, 222, 128, 0.1); border: 1px solid rgba(74, 222, 128, 0.3); padding: 4px 10px; border-radius: 6px;">
                    🛡️ Safe Promotion & Rollback Active
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 7 Comprehensive Tabs
    tab_overview, tab_datasets, tab_feedback, tab_prep, tab_train, tab_eval, tab_deploy = st.tabs([
        "📊 Training Overview",
        "📁 Dataset Management",
        "👥 Human Feedback Curation",
        "⚙️ Dataset Preparation",
        "🚀 Model Training",
        "⚖️ Model Evaluation & Comparison",
        "🎯 Production Promotion & Rollback",
    ])

    # =========================================================================
    # TAB 1: TRAINING OVERVIEW
    # =========================================================================
    with tab_overview:
        st.markdown("### 📊 Platform Training Overview")
        st.caption("Live status of production model versions, training assets, and active learning feedback.")

        active_prods = version_registry.get_active_production_models()
        datasets_list = dataset_mgr.list_datasets()
        fb_stats = feedback_curator.get_curation_statistics()
        all_versions = version_registry.list_versions()
        candidates = [v for v in all_versions if v.lifecycle_state == "candidate"]

        # Top 4 Primary KPI Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Active Custom Models",
                f"{len(active_prods)} / 5 Tasks",
                help="Tasks currently powered by custom trained models (others use baseline/transformers).",
            )
        with c2:
            st.metric(
                "Uploaded Training Datasets",
                f"{len(datasets_list)}",
                help="Validated CSV and Excel datasets uploaded by the owner.",
            )
        with c3:
            st.metric(
                "Approved Human Corrections",
                f"{fb_stats['approved']}",
                help="Human corrections reviewed and approved for future training inclusion.",
            )
        with c4:
            st.metric(
                "Candidate Models",
                f"{len(candidates)} Pending",
                help="Trained models awaiting evaluation and explicit owner promotion.",
            )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)

        # Production Models Table
        st.markdown("#### 🎯 Active Production Models by Task")
        prod_table_data = []
        for task in ["language", "sentiment", "text_type", "abuse_category", "abuse_severity"]:
            task_label = task.replace("_", " ").title()
            ver_id = active_prods.get(task)
            if ver_id:
                ver = version_registry.get_version(ver_id)
                prod_table_data.append({
                    "Task": task_label,
                    "Model Version": ver_id,
                    "Architecture": ver.architecture if ver else "custom",
                    "Promoted At": ver.promoted_at[:19].replace("T", " ") if (ver and ver.promoted_at) else "—",
                    "Val Accuracy": f"{ver.metrics.get('val_accuracy', 0.0)*100:.1f}%" if ver else "—",
                    "Status": "🟢 Production (Custom)",
                })
            else:
                prod_table_data.append({
                    "Task": task_label,
                    "Model Version": "default_baseline_v1",
                    "Architecture": "Pre-trained Multilingual Hybrid",
                    "Promoted At": "Default System",
                    "Val Accuracy": "Parity Baseline",
                    "Status": "⚪ Production (Default)",
                })

        st.dataframe(pd.DataFrame(prod_table_data), width="stretch", hide_index=True)

        # Recent Training Runs Summary
        st.markdown("#### 🕒 Recent Model Training History")
        if all_versions:
            ver_rows = []
            for v in all_versions[-8:]:
                state_badge = {
                    "production": "🟢 Production",
                    "candidate": "🟡 Candidate",
                    "archived": "⚪ Archived",
                    "rejected": "🔴 Rejected",
                    "rolled_back": "↩️ Rolled Back",
                }.get(v.lifecycle_state, v.lifecycle_state)
                ver_rows.append({
                    "Version ID": v.version_id,
                    "Model Name": v.model_name,
                    "Task": v.task,
                    "Architecture": v.architecture,
                    "Created At": v.created_at[:19].replace("T", " "),
                    "State": state_badge,
                    "Val Acc": f"{v.metrics.get('val_accuracy', 0.0)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(ver_rows), width="stretch", hide_index=True)
        else:
            st.info("No custom models trained yet. Upload datasets and run your first training cycle in the tabs above!")

    # =========================================================================
    # TAB 2: DATASET MANAGEMENT
    # =========================================================================
    with tab_datasets:
        st.markdown("### 📁 Dataset Management")
        st.caption("Upload, inspect, and validate owner training datasets in CSV or Excel format.")

        u_col1, u_col2 = st.columns([1.5, 1])

        with u_col1:
            st.markdown("#### Upload Labelled Dataset")
            uploaded_file = st.file_uploader(
                "Choose CSV or Excel file",
                type=["csv", "xlsx", "xls"],
                key="tr_dataset_uploader",
                help="File must contain sentence text and a label column for the target task.",
            )

            ds_task = st.selectbox(
                "Target Classification Task",
                [
                    ("language", "Language Detection"),
                    ("sentiment", "Sentiment Analysis"),
                    ("text_type", "Sentence Type Classification"),
                    ("complaint", "Complaining Detection"),
                    ("abuse", "Abusive Language Detection"),
                    ("abuse_severity", "Abuse Severity (None, Mild, Moderate, Severe)"),
                ],
                format_func=lambda x: x[1],
                key="tr_ds_task_select",
            )[0]

            ds_desc = st.text_input("Dataset Description / Provenance Notes", placeholder="e.g. Curated bilingual customer support messages 2026", key="tr_ds_desc")

            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                # Preview dataframe
                try:
                    raw_df = dataset_mgr._read_file_to_df(file_bytes, uploaded_file.name)
                    st.markdown(f"**Preview of `{uploaded_file.name}`** ({len(raw_df)} rows, {len(raw_df.columns)} columns):")
                    st.dataframe(raw_df.head(5), width="stretch")

                    text_col_guess, lbl_col_guess = dataset_mgr.detect_columns(raw_df, ds_task)
                    col_names = list(raw_df.columns)

                    c_tc1, c_tc2 = st.columns(2)
                    with c_tc1:
                        sel_text_col = st.selectbox("Sentence Text Column", col_names, index=col_names.index(text_col_guess) if text_col_guess in col_names else 0)
                    with c_tc2:
                        sel_lbl_col = st.selectbox("Label Column", col_names, index=col_names.index(lbl_col_guess) if lbl_col_guess in col_names else 0)

                    # Validation Report Button
                    if st.button("🔍 Validate Dataset Quality", width="stretch"):
                        val_rep = dataset_mgr.validate_dataframe(raw_df, sel_text_col, sel_lbl_col, ds_task)
                        st.session_state["_last_ds_val_rep"] = val_rep

                    val_rep = st.session_state.get("_last_ds_val_rep")
                    if val_rep:
                        if val_rep.is_valid:
                            st.success(f"✅ Dataset Quality Verified! Total valid samples: **{val_rep.valid_samples_count:,}**")
                        else:
                            st.error(f"⚠️ Validation Issues Detected! ({len(val_rep.errors)} errors)")

                        # Metrics breakdown
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Rows", f"{val_rep.total_rows:,}")
                        m2.metric("Missing Values", f"{val_rep.missing_values_count}")
                        m3.metric("Duplicate Texts", f"{val_rep.duplicate_text_count}")
                        m4.metric("Class Imbalance", f"{val_rep.class_imbalance_ratio:.1f}:1")

                        if val_rep.invalid_labels:
                            st.warning(f"**Incompatible Labels Found:** {', '.join(val_rep.invalid_labels)}")

                        if val_rep.class_distribution:
                            st.markdown("**Class Label Distribution:**")
                            st.dataframe(pd.DataFrame(list(val_rep.class_distribution.items()), columns=["Label", "Count"]), width="stretch", hide_index=True)

                        if st.button("💾 Save to Training Repository", type="primary", width="stretch"):
                            entry = dataset_mgr.save_dataset(
                                file_bytes=file_bytes,
                                filename=uploaded_file.name,
                                task=ds_task,
                                text_col=sel_text_col,
                                label_col=sel_lbl_col,
                                description=ds_desc,
                            )
                            st.success(f"Successfully saved `{entry.filename}` (ID: `{entry.dataset_id}`) to training repository!")
                            st.session_state.pop("_last_ds_val_rep", None)
                            st.rerun()

                except Exception as e:
                    st.error(f"Error parsing uploaded file: {e}")

        with u_col2:
            st.markdown("#### Available Datasets Repository")
            cur_datasets = dataset_mgr.list_datasets()
            if not cur_datasets:
                st.info("No training datasets uploaded yet.")
            else:
                for ds in cur_datasets:
                    with st.expander(f"📄 {ds.filename} ({ds.task.upper()}) — {ds.num_samples:,} rows"):
                        st.caption(f"**ID:** `{ds.dataset_id}` | **Uploaded:** {ds.uploaded_at[:19].replace('T', ' ')}")
                        st.caption(f"**Text Col:** `{ds.text_col}` | **Label Col:** `{ds.label_col}`")
                        st.caption(f"**Class Imbalance:** {ds.validation_summary.get('class_imbalance_ratio', 1.0):.1f}:1")
                        if ds.description:
                            st.caption(f"**Notes:** {ds.description}")
                        if st.button("🗑️ Delete Dataset", key=f"del_ds_{ds.dataset_id}", help="Remove dataset from repository"):
                            dataset_mgr.delete_dataset(ds.dataset_id)
                            st.success(f"Deleted dataset {ds.filename}")
                            st.rerun()

    # =========================================================================
    # TAB 3: HUMAN FEEDBACK CURATION
    # =========================================================================
    with tab_feedback:
        st.markdown("### 👥 Human Feedback Curation")
        st.caption("Review, approve, or reject human corrections. Only approved feedback is included in model training.")

        fb_stats = feedback_curator.get_curation_statistics()
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Total Submitted Corrections", f"{fb_stats['total_feedback']}")
        fc2.metric("Pending Review", f"{fb_stats['pending']}", delta=f"{fb_stats['pending']} unreviewed")
        fc3.metric("Approved for Training", f"{fb_stats['approved']}")
        fc4.metric("Rejected (Excluded)", f"{fb_stats['rejected']}")

        st.markdown("""
        <div style="background: rgba(14, 165, 233, 0.08); border-left: 3px solid #38bdf8; padding: 0.6rem 1rem; border-radius: 4px; margin: 1rem 0; font-size: 0.85rem; color: #cbd5e1;">
            <strong>Controlled Training Guardrail:</strong> Unreviewed visitor submissions and rejected corrections are <span style="color: #f87171; font-weight: 600;">STRICTLY EXCLUDED</span> from all model training runs to prevent dataset poisoning and label drift.
        </div>
        """, unsafe_allow_html=True)

        all_curated = feedback_curator.list_curated_feedback()

        # Filter controls
        f_c1, f_c2 = st.columns([2, 1])
        with f_c1:
            status_filter = st.radio("Filter Status", ["All", "pending", "approved", "rejected"], horizontal=True, key="tr_fb_filter")
        with f_c2:
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            if st.button("✅ Approve All Pending", width="stretch", help="Approve all currently pending corrections"):
                count = feedback_curator.bulk_curate("approve_all_pending")
                st.success(f"Approved {count} pending feedback items!")
                st.rerun()

        filtered_items = [
            item for item in all_curated
            if status_filter == "All" or item.curation_status == status_filter
        ]

        if not filtered_items:
            st.info("No feedback items match the selected filter.")
        else:
            for item in filtered_items:
                badge_color = {
                    "pending": "#fbbf24",
                    "approved": "#4ade80",
                    "rejected": "#f87171",
                }.get(item.curation_status, "#94a3b8")

                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.5); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.4rem;">
                            <span style="font-weight: 600; color: #f1f5f9; font-size: 0.95rem;">"{item.text}"</span>
                            <span style="font-size: 0.75rem; color: {badge_color}; border: 1px solid {badge_color}; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; font-weight: 700;">
                                {item.curation_status}
                            </span>
                        </div>
                        <div style="display: flex; gap: 1.5rem; font-size: 0.82rem; color: #94a3b8; margin-bottom: 0.5rem;">
                            <span>Task: <strong style="color: #e2e8f0;">{item.task.title()}</strong></span>
                            <span>AI Suggested: <strong style="color: #f87171;">{item.suggested_label}</strong></span>
                            <span>Human Corrected: <strong style="color: #4ade80;">{item.corrected_label}</strong></span>
                            <span>Notes: <em style="color: #cbd5e1;">{item.notes or 'None'}</em></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    btn_c1, btn_c2, btn_c3 = st.columns([1, 1, 4])
                    if btn_c1.button("✅ Approve", key=f"app_{item.feedback_id}", width="stretch"):
                        feedback_curator.curate_feedback(item.feedback_id, "approved", reviewer_notes="Approved via Training Center")
                        st.rerun()
                    if btn_c2.button("❌ Reject", key=f"rej_{item.feedback_id}", width="stretch"):
                        feedback_curator.curate_feedback(item.feedback_id, "rejected", reviewer_notes="Rejected via Training Center")
                        st.rerun()

    # =========================================================================
    # TAB 4: DATASET PREPARATION
    # =========================================================================
    with tab_prep:
        st.markdown("### ⚙️ Dataset Preparation & Partitioning")
        st.caption("Merge owner datasets and approved feedback with provenance tracking, leakage prevention, and stratified splits.")

        datasets = dataset_mgr.list_datasets()
        if not datasets:
            st.warning("Please upload at least one training dataset in the 'Dataset Management' tab first.")
        else:
            p_task = st.selectbox(
                "Select Target Task for Training Split",
                ["sentiment", "language", "text_type", "complaint", "abuse", "abuse_severity"],
                key="tr_prep_task",
            )

            compatible_datasets = [d for d in datasets if d.task.lower() == p_task.lower()]
            if not compatible_datasets:
                st.warning(f"No uploaded datasets match task '{p_task}'. Upload a dataset for this task in Tab 2.")
            else:
                sel_dataset_ids = st.multiselect(
                    "Select Datasets to Include",
                    options=[d.dataset_id for d in compatible_datasets],
                    default=[compatible_datasets[0].dataset_id],
                    format_func=lambda x: f"{next(d.filename for d in compatible_datasets if d.dataset_id == x)} ({next(d.num_samples for d in compatible_datasets if d.dataset_id == x)} samples)",
                    key="tr_prep_ds_select",
                )

                include_fb = st.checkbox("Include Approved Human Feedback Corrections", value=True, key="tr_prep_inc_fb")

                sp_c1, sp_c2, sp_c3 = st.columns(3)
                with sp_c1:
                    train_ratio = st.slider("Train Split Ratio", min_value=0.50, max_value=0.85, value=0.70, step=0.05)
                with sp_c2:
                    val_ratio = st.slider("Validation Split Ratio", min_value=0.10, max_value=0.30, value=0.15, step=0.05)
                with sp_c3:
                    test_ratio = round(1.0 - train_ratio - val_ratio, 2)
                    st.metric("Test Split Ratio (Sequestered)", f"{test_ratio:.2f}")

                random_seed = st.number_input("Random Seed (Reproducibility)", value=42, step=1)

                if st.button("⚡ Generate Leakage-Safe Training Split", type="primary", width="stretch"):
                    with st.spinner("Merging, deduplicating, resolving provenance, and stratifying..."):
                        split_res = prep_engine.prepare_unified_dataset(
                            dataset_ids=sel_dataset_ids,
                            task=p_task,
                            include_approved_feedback=include_fb,
                            train_ratio=train_ratio,
                            val_ratio=val_ratio,
                            random_seed=int(random_seed),
                        )
                        st.session_state["_last_prepared_split"] = split_res

                split_res = st.session_state.get("_last_prepared_split")
                if split_res:
                    st.success(f"✅ Split Prepared Successfully! ID: `{split_res.split_id}`")
                    sr1, sr2, sr3, sr4 = st.columns(4)
                    sr1.metric("Total Unique Records", f"{split_res.total_records:,}")
                    sr2.metric("Train Split", f"{split_res.train_records:,}")
                    sr3.metric("Validation Split", f"{split_res.val_records:,}")
                    sr4.metric("Sequestered Test", f"{split_res.test_records:,}")

                    # Provenance breakdown
                    st.markdown("**Dataset Provenance Breakdown:**")
                    prov_rows = [
                        {"Source": src.replace("_", " ").title(), "Record Count": count}
                        for src, count in split_res.provenance_breakdown.items()
                    ]
                    st.dataframe(pd.DataFrame(prov_rows), width="stretch", hide_index=True)

    # =========================================================================
    # TAB 5: MODEL TRAINING EXECUTION
    # =========================================================================
    with tab_train:
        st.markdown("### 🚀 Model Training Execution")
        st.caption("Train candidate models using reproducible pipelines, isolated scratch builds, and GPU acceleration.")

        splits = prep_engine.list_prepared_splits()
        if not splits:
            st.warning("No prepared training splits found. Please prepare a split in the 'Dataset Preparation' tab.")
        else:
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                sel_split = st.selectbox(
                    "Choose Training Split",
                    options=splits,
                    format_func=lambda s: f"[{s.task.upper()}] {s.split_id} ({s.train_records} train / {s.val_records} val)",
                    key="tr_train_sel_split",
                )
                model_name = st.text_input("Candidate Model Name", value=f"{sel_split.task.title()} Classifier Candidate", key="tr_train_mname")
                notes = st.text_area("Training Run Notes / Objective", placeholder="e.g. Iteration with 50 new Hinglish complaints and high C regularization", key="tr_train_notes")

            with t_col2:
                arch = st.selectbox(
                    "Model Architecture",
                    [
                        ("tfidf_logistic_regression", "TF-IDF + Calibrated Logistic Regression (Fast, Interpretable)"),
                        ("calibrated_sgd", "Linear SGD Classifier (High dimensional, L2 regularized)"),
                    ],
                    format_func=lambda x: x[1],
                    key="tr_train_arch",
                )[0]

                # Hardware device resolution
                dev_key, dev_label = trainer_engine.resolve_device("auto")
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem; margin-bottom: 0.75rem;">
                    <span style="font-size: 0.8rem; color: #94a3b8;">Hardware Acceleration Target:</span><br>
                    <strong style="color: #4ade80; font-size: 0.95rem;">⚡ {dev_label}</strong>
                </div>
                """, unsafe_allow_html=True)

                c_param = st.slider("Regularization Parameter (C)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                max_feat = st.select_slider("Max TF-IDF Features", options=[5000, 10000, 20000, 30000, 50000], value=20000)

            if st.button("🔥 Launch Model Training Job", type="primary", width="stretch"):
                train_cfg = TrainingJobConfig(
                    task=sel_split.task,
                    split_id=sel_split.split_id,
                    model_name=model_name,
                    architecture=arch,
                    hyperparameters={"C": c_param, "max_features": max_feat},
                    device="auto",
                    notes=notes,
                )

                prog_bar = st.progress(0.0)
                status_txt = st.empty()
                log_box = st.empty()

                def update_ui(pct: float, msg: str):
                    prog_bar.progress(pct)
                    status_txt.text(f"⏳ {msg}")

                with st.spinner("Training model candidate in isolated scratch directory..."):
                    result = trainer_engine.train_model(train_cfg, progress_callback=update_ui)
                    st.session_state["_last_training_result"] = result

            tr_res = st.session_state.get("_last_training_result")
            if tr_res:
                if tr_res.success:
                    st.success(f"🎉 Model Trained Successfully! Registered Version ID: `{tr_res.version_id}`")
                    st.info(f"**Lifecycle State:** `candidate` (Preserved in candidate holding; production model untouched).")
                    m_c1, m_c2, m_c3 = st.columns(3)
                    m_c1.metric("Train Accuracy", f"{tr_res.train_accuracy*100:.1f}%")
                    m_c2.metric("Validation Accuracy", f"{tr_res.val_accuracy*100:.1f}%")
                    m_c3.metric("Duration", f"{tr_res.duration_seconds:.2f}s")

                    with st.expander("📜 Training Run Execution Logs", expanded=False):
                        st.code("\n".join(tr_res.logs), language="text")
                else:
                    st.error(f"Training failed: {tr_res.error_message}")

    # =========================================================================
    # TAB 6: MODEL COMPARISON & EVALUATION
    # =========================================================================
    with tab_eval:
        st.markdown("### ⚖️ Model Comparison & Rigorous Evaluation")
        st.caption("Side-by-side performance evaluation of candidate models against the active production model.")

        all_vers = version_registry.list_versions()
        candidates = [v for v in all_vers if v.lifecycle_state == "candidate"]

        if not candidates:
            st.info("No candidate models awaiting evaluation. Train a candidate in Tab 5 first!")
        else:
            sel_candidate = st.selectbox(
                "Select Candidate Model to Evaluate",
                candidates,
                format_func=lambda v: f"{v.model_name} (ID: {v.version_id} | Task: {v.task.upper()})",
                key="tr_eval_sel_cand",
            )

            # Choose test set: Sequestered split or fixed benchmark
            eval_source = st.radio(
                "Evaluation Dataset Source",
                ["Fixed Benchmark Suite", "Sequestered Test Split"],
                horizontal=True,
                key="tr_eval_src",
            )

            if st.button("🔬 Run Comprehensive Evaluation & Comparison", type="primary", width="stretch"):
                with st.spinner("Loading candidate model and executing comparative evaluation..."):
                    cand_predictor = TrainedModelPredictor(
                        artifacts_dir=Path(sel_candidate.artifacts_path),
                        task=sel_candidate.task,
                    )

                    # Determine production predictor for baseline comparison
                    active_map = version_registry.get_active_production_models()
                    prod_ver_id = active_map.get(sel_candidate.task)

                    if prod_ver_id:
                        prod_ver = version_registry.get_version(prod_ver_id)
                        prod_predictor = TrainedModelPredictor(
                            artifacts_dir=Path(prod_ver.artifacts_path),
                            task=prod_ver.task,
                        )
                        prod_model_name = prod_ver.model_name
                    else:
                        # Fallback baseline wrapper
                        engine = ml_engine or MLEngine()
                        class BaselinePredictorWrapper:
                            def __init__(self, task: str):
                                self.task = task
                            def predict(self, text: str):
                                if self.task == "sentiment":
                                    return engine.predict_sentiment(text)
                                elif self.task in ("text_type", "complaint"):
                                    return engine.predict_text_type(text)
                                elif self.task == "language":
                                    return engine.predict_language(text)
                                elif self.task in ("abuse", "abuse_category"):
                                    return engine.predict_abuse(text)
                                elif self.task in ("abuse_severity", "severity"):
                                    return engine.predict_abuse_severity(text)
                                return engine.predict_sentiment(text)
                        prod_predictor = BaselinePredictorWrapper(sel_candidate.task)
                        prod_model_name = "Production Default Baseline"

                    # Determine dataset
                    test_df = None
                    ds_name = "Benchmark"
                    if eval_source == "Fixed Benchmark Suite":
                        benchmark_file = "complaints_eval.csv" if sel_candidate.task in ("complaint", "text_type") else "abuse_eval.csv"
                        b_path = root / "evaluation_datasets" / benchmark_file
                        if b_path.exists():
                            test_df = pd.read_csv(b_path)
                            ds_name = benchmark_file
                    else:
                        # Load from prepared split test.csv
                        sp_path = root / "training_splits" / sel_candidate.training_config.get("split_id", "") / "test.csv"
                        if sp_path.exists():
                            test_df = pd.read_csv(sp_path)
                            ds_name = f"Test Split ({sel_candidate.training_config.get('split_id')})"

                    if test_df is None or test_df.empty:
                        st.error("Evaluation dataset could not be loaded. Please ensure benchmarks or splits exist.")
                    else:
                        prod_rep = evaluator_engine.evaluate(
                            predictor=prod_predictor,
                            dataset=test_df,
                            task=sel_candidate.task,
                            model_name=prod_model_name,
                            dataset_name=ds_name,
                        )
                        cand_rep = evaluator_engine.evaluate(
                            predictor=cand_predictor,
                            dataset=test_df,
                            task=sel_candidate.task,
                            model_name=sel_candidate.model_name,
                            dataset_name=ds_name,
                        )
                        comp_rep = evaluator_engine.compare_reports(prod_rep, cand_rep)
                        st.session_state["_last_comparison_report"] = comp_rep

            comp_rep = st.session_state.get("_last_comparison_report")
            if comp_rep:
                st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.25rem 0;'>", unsafe_allow_html=True)

                # Verdict Banner
                if comp_rep.recommend_promotion:
                    st.success(f"### ✅ {comp_rep.verdict_summary}")
                else:
                    st.error(f"### ⛔ {comp_rep.verdict_summary}")

                if comp_rep.regression_warnings:
                    for w in comp_rep.regression_warnings:
                        st.warning(f"⚠️ **Regression Warning:** {w}")

                # Metrics Comparison Table
                st.markdown("#### 📊 Metric Comparison Table")
                table_data = []
                for row in comp_rep.comparison_table:
                    delta_str = f"+{row.delta*100:.1f}%" if row.delta > 0 else f"{row.delta*100:.1f}%"
                    table_data.append({
                        "Metric": row.metric_name,
                        "Production": f"{row.production_value*100:.1f}%",
                        "Candidate": f"{row.candidate_value*100:.1f}%",
                        "Delta": f"{row.status_icon} {delta_str}",
                        "Description": row.description,
                    })
                st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

                # Language Stratified Comparison
                if comp_rep.per_language_comparison:
                    st.markdown("#### 🌐 Language-Stratified Macro F1")
                    lang_rows = []
                    for lang, data in comp_rep.per_language_comparison.items():
                        delta_str = f"+{data['delta']*100:.1f}%" if data['delta'] > 0 else f"{data['delta']*100:.1f}%"
                        icon = "🟢" if data['delta'] > 0 else ("🔴" if data['delta'] < 0 else "⚪")
                        lang_rows.append({
                            "Language": lang,
                            "Production F1": f"{data['production_macro_f1']*100:.1f}%",
                            "Candidate F1": f"{data['candidate_macro_f1']*100:.1f}%",
                            "Delta": f"{icon} {delta_str}",
                        })
                    st.dataframe(pd.DataFrame(lang_rows), width="stretch", hide_index=True)

                # Specialized Diagnostic Checks
                if comp_rep.diagnostic_comparison:
                    st.markdown("#### 🎯 Regression Diagnostic Checks")
                    for diag_name, data in comp_rep.diagnostic_comparison.items():
                        diag_label = diag_name.replace("_", " ").title()
                        st.markdown(f"- **{diag_label}:** Production: `{data['production']*100:.1f}%` → Candidate: `{data['candidate']*100:.1f}%` ({'🟢 Lower FP Rate' if data['improved'] else '🔴 Higher FP Rate'})")

    # =========================================================================
    # TAB 7: PRODUCTION PROMOTION & ROLLBACK
    # =========================================================================
    with tab_deploy:
        st.markdown("### 🎯 Model Promotion & Instant Rollback")
        st.caption("Deploy verified candidate models to active production or roll back to any previous version.")

        all_vers = version_registry.list_versions()
        candidates = [v for v in all_vers if v.lifecycle_state == "candidate"]

        dp_col1, dp_col2 = st.columns(2)

        with dp_col1:
            st.markdown("#### 🚀 Promote Candidate to Production")
            if not candidates:
                st.info("No candidates available for promotion.")
            else:
                sel_cand = st.selectbox(
                    "Choose Candidate to Promote",
                    candidates,
                    format_func=lambda v: f"{v.model_name} (ID: {v.version_id} | Task: {v.task.upper()})",
                    key="tr_deploy_sel_cand",
                )

                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.75rem;">
                    <strong>Target Task:</strong> {sel_cand.task.title()}<br>
                    <strong>Architecture:</strong> {sel_cand.architecture}<br>
                    <strong>Validation Accuracy:</strong> {sel_cand.metrics.get('val_accuracy', 0.0)*100:.1f}%
                </div>
                """, unsafe_allow_html=True)

                confirm_promote = st.checkbox(
                    "I confirm that I have evaluated this candidate and approve promoting it to active production.",
                    key="tr_confirm_promote",
                )

                if st.button("🚀 Promote to Production Now", type="primary", disabled=not confirm_promote, width="stretch"):
                    success = version_registry.promote_to_production(
                        version_id=sel_cand.version_id,
                        actor="website_owner",
                        notes=f"Owner approved promotion via Training Center for task {sel_cand.task}",
                    )
                    if success:
                        if ml_engine:
                            ml_engine.reload_active_models()
                        st.success(f"🎉 Version `{sel_cand.version_id}` is now active in production!")
                        st.rerun()
                    else:
                        st.error("Failed to promote candidate model.")

        with dp_col2:
            st.markdown("#### ↩️ Instant Version Rollback")
            active_prods = version_registry.get_active_production_models()
            if not active_prods:
                st.info("All tasks are running on default system baselines. No custom models to roll back.")
            else:
                roll_task = st.selectbox("Select Task to Roll Back", list(active_prods.keys()), key="tr_rollback_task")
                cur_active_ver = active_prods[roll_task]

                st.markdown(f"**Current Production Model:** `{cur_active_ver}`")

                rb_action = st.radio(
                    "Rollback Destination",
                    ["Reset to Default System Baseline", "Roll Back to Previous Archived Version"],
                    key="tr_rb_dest",
                )

                target_rb_ver = None
                if rb_action == "Roll Back to Previous Archived Version":
                    archived = [v for v in all_vers if v.task == roll_task and v.lifecycle_state in ("archived", "rolled_back")]
                    if archived:
                        target_rb_ver = st.selectbox(
                            "Select Previous Version",
                            archived,
                            format_func=lambda v: f"{v.version_id} ({v.model_name})",
                            key="tr_rb_target_ver",
                        ).version_id
                    else:
                        st.caption("No previous archived versions available for this task.")

                if st.button("↩️ Execute Safe Rollback", type="secondary", width="stretch"):
                    success = version_registry.rollback_production(
                        task=roll_task,
                        target_version_id=target_rb_ver,
                        actor="website_owner",
                        notes=f"Owner requested rollback for task {roll_task}",
                    )
                    if success:
                        if ml_engine:
                            ml_engine.reload_active_models()
                        st.success(f"Successfully rolled back {roll_task}!")
                        st.rerun()
                    else:
                        st.error("Rollback failed.")

        # Lifecycle Audit History Table
        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.markdown("#### 📜 Production Deployment & Rollback Audit Trail")
        audit_trail = version_registry.get_audit_trail(limit=20)
        if audit_trail:
            audit_rows = [
                {
                    "Timestamp": entry.get("timestamp", "")[:19].replace("T", " "),
                    "Action": entry.get("action", "").upper(),
                    "Task": entry.get("task", ""),
                    "Version ID": entry.get("version_id", ""),
                    "Actor": entry.get("actor", ""),
                    "Notes": entry.get("notes", ""),
                }
                for entry in audit_trail
            ]
            st.dataframe(pd.DataFrame(audit_rows), width="stretch", hide_index=True)
        else:
            st.caption("No deployment lifecycle actions recorded yet.")


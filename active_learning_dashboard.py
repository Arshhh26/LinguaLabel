"""
LinguaLabel — Active Learning & Intelligent Annotation Dashboard (Part 9)

Provides:
- Real-time multi-criteria prioritization queue (Uncertainty, Disagreement, Review Status, Ambiguity).
- Configurable sampling strategies (Lowest-Confidence, Margin, Shannon Entropy, Disagreement).
- Seamless deep-link integration into LinguaLabel Annotation Workspace.
- Isolated Human Feedback persistence, tracking, and analytics.
- Controlled Training Data preparation, label validation, fingerprinting, and split generation.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import altair as alt
import pandas as pd
import streamlit as st

from ml.active_learning.disagreement import compute_record_disagreement
from ml.active_learning.feedback import (
    FeedbackManager,
    compute_feedback_analytics,
)
from ml.active_learning.preparation import (
    format_feedback_for_training,
    prepare_controlled_split,
)
from ml.active_learning.priority import PriorityWeights
from ml.active_learning.queue import ActiveLearningQueue
from ml.active_learning.uncertainty import compute_record_uncertainty
from ml.engine import MLEngine
from ml.model_config import MLBackend


def render_active_learning_dashboard(
    root_dir: Path,
    df: pd.DataFrame,
    ml_engine: MLEngine,
    dataset_key: str = "default_sample",
    storage_path: Optional[Path] = None,
):
    """
    Renders the dark-themed Active Learning Dashboard and Prioritized Queue.
    """
    st.markdown("""
    <div class="header-container" style="margin-bottom: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
                    <span class="header-badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.3);">
                        🎯 ACTIVE LEARNING ENGINE
                    </span>
                    <span class="header-badge" style="background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);">
                        ⚡ HUMAN-IN-THE-LOOP PRIORITIZATION
                    </span>
                </div>
                <h1 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.02em;">
                    Active Learning & Intelligent Annotation Engine
                </h1>
                <div class="header-subtitle" style="color: #94a3b8; font-size: 0.9rem;">
                    Empirically sample uncertain items, surface cross-model disagreements, prioritize human review, and track verified feedback.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("⚠️ No active dataset loaded. Please upload a dataset or select the default sample from the sidebar.")
        return

    # Initialize feedback manager
    feedback_mgr = FeedbackManager(root_dir)

    # Top Tabs
    tab_queue, tab_feedback, tab_prep = st.tabs([
        "📋 Prioritized Annotation Queue",
        "📈 Human Feedback & Analytics",
        "📦 Controlled Learning Preparation",
    ])

    # =========================================================================
    # TAB 1: PRIORITIZED ANNOTATION QUEUE
    # =========================================================================
    with tab_queue:
        # --- SAMPLING STRATEGY & CONFIGURATION SIDEBAR/CONTROLS ---
        with st.expander("⚙️ Prioritization & Sampling Parameters", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                strategy = st.selectbox(
                    "Primary Sampling Strategy",
                    options=["lowest_confidence", "margin", "entropy", "disagreement", "priority"],
                    index=4,
                    format_func=lambda x: {
                        "lowest_confidence": "Lowest Confidence (1 - max score)",
                        "margin": "Smallest Margin (Top 1 vs Top 2 class gap)",
                        "entropy": "Shannon Entropy (Probability distribution)",
                        "disagreement": "Cross-Model Disagreement (Baseline vs Transformer)",
                        "priority": "Multi-Factor Weighted Priority",
                    }.get(x, x),
                    help="Determines the mathematical uncertainty/disagreement criteria used to evaluate items."
                )
            with sc2:
                run_transformer = st.checkbox(
                    "Evaluate Transformer Backend",
                    value=True,
                    help="Runs XLM-R sentiment and mDeBERTa-v3 intent models alongside baseline to detect genuine model disagreement."
                )
            with sc3:
                sort_order = st.selectbox(
                    "Queue Sort Order",
                    options=["priority", "uncertainty", "margin", "entropy", "disagreement"],
                    index=0,
                    format_func=lambda x: {
                        "priority": "Composite Priority (High to Low)",
                        "uncertainty": "Uncertainty (Highest First)",
                        "margin": "Smallest Margin (Smallest Gap First)",
                        "entropy": "Normalized Entropy (Highest First)",
                        "disagreement": "Disagreement Score (Highest First)",
                    }.get(x, x),
                )

            st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #94a3b8; margin: 0.5rem 0 0.25rem 0;'>Multi-Factor Priority Weights</div>", unsafe_allow_html=True)
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                w_disag = st.slider("Disagreement Weight", 0.0, 1.0, 0.35, 0.05)
            with w2:
                w_unc = st.slider("Uncertainty Weight", 0.0, 1.0, 0.35, 0.05)
            with w3:
                w_miss = st.slider("Missing Fields Weight", 0.0, 1.0, 0.15, 0.05)
            with w4:
                w_amb = st.slider("Ambiguity Weight", 0.0, 1.0, 0.15, 0.05)

            weights = PriorityWeights(
                weight_disagreement=w_disag,
                weight_uncertainty=w_unc,
                weight_missing=w_miss,
                weight_ambiguous=w_amb,
            )

        # Build Queue cache
        cache_key = f"al_queue_{dataset_key}_{strategy}_{run_transformer}_{len(df)}"
        if cache_key not in st.session_state or st.button("🔄 Refresh Priority Queue", key="btn_refresh_al_queue"):
            with st.spinner("Analyzing dataset with hybrid models & computing active learning scores..."):
                queue_obj = ActiveLearningQueue(
                    df=df,
                    ml_engine=ml_engine,
                    strategy=strategy,
                    weights=weights,
                    run_transformer=run_transformer,
                )
                st.session_state[cache_key] = queue_obj
        else:
            queue_obj = st.session_state[cache_key]

        summary = queue_obj.summary

        # --- A. OVERVIEW METRICS CARDS ---
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1.25rem;">
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; font-weight: 600;">Total Records</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #f8fafc; margin-top: 0.2rem;">{summary.total_records}</div>
                <div style="font-size: 0.7rem; color: #64748b;">In active dataset</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.72rem; color: #fbbf24; text-transform: uppercase; font-weight: 600;">Unreviewed</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #fbbf24; margin-top: 0.2rem;">{summary.unreviewed_count}</div>
                <div style="font-size: 0.7rem; color: #64748b;">Pending human review</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.72rem; color: #f87171; text-transform: uppercase; font-weight: 600;">High Priority</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #f87171; margin-top: 0.2rem;">{summary.high_priority_count}</div>
                <div style="font-size: 0.7rem; color: #64748b;">Requires immediate review</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 8px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.72rem; color: #c084fc; text-transform: uppercase; font-weight: 600;">Disagreements</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #c084fc; margin-top: 0.2rem;">{summary.disagreement_count}</div>
                <div style="font-size: 0.7rem; color: #64748b;">Baseline vs Transformer</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(14, 165, 233, 0.2); border-radius: 8px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.72rem; color: #38bdf8; text-transform: uppercase; font-weight: 600;">Uncertain</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8; margin-top: 0.2rem;">{summary.uncertain_count}</div>
                <div style="font-size: 0.7rem; color: #64748b;">Confidence &lt; 60%</div>
            </div>
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 0.85rem 1rem;">
                <div style="font-size: 0.72rem; color: #4ade80; text-transform: uppercase; font-weight: 600;">Completed</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #4ade80; margin-top: 0.2rem;">{summary.completed_count}</div>
                <div style="font-size: 0.7rem; color: #64748b;">Verified & approved</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- D. QUEUE FILTERS ---
        fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1.8, 2.5])
        with fc1:
            p_filter = st.selectbox("Priority Level", ["All", "High", "Medium", "Low"], key="al_p_filter")
        with fc2:
            s_filter = st.selectbox("Review Status", ["All", "Pending", "Reviewed", "Ambiguous / Second Review"], key="al_s_filter")
        with fc3:
            c_filter = st.selectbox(
                "Condition Filter",
                ["All", "Model Disagreement Only", "Uncertain Only", "Missing Fields Only"],
                key="al_c_filter"
            )
        with fc4:
            search_txt = st.text_input("Search Sentence or ID", placeholder="Filter by keyword...", key="al_search_txt")

        # Retrieve filtered records
        filtered_records = queue_obj.get_filtered_queue(
            sort_by=sort_order,
            priority_filter=p_filter,
            status_filter=s_filter,
            condition_filter=c_filter,
            search_query=search_txt,
        )

        st.caption(f"Showing **{len(filtered_records)}** of **{len(queue_obj.records)}** prioritized records (sorted by {sort_order}).")

        # --- C & E. PRIORITIZED ANNOTATION QUEUE CARDS & ACTIONS ---
        if not filtered_records:
            st.info("No records match the current filter criteria.")
        else:
            # Display paginated or top items
            page_size = 15
            total_pages = max(1, (len(filtered_records) + page_size - 1) // page_size)
            page = 1
            if total_pages > 1:
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="al_page_num")

            start_idx = (page - 1) * page_size
            page_records = filtered_records[start_idx : start_idx + page_size]

            for prec in page_records:
                # Color code priority
                if prec.priority_level == "High":
                    p_badge = '<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.72rem;">HIGH PRIORITY</span>'
                elif prec.priority_level == "Medium":
                    p_badge = '<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.72rem;">MEDIUM</span>'
                else:
                    p_badge = '<span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.72rem;">LOW</span>'

                disag_badge = ""
                if prec.disagreement_score > 0:
                    disag_badge = '<span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem;">⚡ MODEL CONFLICT</span>'

                status_badge = f'<span style="color: #94a3b8; font-size: 0.75rem;">Status: <strong style="color: #cbd5e1;">{prec.human_status}</strong></span>'

                with st.container(border=True):
                    card_c1, card_c2 = st.columns([4, 1.2])
                    with card_c1:
                        st.markdown(f"""
                        <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.35rem;">
                            <strong style="color: #38bdf8; font-size: 0.85rem;">Item #{prec.record_id}</strong>
                            {p_badge}
                            {disag_badge}
                            <span style="font-size: 0.75rem; color: #64748b;">• Score: <strong>{prec.priority_score:.2f}</strong></span>
                            {status_badge}
                        </div>
                        <div style="font-size: 0.95rem; color: #f1f5f9; font-weight: 500; margin-bottom: 0.4rem; line-height: 1.4;">
                            "{prec.text}"
                        </div>
                        <div style="font-size: 0.78rem; color: #fbbf24; margin-bottom: 0.3rem;">
                            <strong>Why Prioritized:</strong> {prec.primary_reason}
                        </div>
                        """, unsafe_allow_html=True)

                    with card_c2:
                        # Deep-link Action: Open item directly in existing Annotation Workspace
                        # Find index of this record_id in the original dataframe
                        id_matches = df.index[df["id"].astype(str) == prec.record_id].tolist()
                        target_row_idx = id_matches[0] if id_matches else 0

                        if st.button("✏️ Annotate Item", key=f"btn_annot_{prec.record_id}", use_container_width=True, type="primary"):
                            st.session_state.idx = target_row_idx
                            st.query_params["view"] = "Annotate"
                            st.rerun()

                    # Deep Dive Collapsible Details
                    with st.expander(f"🔍 Detailed Model Predictions & Rationale for #{prec.record_id}"):
                        dcol1, dcol2 = st.columns(2)
                        with dcol1:
                            st.markdown("**Baseline vs Transformer Comparison:**")
                            if prec.disagreement_report:
                                for t_name, td in prec.disagreement_report.tasks.items():
                                    b_lbl = f"{td.baseline_label} ({td.baseline_confidence:.0%})"
                                    t_lbl = f"{td.transformer_label} ({td.transformer_confidence:.0%})" if td.transformer_label else "Unavailable"
                                    conflict_icon = "❌ CONFLICT" if td.is_conflict else ("⚠️ FALLBACK" if td.is_fallback else "✅ AGREE")
                                    st.markdown(f"- **{t_name.replace('_', ' ').capitalize()}**: {conflict_icon}<br>&nbsp;&nbsp;&nbsp;&nbsp;Baseline: `{b_lbl}` vs Transformer: `{t_lbl}`", unsafe_allow_html=True)
                            else:
                                st.caption("No disagreement report available.")

                        with dcol2:
                            st.markdown("**Uncertainty & Audit Explanations:**")
                            for exp in prec.explanations:
                                st.markdown(f"- {exp}")

    # =========================================================================
    # TAB 2: HUMAN FEEDBACK & ANALYTICS (PHASE 6 & 7)
    # =========================================================================
    with tab_feedback:
        st.markdown("### 📈 Human Feedback Collection & Model Correction Analytics")
        st.caption("Captures instances where human annotators modified or verified model advisory suggestions. Human-approved labels remain the ground truth.")

        fb_records = feedback_mgr.load_feedback_records(dataset_key)
        all_fb = feedback_mgr.load_feedback_records("all")
        active_fb = fb_records if fb_records else all_fb
        analytics = compute_feedback_analytics(active_fb)

        fbc1, fbc2, fbc3, fbc4 = st.columns(4)
        with fbc1:
            st.metric("Total Feedback Items", analytics["total_feedback_count"])
        with fbc2:
            st.metric("Human Corrections", analytics["correction_count"], help="Instances where human label differed from model suggestion")
        with fbc3:
            st.metric("Baseline Corrections", analytics["baseline_corrections"], help="Corrections made against baseline heuristic predictions")
        with fbc4:
            st.metric("Transformer Corrections", analytics["transformer_corrections"], help="Corrections made against genuine transformer predictions")

        if analytics["total_feedback_count"] == 0:
            st.info("💡 No feedback records captured yet. As annotators modify suggestions and save records in the Annotation Workspace, corrections are tracked automatically.")
        else:
            st.markdown("---")
            fc_col1, fc_col2 = st.columns(2)
            with fc_col1:
                st.markdown("#### Most Frequently Corrected Classes")
                st.caption("Model predicted class vs human annotator correction.")
                if analytics["top_corrected_classes"]:
                    top_df = pd.DataFrame(analytics["top_corrected_classes"], columns=["Model Predicted Class", "Correction Count"])
                    st.dataframe(top_df, use_container_width=True)
                else:
                    st.caption("No class corrections recorded.")

            with fc_col2:
                st.markdown("#### Error Patterns (Transitions)")
                st.caption("Common transition pairs from model prediction to human label.")
                if analytics["error_patterns"]:
                    ep_df = pd.DataFrame(analytics["error_patterns"], columns=["Prediction Transition", "Frequency"])
                    st.dataframe(ep_df, use_container_width=True)
                else:
                    st.caption("No transition patterns recorded.")

            st.markdown("---")
            st.markdown("#### Export Human Feedback Data")
            st.caption("Safely export isolated feedback history without mutating or touching the original annotation dataset.")

            csv_data = feedback_mgr.export_feedback_csv(active_fb)
            jsonl_data = "\n".join(import_json_dumps(r.to_dict()) for r in active_fb)

            exp_col1, exp_col2, _ = st.columns([1.5, 1.5, 3])
            with exp_col1:
                st.download_button(
                    label="📥 Download Feedback CSV",
                    data=csv_data.encode("utf-8-sig"),
                    file_name=f"feedback_{dataset_key}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with exp_col2:
                st.download_button(
                    label="📥 Download Feedback JSONL",
                    data=jsonl_data.encode("utf-8"),
                    file_name=f"feedback_{dataset_key}.jsonl",
                    mime="application/jsonlines",
                    use_container_width=True,
                )

    # =========================================================================
    # TAB 3: CONTROLLED LEARNING PREPARATION (PHASE 8)
    # =========================================================================
    with tab_prep:
        st.markdown("### 📦 Controlled Learning Dataset Preparation")
        st.caption("Transforms human-verified annotations and feedback into clean, schema-validated training splits with deterministic fingerprinting and cross-split leakage checks.")

        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #fbbf24; padding: 0.75rem 1rem; border-radius: 6px; margin: 0.5rem 0 1rem 0; font-size: 0.85rem; color: #cbd5e1; line-height: 1.5;">
            <strong>⚠️ Controlled Execution Notice:</strong> This module solely validates, audits, partitions, and exports clean datasets. Per LinguaLabel engineering protocols, <strong>no models are automatically fine-tuned or retrained in this phase</strong>.
        </div>
        """, unsafe_allow_html=True)

        # Prepare verified training samples from active dataset and feedback
        prep_source = st.radio(
            "Source Data for Training Preparation",
            ["Active Dataset Human Annotations (Reviewed only)", "Human Feedback Corrections"],
            horizontal=True,
        )

        target_task = st.selectbox(
            "Target Training Task",
            ["sentiment", "text_type", "language"],
            format_func=lambda x: f"{x.replace('_', ' ').capitalize()} Classification",
        )

        clean_prep_rows = []
        if prep_source == "Active Dataset Human Annotations (Reviewed only)":
            reviewed_mask = df["review_status"].astype(str).str.strip().eq("Reviewed — looks correct")
            reviewed_df = df[reviewed_mask]
            for _, row in reviewed_df.iterrows():
                lbl = str(row.get(target_task, "")).strip()
                if lbl and lbl not in ("Uncertain", "— Select —", ""):
                    clean_prep_rows.append({
                        "record_id": str(row.get("id", "")),
                        "text": str(row.get("text", "")).strip(),
                        "label": lbl,
                        "task": target_task,
                        "dataset_key": dataset_key,
                        "source": "reviewed_annotation",
                    })
            candidate_df = pd.DataFrame(clean_prep_rows)
        else:
            fb_list = feedback_mgr.load_feedback_records(dataset_key) or feedback_mgr.load_feedback_records("all")
            candidate_df, fb_warnings = format_feedback_for_training(fb_list, task=target_task)
            for w in fb_warnings:
                st.caption(f"• {w}")

        st.markdown(f"**Found {len(candidate_df)} valid human-verified samples for '{target_task}'**.")

        if len(candidate_df) < 3:
            st.warning("Insufficient verified samples to create train/val/test splits (minimum 3 required). Please review and approve more sentences in the Annotation Workspace.")
        else:
            sp_c1, sp_c2, sp_c3 = st.columns(3)
            with sp_c1:
                tr_ratio = st.slider("Train Split Ratio", 0.50, 0.90, 0.70, 0.05)
            with sp_c2:
                val_ratio = st.slider("Val Split Ratio", 0.05, 0.30, 0.15, 0.05)
            with sp_c3:
                test_ratio = round(1.0 - tr_ratio - val_ratio, 2)
                st.metric("Test Split Ratio", f"{test_ratio:.2f}")

            split_res = prepare_controlled_split(
                candidate_df,
                train_ratio=tr_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=42,
            )

            # Display Split Audit Card
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong>Dataset Fingerprint (SHA-256):</strong>
                    <code style="background: rgba(14, 165, 233, 0.15); color: #38bdf8; padding: 2px 6px; border-radius: 4px;">{split_res.fingerprint}</code>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 0.5rem;">
                    <div>• <strong>Train Samples:</strong> {split_res.train_count} ({tr_ratio:.0%})</div>
                    <div>• <strong>Validation Samples:</strong> {split_res.val_count} ({val_ratio:.0%})</div>
                    <div>• <strong>Test Samples:</strong> {split_res.test_count} ({test_ratio:.0%})</div>
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.8rem; color: {'#f87171' if split_res.leakage_detected else '#4ade80'};">
                    {'❌ Data Leakage Detected between splits!' if split_res.leakage_detected else '✅ Zero Cross-Split Data Leakage Verified.'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Download Split Buttons
            d_col1, d_col2, d_col3 = st.columns(3)
            with d_col1:
                tr_csv = split_res.train.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label=f"📥 Download Train Split ({len(split_res.train)})",
                    data=tr_csv.encode("utf-8-sig"),
                    file_name=f"{target_task}_train_{split_res.fingerprint[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with d_col2:
                val_csv = split_res.val.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label=f"📥 Download Val Split ({len(split_res.val)})",
                    data=val_csv.encode("utf-8-sig"),
                    file_name=f"{target_task}_val_{split_res.fingerprint[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with d_col3:
                test_csv = split_res.test.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    label=f"📥 Download Test Split ({len(split_res.test)})",
                    data=test_csv.encode("utf-8-sig"),
                    file_name=f"{target_task}_test_{split_res.fingerprint[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def import_json_dumps(d: Dict[str, Any]) -> str:
    import json
    return json.dumps(d, ensure_ascii=False)


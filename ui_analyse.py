"""
LinguaLabel — Analyse & Label Workspace (Part 10)

The primary human-in-the-loop sentence analysis interface:
- Prominently showcases the original sentence text.
- Replaces technical ML jargon with intuitive natural language questions:
  1. What language is this?
  2. What does the person feel?
  3. What is the person doing?
- Simplifies confidence into actionable human guidance ("AI is fairly sure", "Needs your attention").
- Provides a grounded, non-hallucinated plain-language explanation of each sentence.
- Features quick actions (Accept All, Save, Mark Uncertain, Skip) with full safety validation.
- Includes an optional collapsible technical inspector for advanced ML researchers.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st

from data_manager import save_dataset_record, validation_issues
from ml.active_learning import FeedbackManager
from ml.base import PredictionResult
from ml.engine import MLEngine
from ml.model_config import MLBackend
from ui_mappings import (
    DISPLAY_LANGUAGE_OPTIONS,
    DISPLAY_REVIEW_STATUS_OPTIONS,
    DISPLAY_SENTIMENT_OPTIONS,
    DISPLAY_TEXT_TYPE_OPTIONS,
    generate_sentence_explanation,
    simplify_confidence,
    to_display_label,
    to_internal_label,
)


def render_analyse_page(
    root_dir: Path,
    df: pd.DataFrame,
    ml_engine: MLEngine,
    dataset_key: str,
    storage_path: Path,
    show_explanations: bool = True,
):
    """Renders the simplified, beginner-friendly sentence analysis page."""
    if df is None or df.empty:
        st.warning("No active dataset loaded. Please import sentences first.")
        if st.button("📥 Go to Import Sentences"):
            st.query_params["view"] = "Import Sentences"
            st.rerun()
        return

    total = len(df)
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    idx = max(0, min(int(st.session_state.idx), total - 1))
    row = df.iloc[idx]
    row_id = str(row.get("id", f"rec_{idx}"))
    row_text = str(row.get("text", "")).strip()

    # Pre-fetch existing annotations
    curr_lang_internal = str(row.get("language", "")).strip()
    curr_sent_internal = str(row.get("sentiment", "")).strip()
    curr_tt_internal = str(row.get("text_type", "")).strip()
    curr_status_internal = str(row.get("review_status", "")).strip()
    curr_notes = str(row.get("annotator_notes", "")).strip()

    curr_lang_disp = to_display_label("language", curr_lang_internal)
    curr_sent_disp = to_display_label("sentiment", curr_sent_internal)
    curr_tt_disp = to_display_label("text_type", curr_tt_internal)
    curr_status_disp = to_display_label("review_status", curr_status_internal)

    # Top Navigation and Sentence Selector Bar
    t_col1, t_col2 = st.columns([3, 2])
    with t_col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.6rem;">
            <span style="font-size: 0.85rem; font-weight: 700; color: #38bdf8;">SENTENCE #{idx + 1}</span>
            <span style="font-size: 0.75rem; color: #64748b;">(ID: {row_id})</span>
            <span style="background: rgba(255,255,255,0.06); color: #cbd5e1; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem;">
                {curr_status_disp if curr_status_disp != '— Select —' else 'Pending Review'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.progress((idx + 1) / total)
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #94a3b8; margin-top: -0.3rem;">
            <span>Progress</span>
            <strong style="color: #38bdf8;">Sentence {idx + 1} of {total} ({int((idx + 1)/total * 100)}%)</strong>
        </div>
        """, unsafe_allow_html=True)

    # Advisory AI Predictions
    preds: Dict[str, PredictionResult] = {}
    if ml_engine.is_available():
        preds = ml_engine.predict_all(row_text)
        st.session_state[f"ml_preds_{row_id}"] = preds

    # Sentence Hero Card
    st.markdown(f"""
    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 1.5rem; margin: 0.75rem 0 1.25rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
        <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #38bdf8; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
            Original Sentence
        </div>
        <div style="font-size: 1.35rem; font-weight: 600; color: #ffffff; line-height: 1.5; font-family: system-ui, -apple-system, sans-serif;">
            "{row_text}"
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Plain-language explanation box
    if show_explanations and preds:
        explanation_text = generate_sentence_explanation(row_text, preds)
        st.markdown(f"""
        <div style="background: rgba(14, 165, 233, 0.08); border-left: 3px solid #38bdf8; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1.25rem; font-size: 0.88rem; color: #e2e8f0; line-height: 1.5;">
            <strong>💡 AI Explanation:</strong> {explanation_text}
        </div>
        """, unsafe_allow_html=True)

    # Main Form Area: 3 Questions Side-by-Side or Clean 2-Column
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.markdown("### ✍️ Label This Sentence")

        # -------------------------------------------------------------
        # Question 1: What language is this?
        # -------------------------------------------------------------
        p_lang = preds.get("language")
        sug_lang_disp = to_display_label("language", p_lang.label) if p_lang else "Not sure"
        conf_text_lang, conf_color_lang, conf_tip_lang = simplify_confidence(p_lang)

        def cb_accept_lang():
            if sug_lang_disp in DISPLAY_LANGUAGE_OPTIONS:
                st.session_state[f"widget_lang_{row_id}"] = sug_lang_disp

        q1_c1, q1_c2 = st.columns([3.8, 1.2])
        with q1_c1:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.5rem; margin-bottom: 0.2rem;">
                <strong style="font-size: 0.95rem; color: #f1f5f9;">1. What language is this?</strong>
                <span style="font-size: 0.75rem; color: {conf_color_lang}; font-weight: 600;" title="{conf_tip_lang}">
                    AI suggests: <strong>{sug_lang_disp}</strong> • {conf_text_lang}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with q1_c2:
            st.button("Use AI", key=f"btn_use_lang_{row_id}", on_click=cb_accept_lang, use_container_width=True, help=f"Use suggestion '{sug_lang_disp}'")

        lang_opts = ["— Select —"] + DISPLAY_LANGUAGE_OPTIONS
        curr_val_lang = st.session_state.get(f"widget_lang_{row_id}", curr_lang_disp)
        lang_idx = lang_opts.index(curr_val_lang) if curr_val_lang in lang_opts else 0
        sel_lang_disp = st.selectbox(
            "Language Selector",
            lang_opts,
            index=lang_idx,
            key=f"widget_lang_{row_id}",
            label_visibility="collapsed",
        )

        # -------------------------------------------------------------
        # Question 2: What does the person feel?
        # -------------------------------------------------------------
        p_sent = preds.get("sentiment")
        sug_sent_disp = to_display_label("sentiment", p_sent.label) if p_sent else "Not sure"
        conf_text_sent, conf_color_sent, conf_tip_sent = simplify_confidence(p_sent)

        def cb_accept_sent():
            if sug_sent_disp in DISPLAY_SENTIMENT_OPTIONS:
                st.session_state[f"widget_sent_{row_id}"] = sug_sent_disp

        q2_c1, q2_c2 = st.columns([3.8, 1.2])
        with q2_c1:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem; margin-bottom: 0.2rem;">
                <strong style="font-size: 0.95rem; color: #f1f5f9;">2. What does the person feel? (Sentiment)</strong>
                <span style="font-size: 0.75rem; color: {conf_color_sent}; font-weight: 600;" title="{conf_tip_sent}">
                    AI suggests: <strong>{sug_sent_disp}</strong> • {conf_text_sent}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with q2_c2:
            st.button("Use AI", key=f"btn_use_sent_{row_id}", on_click=cb_accept_sent, use_container_width=True, help=f"Use suggestion '{sug_sent_disp}'")

        sent_opts = ["— Select —"] + DISPLAY_SENTIMENT_OPTIONS
        curr_val_sent = st.session_state.get(f"widget_sent_{row_id}", curr_sent_disp)
        sent_idx = sent_opts.index(curr_val_sent) if curr_val_sent in sent_opts else 0
        sel_sent_disp = st.selectbox(
            "Sentiment Selector",
            sent_opts,
            index=sent_idx,
            key=f"widget_sent_{row_id}",
            label_visibility="collapsed",
        )

        # -------------------------------------------------------------
        # Question 3: What is the person doing?
        # -------------------------------------------------------------
        p_tt = preds.get("text_type")
        sug_tt_disp = to_display_label("text_type", p_tt.label) if p_tt else "Not sure"
        conf_text_tt, conf_color_tt, conf_tip_tt = simplify_confidence(p_tt)

        def cb_accept_tt():
            if sug_tt_disp in DISPLAY_TEXT_TYPE_OPTIONS:
                st.session_state[f"widget_tt_{row_id}"] = sug_tt_disp

        q3_c1, q3_c2 = st.columns([3.8, 1.2])
        with q3_c1:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem; margin-bottom: 0.2rem;">
                <strong style="font-size: 0.95rem; color: #f1f5f9;">3. What is the person doing? (Sentence Type)</strong>
                <span style="font-size: 0.75rem; color: {conf_color_tt}; font-weight: 600;" title="{conf_tip_tt}">
                    AI suggests: <strong>{sug_tt_disp}</strong> • {conf_text_tt}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with q3_c2:
            st.button("Use AI", key=f"btn_use_tt_{row_id}", on_click=cb_accept_tt, use_container_width=True, help=f"Use suggestion '{sug_tt_disp}'")

        tt_opts = ["— Select —"] + DISPLAY_TEXT_TYPE_OPTIONS
        curr_val_tt = st.session_state.get(f"widget_tt_{row_id}", curr_tt_disp)
        tt_idx = tt_opts.index(curr_val_tt) if curr_val_tt in tt_opts else 0
        sel_tt_disp = st.selectbox(
            "Text Type Selector",
            tt_opts,
            index=tt_idx,
            key=f"widget_tt_{row_id}",
            label_visibility="collapsed",
        )

    with c_right:
        st.markdown("### 📋 Review Status & Notes")

        status_opts = ["— Select —"] + DISPLAY_REVIEW_STATUS_OPTIONS
        curr_val_status = st.session_state.get(f"widget_status_{row_id}", curr_status_disp)
        stat_idx = status_opts.index(curr_val_status) if curr_val_status in status_opts else 0
        sel_status_disp = st.selectbox(
            "Review Status",
            status_opts,
            index=stat_idx,
            key=f"widget_status_{row_id}",
            help="Mark as 'Reviewed & approved' when you have verified all labels."
        )

        # Annotator Explanatory Notes
        notes_input = st.text_area(
            "Explanatory Notes (required if unsure or flagged)",
            value=curr_notes,
            key=f"widget_notes_{row_id}",
            height=110,
            placeholder="Add a short note explaining why this sentence is uncertain or difficult...",
        )

        # Quick Actions Card: Accept All AI Suggestions via on_click callback
        def cb_accept_all():
            if sug_lang_disp in DISPLAY_LANGUAGE_OPTIONS:
                st.session_state[f"widget_lang_{row_id}"] = sug_lang_disp
            if sug_sent_disp in DISPLAY_SENTIMENT_OPTIONS:
                st.session_state[f"widget_sent_{row_id}"] = sug_sent_disp
            if sug_tt_disp in DISPLAY_TEXT_TYPE_OPTIONS:
                st.session_state[f"widget_tt_{row_id}"] = sug_tt_disp
            st.session_state[f"widget_status_{row_id}"] = "Reviewed & approved"

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        st.button(
            "✨ Accept All AI Suggestions",
            use_container_width=True,
            on_click=cb_accept_all,
            help="Fill the form with the AI suggestions without saving yet",
        )

    # Map selected display values back to canonical internal schema
    internal_vals = {
        "language": to_internal_label("language", sel_lang_disp),
        "sentiment": to_internal_label("sentiment", sel_sent_disp),
        "text_type": to_internal_label("text_type", sel_tt_disp),
        "review_status": to_internal_label("review_status", sel_status_disp),
        "annotator_notes": notes_input.strip(),
    }

    # Validation & Save Function
    def execute_save(show_errors: bool = True) -> bool:
        issues = validation_issues(internal_vals)
        if issues:
            if show_errors:
                st.error("Please address the following items before saving:")
                for iss in issues:
                    # Translate internal column names to beginner-friendly terminology
                    friendly_iss = (
                        iss.replace("text type", "sentence type")
                        .replace("Reviewed — looks correct", "'Reviewed & approved'")
                    )
                    st.markdown(f"- {friendly_iss}")
            return False

        save_ok = save_dataset_record(
            storage_path=storage_path,
            current_df=df,
            row_id=row_id,
            vals=internal_vals,
        )

        if save_ok:
            # Also persist directly into user's database dataset if logged in
            curr_user = st.session_state.get("current_user")
            active_ds_id = st.session_state.get("active_dataset_id")
            if curr_user and active_ds_id:
                try:
                    from db.service import update_sentence_annotation as db_update_sentence
                    db_update_sentence(
                        user_id=curr_user["id"],
                        dataset_id=active_ds_id,
                        sentence_id=str(row_id),
                        values=internal_vals,
                    )
                except Exception:
                    pass

            # Record human feedback if model predictions were present
            if preds:
                try:
                    fb_mgr = FeedbackManager(root_dir)
                    for t_name in ["language", "sentiment", "text_type"]:
                        if t_name in preds and internal_vals.get(t_name):
                            fb_mgr.record_feedback(
                                dataset_key=dataset_key,
                                record_id=row_id,
                                task=t_name,
                                original_text=row_text,
                                model_prediction=preds[t_name],
                                human_label=internal_vals[t_name],
                                annotator_notes=internal_vals.get("annotator_notes", ""),
                            )
                            if curr_user and active_ds_id:
                                try:
                                    from db.service import record_feedback as db_record_feedback
                                    db_record_feedback(
                                        user_id=curr_user["id"],
                                        dataset_id=active_ds_id,
                                        sentence_id=str(row_id),
                                        task=t_name,
                                        original_prediction=preds[t_name].to_dict(),
                                        corrected_label=internal_vals[t_name],
                                        annotator_notes=internal_vals.get("annotator_notes", ""),
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass

            st.session_state.msg = f"Saved sentence #{idx + 1}."
            return True
        else:
            st.error("Could not write annotation to disk. Please check storage permissions.")
            return False

    # Action Bar: Save & Navigation
    st.markdown("---")
    act_col1, act_col2, act_col3, act_col4 = st.columns([1, 1, 1.6, 1.6])

    with act_col1:
        if st.button("← Previous", disabled=(idx == 0), use_container_width=True):
            execute_save(show_errors=False)
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    with act_col2:
        if st.button("Next →", disabled=(idx == total - 1), use_container_width=True):
            execute_save(show_errors=False)
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()

    with act_col3:
        if st.button("💾 Save Annotation", type="primary", use_container_width=True):
            if execute_save(show_errors=True):
                st.rerun()

    with act_col4:
        if st.button("⏩ Go to Next Unreviewed", use_container_width=True):
            execute_save(show_errors=False)
            pending_idxs = df.index[df["review_status"].astype(str).str.strip().eq("")].tolist()
            future_candidates = [i for i in pending_idxs if i > idx]
            next_target = future_candidates[0] if future_candidates else (pending_idxs[0] if pending_idxs else None)
            if next_target is not None:
                st.session_state.idx = next_target
                st.rerun()
            else:
                st.success("🎉 All sentences in this dataset have been reviewed!")

    # Optional Technical Expander (Research & Engineering Mode)
    with st.expander("🔍 Show Technical Prediction Details (for ML Researchers)"):
        st.caption("Inspect the exact mathematical model checkpoint, backend provenance, and raw probabilities.")
        if not preds:
            st.write("No technical prediction data available.")
        else:
            for task_name, pred_obj in preds.items():
                st.markdown(f"**{task_name.capitalize()}**: `{pred_obj.label}`")
                st.markdown(f"- Confidence: `{pred_obj.confidence:.4f}` ({pred_obj.confidence_type})")
                st.markdown(f"- Method: `{pred_obj.method}` | Actual Backend: `{pred_obj.actual_backend}`")
                if "all_scores" in pred_obj.metadata:
                    st.write("Score Distribution:", pred_obj.metadata["all_scores"])


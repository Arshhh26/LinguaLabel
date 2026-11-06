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
    DISPLAY_ABUSE_OPTIONS,
    DISPLAY_LANGUAGE_OPTIONS,
    DISPLAY_REVIEW_STATUS_OPTIONS,
    DISPLAY_SENTIMENT_OPTIONS,
    DISPLAY_SEVERITY_OPTIONS,
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
    curr_abuse_internal = str(row.get("abuse_category", "")).strip()
    curr_sev_internal = str(row.get("abuse_severity", "")).strip()
    curr_status_internal = str(row.get("review_status", "")).strip()
    curr_notes = str(row.get("annotator_notes", "")).strip()

    curr_lang_disp = to_display_label("language", curr_lang_internal)
    curr_sent_disp = to_display_label("sentiment", curr_sent_internal)
    curr_tt_disp = to_display_label("text_type", curr_tt_internal)
    curr_abuse_disp = to_display_label("abuse", curr_abuse_internal)
    curr_sev_disp = curr_sev_internal if curr_sev_internal in DISPLAY_SEVERITY_OPTIONS else ("None" if curr_abuse_disp == "Not abusive" else "— Select —")
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

        # Anonymous aggregate telemetry: count sentence analysis once per row in this session
        tracked_analysis_key = f"_tracked_analysis_{row_id}"
        if not st.session_state.get(tracked_analysis_key, False):
            st.session_state[tracked_analysis_key] = True
            try:
                from db.analytics import record_usage_event
                record_usage_event(
                    "analysis",
                    "sentence_analysis",
                    count=1,
                    metadata={"task": "predict_all"},
                    visitor_id=st.session_state.get("_visitor_id"),
                    session_id=st.session_state.get("_session_id"),
                )
            except Exception:
                pass

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

        lang_key = f"widget_lang_{row_id}"
        lang_opts = ["— Select —"] + DISPLAY_LANGUAGE_OPTIONS
        if lang_key not in st.session_state:
            st.session_state[lang_key] = curr_lang_disp if curr_lang_disp in lang_opts else "— Select —"

        def cb_accept_lang():
            if sug_lang_disp in DISPLAY_LANGUAGE_OPTIONS:
                st.session_state[lang_key] = sug_lang_disp

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
            st.button("Use AI", key=f"btn_use_lang_{row_id}", on_click=cb_accept_lang, width="stretch", help=f"Use suggestion '{sug_lang_disp}'")

        sel_lang_disp = st.selectbox(
            "Language Selector",
            lang_opts,
            key=lang_key,
            label_visibility="collapsed",
        )

        # -------------------------------------------------------------
        # Question 2: What does the person feel?
        # -------------------------------------------------------------
        p_sent = preds.get("sentiment")
        sug_sent_disp = to_display_label("sentiment", p_sent.label) if p_sent else "Not sure"
        conf_text_sent, conf_color_sent, conf_tip_sent = simplify_confidence(p_sent)

        sent_key = f"widget_sent_{row_id}"
        sent_opts = ["— Select —"] + DISPLAY_SENTIMENT_OPTIONS
        if sent_key not in st.session_state:
            st.session_state[sent_key] = curr_sent_disp if curr_sent_disp in sent_opts else "— Select —"

        def cb_accept_sent():
            if sug_sent_disp in DISPLAY_SENTIMENT_OPTIONS:
                st.session_state[sent_key] = sug_sent_disp

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
            st.button("Use AI", key=f"btn_use_sent_{row_id}", on_click=cb_accept_sent, width="stretch", help=f"Use suggestion '{sug_sent_disp}'")

        sel_sent_disp = st.selectbox(
            "Sentiment Selector",
            sent_opts,
            key=sent_key,
            label_visibility="collapsed",
        )

        # -------------------------------------------------------------
        # Question 3: What is the person doing?
        # -------------------------------------------------------------
        p_tt = preds.get("text_type")
        sug_tt_disp = to_display_label("text_type", p_tt.label) if p_tt else "Not sure"
        conf_text_tt, conf_color_tt, conf_tip_tt = simplify_confidence(p_tt)

        tt_key = f"widget_tt_{row_id}"
        tt_opts = ["— Select —"] + DISPLAY_TEXT_TYPE_OPTIONS
        if tt_key not in st.session_state:
            st.session_state[tt_key] = curr_tt_disp if curr_tt_disp in tt_opts else "— Select —"

        def cb_accept_tt():
            if sug_tt_disp in DISPLAY_TEXT_TYPE_OPTIONS:
                st.session_state[tt_key] = sug_tt_disp

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
            st.button("Use AI", key=f"btn_use_tt_{row_id}", on_click=cb_accept_tt, width="stretch", help=f"Use suggestion '{sug_tt_disp}'")

        sel_tt_disp = st.selectbox(
            "Text Type Selector",
            tt_opts,
            key=tt_key,
            label_visibility="collapsed",
        )

        # -------------------------------------------------------------
        # Question 4: Is there abusive language or profanity? (Part 11.2)
        # -------------------------------------------------------------
        p_abuse = preds.get("abuse_category") or preds.get("abuse")
        p_sev = preds.get("abuse_severity")
        sug_abuse_disp = to_display_label("abuse", p_abuse.label) if p_abuse else "Not sure"
        if p_sev and p_sev.label in DISPLAY_SEVERITY_OPTIONS:
            sug_sev_disp = p_sev.label
        elif p_abuse and p_abuse.metadata and p_abuse.metadata.get("severity"):
            sug_sev_disp = p_abuse.metadata.get("severity", "None")
        else:
            sug_sev_disp = "None"
        conf_text_abuse, conf_color_abuse, conf_tip_abuse = simplify_confidence(p_abuse)

        abuse_key = f"widget_abuse_{row_id}"
        abuse_opts = ["— Select —"] + DISPLAY_ABUSE_OPTIONS
        if abuse_key not in st.session_state:
            st.session_state[abuse_key] = curr_abuse_disp if curr_abuse_disp in abuse_opts else "— Select —"

        sev_key = f"widget_sev_{row_id}"
        sev_opts = DISPLAY_SEVERITY_OPTIONS
        if sev_key not in st.session_state:
            st.session_state[sev_key] = curr_sev_disp if curr_sev_disp in sev_opts else ("None" if st.session_state[abuse_key] == "Not abusive" else "Mild")

        def cb_accept_abuse():
            if sug_abuse_disp in DISPLAY_ABUSE_OPTIONS:
                st.session_state[abuse_key] = sug_abuse_disp
            if sug_sev_disp in DISPLAY_SEVERITY_OPTIONS:
                st.session_state[sev_key] = sug_sev_disp

        q4_c1, q4_c2 = st.columns([3.8, 1.2])
        with q4_c1:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.75rem; margin-bottom: 0.2rem;">
                <strong style="font-size: 0.95rem; color: #f1f5f9;">4. Is there abusive language or profanity?</strong>
                <span style="font-size: 0.75rem; color: {conf_color_abuse}; font-weight: 600;" title="{conf_tip_abuse}">
                    AI suggests: <strong>{sug_abuse_disp}</strong> • {conf_text_abuse}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with q4_c2:
            st.button("Use AI", key=f"btn_use_abuse_{row_id}", on_click=cb_accept_abuse, width="stretch", help=f"Use suggestion '{sug_abuse_disp}' ({sug_sev_disp})")

        # Multi-label tags display if additional categories detected
        if p_abuse and p_abuse.metadata and p_abuse.metadata.get("additional_categories"):
            add_cats = p_abuse.metadata["additional_categories"]
            pills = " ".join([f"<span style='background: rgba(239, 68, 68, 0.2); color: #fca5a5; padding: 2px 7px; border-radius: 4px; font-size: 0.74rem;'>+ {c}</span>" for c in add_cats])
            st.markdown(f"<div style='margin-bottom: 0.4rem;'>{pills}</div>", unsafe_allow_html=True)

        ab_col1, ab_col2 = st.columns([3, 2])
        with ab_col1:
            sel_abuse_disp = st.selectbox(
                "Abuse Category Selector",
                abuse_opts,
                key=abuse_key,
                label_visibility="collapsed",
            )
        with ab_col2:
            # If user selected 'Not abusive', automatically sync severity to 'None' before selectbox instantiates
            if sel_abuse_disp == "Not abusive":
                st.session_state[sev_key] = "None"
            sel_sev_disp = st.selectbox(
                "Severity",
                sev_opts,
                key=sev_key,
                label_visibility="collapsed",
            )

    with c_right:
        st.markdown("### 📋 Review Status & Notes")

        status_key = f"widget_status_{row_id}"
        status_opts = ["— Select —"] + DISPLAY_REVIEW_STATUS_OPTIONS
        if status_key not in st.session_state:
            st.session_state[status_key] = curr_status_disp if curr_status_disp in status_opts else "— Select —"

        sel_status_disp = st.selectbox(
            "Review Status",
            status_opts,
            key=status_key,
            help="Mark as 'Reviewed & approved' when you have verified all labels."
        )

        # Annotator Explanatory Notes
        notes_key = f"widget_notes_{row_id}"
        if notes_key not in st.session_state:
            st.session_state[notes_key] = curr_notes

        notes_input = st.text_area(
            "Explanatory Notes (required if unsure or flagged)",
            key=notes_key,
            height=110,
            placeholder="Add a short note explaining why this sentence is uncertain or difficult...",
        )

        # Quick Actions Card: Accept All AI Suggestions via on_click callback
        def cb_accept_all():
            if sug_lang_disp in DISPLAY_LANGUAGE_OPTIONS:
                st.session_state[lang_key] = sug_lang_disp
            if sug_sent_disp in DISPLAY_SENTIMENT_OPTIONS:
                st.session_state[sent_key] = sug_sent_disp
            if sug_tt_disp in DISPLAY_TEXT_TYPE_OPTIONS:
                st.session_state[tt_key] = sug_tt_disp
            if sug_abuse_disp in DISPLAY_ABUSE_OPTIONS:
                st.session_state[abuse_key] = sug_abuse_disp
            if sug_sev_disp in DISPLAY_SEVERITY_OPTIONS:
                st.session_state[sev_key] = sug_sev_disp
            st.session_state[status_key] = "Reviewed & approved"
            try:
                from db.analytics import record_usage_event
                record_usage_event("interaction", "accept_ai_suggestions", count=1, visitor_id=st.session_state.get("_visitor_id"), session_id=st.session_state.get("_session_id"))
            except Exception:
                pass

        st.markdown("<div style='margin-top: 0.75rem;'></div>", unsafe_allow_html=True)
        st.button(
            "✨ Accept All AI Suggestions",
            width="stretch",
            on_click=cb_accept_all,
            help="Fill the form with the AI suggestions without saving yet",
        )

    # Map selected display values back to canonical internal schema
    internal_vals = {
        "language": to_internal_label("language", sel_lang_disp),
        "sentiment": to_internal_label("sentiment", sel_sent_disp),
        "text_type": to_internal_label("text_type", sel_tt_disp),
        "abuse_category": to_internal_label("abuse", sel_abuse_disp),
        "abuse_severity": sel_sev_disp,
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
            try:
                from db.analytics import record_usage_event
                record_usage_event("interaction", "save_annotation", count=1, visitor_id=st.session_state.get("_visitor_id"), session_id=st.session_state.get("_session_id"))
            except Exception:
                pass

            # Record human feedback if model predictions were present
            if preds:
                try:
                    fb_mgr = FeedbackManager(root_dir)
                    for t_name in ["language", "sentiment", "text_type", "abuse_category"]:
                        pred_key = "abuse" if t_name == "abuse_category" else t_name
                        if pred_key in preds and internal_vals.get(t_name):
                            fb_mgr.record_feedback(
                                dataset_key=dataset_key,
                                record_id=row_id,
                                task=t_name,
                                original_text=row_text,
                                model_prediction=preds[pred_key],
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
                                        original_prediction=preds[pred_key].to_dict(),
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
        if st.button("← Previous", disabled=(idx == 0), width="stretch"):
            execute_save(show_errors=False)
            st.session_state.idx = max(0, idx - 1)
            st.rerun()

    with act_col2:
        if st.button("Next →", disabled=(idx == total - 1), width="stretch"):
            execute_save(show_errors=False)
            st.session_state.idx = min(total - 1, idx + 1)
            st.rerun()

    with act_col3:
        if st.button("💾 Save Annotation", type="primary", width="stretch"):
            if execute_save(show_errors=True):
                st.rerun()

    with act_col4:
        if st.button("⏩ Go to Next Unreviewed", width="stretch"):
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


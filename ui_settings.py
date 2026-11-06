"""
LinguaLabel — Settings & Research Mode (Part 10)

Provides:
- User experience controls: Standard Mode vs Research Mode toggle.
- Plain-language AI explanation visibility toggle.
- Analysis backend selection (Baseline Heuristics vs Multilingual Transformer).
- Safety threshold adjustment (default 60%).
- Full access to Empirical Model Benchmarks (Part 8) and Advanced Active Learning (Part 9)
  when Research Mode is active.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from dashboard import render_model_performance_dashboard
from active_learning_dashboard import render_active_learning_dashboard
from ui_training import render_training_center
from ml.engine import MLEngine
from ml.model_config import MLBackend


def render_settings_page(
    root_dir: Path,
    df: pd.DataFrame,
    ml_engine: MLEngine,
    dataset_key: str,
    storage_path: Path,
):
    """Renders the Settings and Research Mode interface."""
    st.markdown("""
    <div class="header-container" style="margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
            <span class="header-badge" style="background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(203, 213, 225, 0.3);">
                ⚙️ PREFERENCES & MODES
            </span>
        </div>
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; letter-spacing: -0.02em;">
            Settings & Research Mode
        </h1>
        <div class="header-subtitle" style="color: #94a3b8; font-size: 0.9rem;">
            Customize your analysis preferences, control AI suggestions, or switch into Research Mode for deep machine-learning diagnostics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode Selector
    st.markdown("### 🎛️ Application Operating Mode")
    current_mode = st.session_state.get("app_mode", "Standard Mode (Beginner-Friendly)")
    mode_options = [
        "Standard Mode (Beginner-Friendly)",
        "Research Mode (Advanced NLP & Model Benchmarking)",
    ]
    selected_mode = st.radio(
        "Application Mode",
        mode_options,
        index=0 if current_mode.startswith("Standard") else 1,
        help="Standard mode focuses on plain-language annotation. Research mode unlocks empirical confusion matrices, F1 benchmarks, and active-learning split generators."
    )
    st.session_state.app_mode = selected_mode

    st.markdown("---")

    # =========================================================================
    # SECTION A: STANDARD PREFERENCES
    # =========================================================================
    st.markdown("### ⚙️ Analysis & Interface Preferences")
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.session_state.show_explanations = st.toggle(
            "Show Plain-Language AI Explanations",
            value=st.session_state.get("show_explanations", True),
            help="Displays a natural-language sentence explanation describing what the sentence expresses."
        )

        st.session_state.enable_ai_suggestions = st.toggle(
            "Enable Advisory AI Suggestions",
            value=st.session_state.get("enable_ai_suggestions", True),
            help="Provides real-time advisory suggestions for language, sentiment, and sentence type."
        )

    with p_col2:
        backend_choice = st.selectbox(
            "AI Model Backend",
            options=["Baseline Heuristics (Fast, rule-based)", "Multilingual Transformer (XLM-R & mDeBERTa-v3)"],
            index=1 if ml_engine.active_backend == MLBackend.TRANSFORMER else 0,
            help="Choose between fast dictionary rules or deep transformer cross-lingual models."
        )
        if "Transformer" in backend_choice:
            ml_engine.set_backend(MLBackend.TRANSFORMER)
        else:
            ml_engine.set_backend(MLBackend.BASELINE)

        st.caption(f"Active Backend: `{ml_engine.active_backend.value}` (Device: `{ml_engine.get_backend_info().get('device', 'cpu')}`)")

    st.markdown("---")

    # =========================================================================
    # SECTION B: RESEARCH MODE WORKSPACES (PART 8 & PART 9)
    # =========================================================================
    if "Research Mode" in selected_mode:
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); border-left: 3px solid #818cf8; padding: 0.85rem 1.1rem; border-radius: 6px; margin-bottom: 1.25rem; font-size: 0.88rem; color: #cbd5e1;">
            <strong>🔬 Research Mode Active:</strong> You have unlocked LinguaLabel's empirical model evaluation suite, confusion matrices, Shannon entropy sampling, and dataset partition preparation tools.
        </div>
        """, unsafe_allow_html=True)

        tabs = [
            "📊 Model Benchmarks & Comparison (Part 8)",
            "🎯 Advanced Active Learning & Split Preparation (Part 9)",
        ]
        is_owner = st.session_state.get("is_owner_authenticated", False)
        if is_owner:
            tabs.append("🧠 Model Training & Improvement Center (Part 11.3)")

        res_tabs = st.tabs(tabs)

        with res_tabs[0]:
            render_model_performance_dashboard(root_dir, df)

        with res_tabs[1]:
            render_active_learning_dashboard(root_dir, df, ml_engine, dataset_key, storage_path)

        if is_owner:
            with res_tabs[2]:
                render_training_center(root_dir, ml_engine)
        else:
            st.caption("🔒 Model Training & Continuous Improvement Center is reserved for the website owner. Authenticate in Owner Analytics to unlock.")
    else:
        st.info("💡 To access side-by-side model accuracy comparisons, confusion matrices, and active-learning split generators, toggle to **Research Mode** above.")


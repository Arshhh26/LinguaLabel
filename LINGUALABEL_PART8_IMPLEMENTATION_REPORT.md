
   - Any transformer inference that falls back to baseline is explicitly labeled as `fallback_baseline` and excluded from pure transformer metrics.
2. **Prominent Small-Sample Guardrails (Phase 3 Compliance):**
   - When evaluated on limited datasets (such as `annotated_output.csv` with $N=12$), the dashboard displays an unambiguous alert:
     > **Preliminary Evaluation Notice:** Small sample size ($N=12$). Results are not sufficient to establish general model superiority. Expand annotated ground truth to $\ge 50$ samples per class for statistical significance.
3. **Paired Baseline vs. Transformer Evaluation:**
   - Both models are evaluated on the exact same ground-truth records.
   - Provides architectural insights explaining why the multilingual transformer dominates on **Sentiment Polarity** (+16.66% gain), while the syntactic baseline currently outperforms on **Text Type / Intent** (91.67% vs. 50.00%) due to unambiguous punctuation/keyword cues versus zero-shot intent diffusion under the 0.60 uncertainty threshold.
4. **Confidence Distribution Separation:**
   - Heuristic rule scores and raw transformer softmax probabilities are kept in separate charts and never conflated into a single misleading probability distribution.
5. **Persistent Benchmark History & Isolation:**
   - Benchmark evaluations are saved as structured JSON in `benchmark_history/`.
   - Custom uploaded evaluation datasets are isolated in `evaluation_datasets/`, guaranteeing zero cross-contamination with annotation datasets (`annotated_output.csv` or `sample_data.csv`).
6. **Multi-Format Export Suite:**
   - 1-click downloads for Comparative Metrics CSV, Record-Level Predictions CSV, Confusion Matrix CSV, and Comprehensive Markdown Reports.

---

## 2. Dashboard Structure & Layout

The dashboard integrates into LinguaLabel's sidebar navigation (`"Model performance"`) using the established Linear/Vercel-inspired dark SaaS aesthetic:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  🌐 LINGUALABEL STUDIO — MODEL PERFORMANCE & QUALITY DASHBOARD                         │
│  [📊 Empirical Evaluation Studio]  [⚡ GPU Accelerated: NVIDIA RTX 4050 Laptop GPU]    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  📂 EVALUATION DATASET & CONTROLS                                                      │
│  Dataset Selector: [annotated_output.csv (12 labeled)] | [🚀 Run Benchmark Evaluation] │
│  ⚠️ Preliminary Evaluation Notice: Small sample size (N=12). Results inconclusive.     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  📈 OVERVIEW METRICS                                                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ Total Records│ │ Evaluated (N)│ │ Tasks Tested │ │ Device Target│ │ TF Inferences│  │
│  │      20      │ │      12      │ │      2       │ │ cuda:0 (GPU) │ │  24 Genuine  │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  ⚖️ SIDE-BY-SIDE MODEL COMPARISON                                                      │
│  [Sentiment Polarity Tab]  [Text Type / Intent Tab]  [Language Identification Tab]     │
│  • Metric Scorecards: Accuracy, Macro F1, Weighted F1, Macro Prec/Rec, Latency        │
│  • Grouped Bar Chart (Altair): Baseline vs. Transformer metrics                        │
│  • Architecture Explanatory Callouts                                                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  🔍 DEEP-DIVE PERFORMANCE & ERROR ANALYSIS                                             │
│  • Side-by-Side Confusion Matrices (Actual vs. Predicted Heatmaps)                     │
│  • Per-Class Metrics Table (Precision, Recall, F1, Support for each class)             │
│  • Multilingual Breakdown (English vs. Hindi, with N < 10 sample guardrails)           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  🎯 PREDICTION CONFIDENCE ANALYSIS                                                     │
│  • Baseline Heuristic Score Distribution (0.0 to 1.0)                                  │
│  • Transformer Raw Softmax Probability Distribution (0.0 to 1.0)                       │
│  • Uncertainty Threshold Guardrail (< 0.60 clamp count & percentage)                   │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  🛡️ BACKEND RELIABILITY & PROVENANCE AUDIT                                            │
│  • Model Checkpoint IDs, Checkpoint Licenses (MIT), VRAM and CUDA Device Details       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  📜 BENCHMARK HISTORY & 📤 EXPORT SUITE                                                │
│  • Historical Runs Table & Previous Run Selector                                       │
│  • [Download Metrics CSV] [Download Predictions CSV] [Download Markdown Report]       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Features Implemented

| Feature Area | Implementation Details |
| :--- | :--- |
| **Dataset Ingestion & Validation** | Multi-encoding CSV reading, text column detection, ground-truth label mapping, duplicate text detection, missing and invalid label validation. |
| **Dataset Storage Isolation** | Uploaded evaluation datasets are saved in `evaluation_datasets/`, preserving all original columns with UTF-8 BOM encoding. |
| **Paired Benchmark Execution** | Synchronized inference across both engines on the exact same row subset; prevents training/test leakage and ensures identical support. |
| **Fallback Monitoring** | Fallback predictions (`fallback_baseline`) are detected, logged, and excluded from pure transformer metrics. |
| **Comprehensive Metrics** | Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, and Confusion Matrices. |
| **Multilingual Breakdown** | Evaluates per-language accuracy and macro-F1 (English vs. Devanagari Hindi), displaying warnings for languages with $N < 10$. |
| **Confidence Integrity** | Strict separation of heuristic confidence vs. raw uncalibrated model probabilities; $< 0.60$ uncertainty tracking. |
| **Historical Run Tracking** | Persists benchmark results as JSON in `benchmark_history/`, allowing users to inspect past benchmarks. |
| **Export Capabilities** | Generates formatted CSVs for metrics, predictions, confusion matrices, and human-readable Markdown evaluation reports. |

---

## 4. Files Created and Modified

### Created Files:
1. [`dashboard.py`](file:///home/arshhh/Downloads/LinguaLabel/dashboard.py): The complete Streamlit dashboard module containing overview KPI cards, Altair comparative charts, confusion matrix heatmaps, multilingual breakdowns, confidence histograms, and export controls.
2. [`ml/benchmark.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/benchmark.py): Core benchmark coordinator handling dataset validation, paired inference, metric calculation, history persistence, and CSV/Markdown export formatting.
3. [`tests/test_benchmark_dashboard.py`](file:///home/arshhh/Downloads/LinguaLabel/tests/test_benchmark_dashboard.py): Comprehensive 14-test suite validating all Phase 8 requirements.

### Modified Files:
1. [`app.py`](file:///home/arshhh/Downloads/LinguaLabel/app.py): Added `"Model performance"` to sidebar navigation `view_options` and connected `render_model_performance_dashboard` without altering existing annotation or review logic.
2. [`ml/evaluate.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/evaluate.py): Added `weighted_f1` calculation and detailed `record_predictions` collection to `evaluate_task`.
3. [`.gitignore`](file:///home/arshhh/Downloads/LinguaLabel/.gitignore): Added `evaluation_datasets/` to prevent uploaded evaluation datasets from polluting Git tracking.

---

## 5. Actual Benchmark Calculations (`annotated_output.csv`, $N=12$)

The table below reflects **actual empirical benchmark results** computed dynamically from the ground-truth annotations on local GPU hardware:

### Task 1: Sentiment Polarity ($N=12$)

| Metric | Baseline (Lexicon + Negation Rules) | Transformer (`twitter-xlm-roberta-base-sentiment`) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 41.67% | **58.33%** | **+16.66%** |
| **Macro Precision** | 0.5000 | 0.5000 | 0.0000 |
| **Macro Recall** | 0.2500 | **0.6250** | **+0.3750** |
| **Macro F1-Score** | 0.3333 | **0.5000** | **+0.1667** |
| **Weighted F1-Score** | 0.5556 | **0.5833** | **+0.0278** |
| **Mean Latency / Item** | **0.26 ms** | 263.13 ms | +262.87 ms |
| **English Accuracy ($N=6$)** | 50.0% | **66.7%** | **+16.7%** |
| **Hindi Accuracy ($N=6$)** | 33.3% | **50.0%** | **+16.7%** |

### Task 2: Text Type / Intent Classification ($N=12$)

| Metric | Baseline (Syntactic Cues & Punctuation) | Transformer (`mDeBERTa-v3-base-xnli`) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **91.67%** | 50.00% | -41.67% |
| **Macro Precision** | **0.9583** | 0.6042 | -0.3541 |
| **Macro Recall** | **0.8750** | 0.6250 | -0.2500 |
| **Macro F1-Score** | **0.8939** | 0.5810 | -0.3129 |
| **Weighted F1-Score** | **0.9028** | 0.5833 | -0.3195 |
| **Mean Latency / Item** | **0.33 ms** | 291.93 ms | +291.60 ms |
| **English Accuracy ($N=6$)** | **83.3%** | 50.0% | -33.3% |
| **Hindi Accuracy ($N=6$)** | **100.0%** | 50.0% | -50.0% |

---

## 6. Performance & Latency Observations

1. **Inference Latency Profile:**
   - **Baseline Engine:** Sub-millisecond latency (**0.26 ms to 0.33 ms per sentence**) using Unicode codepoint checks, compiled regex tokenizers, and vectorized dictionary lookups.
   - **Transformer Engine (NVIDIA RTX 4050 GPU):** Active inference requires **50 ms to 290 ms per sentence** depending on sequence length and candidate intent count.
2. **Memory & Process Caching:**
   - Models are cached inside the singleton `ModelLoader`.
   - Repeated user interactions, navigation actions, and filter adjustments in Streamlit never trigger re-downloads or redundant pipeline allocations.
   - Active GPU VRAM footprint: ~1.7 GB total for both models loaded simultaneously on CUDA.

---

## 7. Test Results (88/88 Passing)

The full test suite was executed across all unit, integration, regression, and benchmark test modules:
```bash
PYTHONPATH=. .venv/bin/pytest
```

**Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.14.10, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/arshhh/Downloads/LinguaLabel
plugins: anyio-4.15.1
collected 88 items

tests/test_audit_fixes_regression.py ............                        [ 13%]
tests/test_benchmark_dashboard.py ..............                         [ 29%]
tests/test_degradation.py ......                                         [ 36%]
tests/test_ml_language.py .....                                          [ 42%]
tests/test_ml_sentiment.py .......                                       [ 50%]
tests/test_ml_text_type.py .....                                         [ 55%]
tests/test_ml_transformers.py .................                          [ 75%]
tests/test_part1_integration.py ..                                       [ 77%]
tests/test_upload_and_ids.py .........                                   [ 87%]
tests/test_workflow.py ...........                                       [100%]

======================== 88 passed, 1 warning in 33.54s ========================
```

---

## 8. Remaining Limitations & Recommendations

1. **Ground-Truth Dataset Scale ($N=12$):**
   - The primary evaluation set (`annotated_output.csv`) currently contains only 12 labeled records (6 English, 6 Hindi).
   - *Recommendation:* Prioritize human annotation of the remaining 8 unlabeled records in `annotated_output.csv` and ingest a benchmark corpus of $\ge 50$ samples per class to establish high statistical power.
2. **Transliterated Hinglish Coverage:**
   - The current baseline handles Romanized Hindi (Hinglish) via character n-grams, while the transformer handles it via subword tokenization.
   - *Recommendation:* Ground-truth datasets should introduce explicit `"Hinglish"` tags once dedicated validation criteria are established.
3. **Sequential Pipeline Invocations:**
   - Zero-shot classification operates sequentially on single sentences. For datasets $> 1,000$ sentences, implementing batched inference via Hugging Face `Dataset` objects will further accelerate throughput.

---

## 9. Usage Instructions

### Launch the Application
```bash
# Activate environment
source .venv/bin/activate

# Launch Streamlit workspace
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Access the Model Performance Dashboard
1. In the sidebar navigation under **Workspace**, select **"Model performance"**.
2. Select your evaluation dataset (e.g. `annotated_output.csv` or upload a new CSV).
3. Click **"🚀 Run Benchmark Evaluation"** to trigger live evaluation across both engines.
4. Explore interactive metrics, Altair bar charts, confusion matrices, and language breakdowns.
5. Download comparative CSVs or the Markdown report from the **Export Suite**.


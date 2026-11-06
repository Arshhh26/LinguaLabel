# LinguaLabel — Part 9: Active Learning & Intelligent Annotation Engine
## Implementation & Technical Verification Report

**Date:** November 6, 2026  
**Status:** Complete & Rigorously Verified  
**Active Test Suite:** 102 / 102 Tests Passing (14 new tests added in Part 9)  
**Target Environment:** Python 3.14.10, Linux (Fedora 43), NVIDIA GeForce RTX 4050 GPU (CUDA 13.0)  

---

## 1. Executive Summary & Architectural Overview

Part 9 elevates **LinguaLabel** from an AI-assisted annotation tool into an **Intelligent Human-in-the-Loop Active Learning Platform**. Instead of presenting sentences sequentially in random or arbitrary order, LinguaLabel now actively evaluates each unlabeled sentence with both the statistical baseline (lexicon/heuristics) and deep transformer models (XLM-RoBERTa & mDeBERTa-v3), detects prediction uncertainty and cross-model disagreements, and dynamically routes high-value, ambiguous, or conflict-ridden samples to human annotators.

Crucially, Part 9 adheres to strict empirical engineering standards:
1. **Never Automatically Accepts Predictions:** All model outputs remain purely advisory until explicitly confirmed by human annotators.
2. **Never Fabricates Confidence or Calibration:** Heuristic scores are strictly distinguished from uncalibrated softmax probabilities.
3. **Restricts Entropy to Probability Distributions:** Shannon entropy is computed exclusively when genuine class probability distributions ($p_i \ge 0, \sum p_i \approx 1.0$) are available from transformer backends; it is explicitly forbidden on heuristic confidence scores.
4. **Excludes Fallback Baseline from Disagreements:** When a transformer falls back to the baseline due to hardware, dependency, or runtime exceptions, that prediction is categorized as `unavailable` or `fallback_baseline` and never counted as a genuine cross-model disagreement.
5. **Preserves Ground Truth Source of Truth:** Human-approved labels are the permanent ground truth. Feedback corrections are persisted in isolated JSONL and CSV stores without mutating or dirtying the uploaded dataset.
6. **No Automatic Retraining:** Model fine-tuning is intentionally sequestered to future milestones, while clean training schema validation, dataset fingerprinting (SHA-256), and data leakage prevention are fully implemented today.

```
                           +-------------------------------------------------+
                           |           Active Dataset (df / CSV)            |
                           +-------------------------------------------------+
                                                   |
                     +-----------------------------+-----------------------------+
                     |                                                           |
                     v                                                           v
       +----------------------------+                             +----------------------------+
       |   Baseline Heuristics      |                             |   Multilingual Transformer  |
       |  (Script / Lexicon / Ngram)|                             |  (XLM-R / mDeBERTa-v3 NLI) |
       +----------------------------+                             +----------------------------+
                     |                                                           |
                     +-----------------------------+-----------------------------+
                                                   v
                           +-------------------------------------------------+
                           |        Model Disagreement & Uncertainty Engine  |
                           |   - Lowest Confidence Sampling (1 - max p)      |
                           |   - Prediction Margin (Top 1 - Top 2 gap)       |
                           |   - Normalized Shannon Entropy (Probabilities)  |
                           |   - Genuine Cross-Model Disagreement            |
                           +-------------------------------------------------+
                                                   |
                                                   v
                           +-------------------------------------------------+
                           |          Intelligent Priority Engine            |
                           |   - Multi-Criteria Weighted Priority            |
                           |   - Demotion for 'Reviewed — looks correct'     |
                           |   - Transparent Human-Readable Explanations     |
                           +-------------------------------------------------+
                                                   |
                                                   v
                           +-------------------------------------------------+
                           |   Prioritized Annotation Queue (Streamlit UI)   |
                           |   - One-Click Deep-Link into Annotation Studio  |
                           |   - Real-Time Dynamic Status & Condition Filter |
                           +-------------------------------------------------+
                                                   |
                                 [Annotator Modifies / Saves Label]
                                                   |
                                                   v
                           +-------------------------------------------------+
                           |          Isolated Human Feedback Store          |
                           |   - Dedicated active_learning_feedback/ Store   |
                           |   - SHA-256 Deduplication Hash on Repeated Save|
                           |   - Transition Analytics & Split Preparation    |
                           +-------------------------------------------------+
```

---

## 2. Active Learning Architecture & Algorithms

The active learning infrastructure is housed in the modular package [`ml/active_learning/`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/):

| Module | Core Responsibility |
| :--- | :--- |
| [`ml/active_learning/uncertainty.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/uncertainty.py) | Lowest-confidence, prediction margin, and normalized Shannon entropy estimation. |
| [`ml/active_learning/disagreement.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/disagreement.py) | Pairwise comparison between Baseline and Transformer with strict fallback exclusion. |
| [`ml/active_learning/priority.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/priority.py) | Composite multi-factor scoring, priority tier assignment, and human-readable audit explanations. |
| [`ml/active_learning/queue.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/queue.py) | Full queue coordination, dynamic status/condition filtering, and summary statistics. |
| [`ml/active_learning/feedback.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/feedback.py) | Human feedback collection, deduplication, isolated JSONL/CSV persistence, and error pattern analytics. |
| [`ml/active_learning/preparation.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/preparation.py) | Training schema formatting, label validation, SHA-256 dataset fingerprinting, and cross-split data leakage detection. |
| [`active_learning_dashboard.py`](file:///home/arshhh/Downloads/LinguaLabel/active_learning_dashboard.py) | Dark-themed, responsive 3-tab Streamlit dashboard with deep links into the annotation workspace. |

---

## 3. Uncertainty Scoring Methodology

The uncertainty engine supports three distinct mathematical formulations:

### A. Lowest-Confidence Sampling
Given an advisory prediction confidence score $c \in [0, 1]$:
$$\text{Uncertainty}_{\text{LC}} = 1.0 - c$$
* **Missing or Error Handling:** If a model prediction is missing, invalid, or produces an error, maximal uncertainty is assigned ($\text{Uncertainty}_{\text{LC}} = 1.0$), marking the record as requiring immediate human inspection.
* **Safety Thresholding:** Any record where $c < 0.60$ is explicitly flagged as `is_uncertain = True`.

### B. Margin Sampling
Evaluates the competition between the top two candidate classes:
$$\Delta = p_{(1)} - p_{(2)}$$
$$\text{Uncertainty}_{\text{Margin}} = 1.0 - \Delta$$
* When the difference between the most probable class and the runner-up class is small ($\Delta \to 0$), the model is indecisive between alternatives, producing high uncertainty ($\to 1.0$).
* **Constraint:** Margin sampling requires $\ge 2$ class scores. If only one score exists, the system gracefully marks margin sampling as unsupported and defaults to lowest-confidence.

### C. Normalized Shannon Entropy Sampling
Evaluates the information entropy of the probability distribution across $K$ classes ($K \ge 2$):
$$H(p) = - \sum_{i=1}^{K} p_i \ln(p_i)$$
Normalized by maximum theoretical entropy $\ln(K)$:
$$H_{\text{norm}}(p) = \frac{H(p)}{\ln(K)} \in [0, 1]$$
* **STRICT PROBABILISTIC CONSTRAINT:** In compliance with the Part 7 engineering audit, Shannon entropy is calculated **only** when `confidence_type in ('model_probability', 'calibrated_probability')` and $\sum p_i \approx 1.0$. Heuristic confidence scores (which are rule-based weights rather than probability measures) are explicitly rejected, ensuring no mathematically fabricated entropy values are generated.

---

## 4. Model Disagreement Detection Methodology

LinguaLabel employs a Query-by-Committee approach comparing the rule-based Baseline against the deep Transformer models across Sentiment and Text-Type / Intent tasks.

### Strict Integrity Rules:
1. **Provenance Verification:** Verifies the `actual_backend` property on every `PredictionResult`.
2. **Fallback Exclusion:** Predictions with `actual_backend == "fallback_baseline"` or `is_fallback == True` are categorized as `is_genuine_comparison = False` and **never** counted as a cross-model disagreement.
3. **Missing Prediction Safety:** Missing or failed predictions are treated as `unavailable`, never as disagreements.
4. **Transparent Explanations:** Every detected conflict records both models' predicted classes, confidences, and confidence types (e.g., *"Conflict on sentiment: Baseline predicted 'Positive' (80% Heuristic), whereas Transformer predicted 'Negative' (85% Prob)"*).

---

## 5. Multi-Criteria Priority Scoring Methodology

The prioritization engine combines uncertainty, model conflict, data completeness, and human annotation status into an explainable score in $[0, 1]$:

$$\text{Priority Raw} = \frac{w_{\text{disag}} \cdot S_{\text{disag}} + w_{\text{unc}} \cdot S_{\text{unc}} + w_{\text{miss}} \cdot S_{\text{miss}} + w_{\text{amb}} \cdot S_{\text{amb}}}{w_{\text{disag}} + w_{\text{unc}} + w_{\text{miss}} + w_{\text{amb}}}$$

* **Default Configurable Weights:**
  - $w_{\text{disag}} = 0.35$ (Cross-model conflict)
  - $w_{\text{unc}} = 0.35$ (Prediction uncertainty / low margin)
  - $w_{\text{miss}} = 0.15$ (Missing schema fields)
  - $w_{\text{amb}} = 0.15$ (Ambiguity / Needs second review flags)
* **Categorical Priority Levels:**
  - **High Priority:** Unreviewed records with genuine model disagreement, ambiguity flags, or composite score $\ge 0.55$.
  - **Medium Priority:** $0.30 \le \text{Score} < 0.55$.
  - **Low Priority:** $\text{Score} < 0.30$.
* **Review Demotion:** If a record has already been reviewed (`review_status == "Reviewed — looks correct"`), its priority is multiplied by a discount factor ($0.05$), ensuring completed records never clutter the annotator's queue.

---

## 6. Dashboard Functionality

The dashboard is accessible via the sidebar navigation item **"Active learning"**:

### Tab 1: Prioritized Annotation Queue
* **Overview Cards:** Displays Total Records, Unreviewed Count, High Priority Items, Disagreements, Uncertain Predictions, and Completed Annotations.
* **Sampling Parameters Expander:** Allows real-time switching between Lowest-Confidence, Smallest Margin, Shannon Entropy, Disagreement, and Multi-Factor Priority, with live slider adjustments for weights.
* **Filters:** Interactive dropdowns to filter by Priority Level (High/Med/Low), Review Status (Pending/Reviewed/Ambiguous), Condition (Disagreements Only/Uncertain Only), and full-text keyword search.
* **Annotate Item Action:** Each queue card features an `"✏️ Annotate Item"` button that automatically sets the active index and deep-links directly into the Annotation Workspace.
* **Deep-Dive Expander:** Exposes side-by-side baseline vs transformer predictions, confidence percentages, fallback status, and audit explanation bullets for each item.

### Tab 2: Human Feedback & Analytics
* **Correction Metrics:** Displays Total Feedback Items, Total Human Corrections, Baseline Corrections vs. Transformer Corrections.
* **Class Correction Frequencies:** Identifies which classes are most frequently adjusted by annotators.
* **Error Transition Patterns:** Tracks the exact transitions made (e.g., `'Negative' → 'Neutral' (sentiment)`).
* **Safe Exports:** One-click downloads for Feedback CSV (`utf-8-sig` with full Devanagari Hindi support) and JSONL.

### Tab 3: Controlled Learning Preparation
* **Standard Training Schema:** Extracts verified annotations into clean `{record_id, text, label, task, dataset_key, timestamp}` format.
* **Label Conformance Check:** Validates that labels strictly adhere to `SENTIMENT_LABELS`, `TEXT_TYPE_LABELS`, or `LANGUAGE_LABELS`.
* **Dataset Fingerprinting:** Generates deterministic 16-character SHA-256 hashes of dataset contents.
* **Train / Val / Test Partitioning:** Generates deterministic partitions (e.g. 70% / 15% / 15%) using fixed seeds.
* **Leakage Detection:** Scans for identical sentence occurrences across train, validation, and test splits to prevent evaluation contamination.

---

## 7. Feedback Storage & Isolation Design

1. **Storage Location:** Stored in [`active_learning_feedback/`](file:///home/arshhh/Downloads/LinguaLabel/active_learning_feedback/), partitioned by dataset key (e.g., `feedback_default_sample.jsonl`) and aggregated into `feedback_store.jsonl`.
2. **Deduplication Hash:** Repeated saves of the same sentence without label changes generate identical SHA-256 deduplication hashes:
   $$\text{Hash} = \text{SHA256}(\text{dataset\_key} \mid \text{record\_id} \mid \text{task} \mid \text{pred\_label} \mid \text{human\_label} \mid \text{backend})[:16]$$
   This prevents log bloating or duplicate feedback entries.
3. **Dataset Non-Destructiveness:** The uploaded CSV and `annotated_output.csv` are never polluted with feedback metadata; human labels remain the authoritative ground truth.

---

## 8. Files Created and Modified

### Created Files:
* [`ml/active_learning/__init__.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/__init__.py): Active learning package interface.
* [`ml/active_learning/uncertainty.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/uncertainty.py): Lowest-confidence, margin, and entropy sampling algorithms.
* [`ml/active_learning/disagreement.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/disagreement.py): Model conflict detection with fallback baseline exclusion.
* [`ml/active_learning/priority.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/priority.py): Explainable multi-factor prioritization engine.
* [`ml/active_learning/queue.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/queue.py): Active learning queue coordinator, filters, and summary metrics.
* [`ml/active_learning/feedback.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/feedback.py): Feedback collection, deduplication, and transition analytics.
* [`ml/active_learning/preparation.py`](file:///home/arshhh/Downloads/LinguaLabel/ml/active_learning/preparation.py): Training schema formatting, fingerprinting, and split leakage auditor.
* [`active_learning_dashboard.py`](file:///home/arshhh/Downloads/LinguaLabel/active_learning_dashboard.py): Streamlit dashboard implementation for Part 9.
* [`tests/test_active_learning.py`](file:///home/arshhh/Downloads/LinguaLabel/tests/test_active_learning.py): 14 comprehensive unit and integration tests.

### Modified Files:
* [`app.py`](file:///home/arshhh/Downloads/LinguaLabel/app.py): Added `"Active learning"` to navigation, routed dashboard rendering, and integrated automated human feedback logging on annotation save.
* [`.gitignore`](file:///home/arshhh/Downloads/LinguaLabel/.gitignore): Added `active_learning_feedback/` and `active_learning_exports/` to prevent tracking temporary feedback artifacts in Git.

---

## 9. Verification & Test Suite Results

The complete test suite was executed against the project virtual environment (`.venv` on Python 3.14.10):

```bash
PYTHONPATH=. .venv/bin/pytest
```

### Pytest Execution Output:
```text
============================= test session starts ==============================
platform linux -- Python 3.14.10, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/arshhh/Downloads/LinguaLabel
plugins: anyio-4.15.1
collected 102 items                                                            

tests/test_active_learning.py ..............                             [ 13%]
tests/test_audit_fixes_regression.py ............                        [ 25%]
tests/test_benchmark_dashboard.py ..............                         [ 39%]
tests/test_degradation.py ......                                         [ 45%]
tests/test_ml_language.py .....                                          [ 50%]
tests/test_ml_sentiment.py .......                                       [ 56%]
tests/test_ml_text_type.py .....                                         [ 61%]
tests/test_ml_transformers.py .................                          [ 78%]
tests/test_part1_integration.py ..                                       [ 80%]
tests/test_upload_and_ids.py .........                                   [ 89%]
tests/test_workflow.py ...........                                       [100%]

======================== 102 passed, 1 warning in 33.12s =======================
```

### Coverage Breakdown of 14 New Tests:
1. `test_uncertainty_lowest_confidence`: Validates $1 - c$ formulation and safety threshold.
2. `test_uncertainty_margin_sampling`: Validates gap calculation between top 2 class scores.
3. `test_uncertainty_entropy_sampling_probabilistic_only`: Verifies that entropy is computed exclusively for valid probability distributions and never for heuristic scores.
4. `test_uncertainty_invalid_or_missing_score_handling`: Graceful handling of missing predictions.
5. `test_model_disagreement_detection`: Confirms conflict detection between Baseline and Transformer.
6. `test_disagreement_transformer_fallback_exclusion`: Confirms fallback baseline predictions are never counted as disagreements.
7. `test_record_disagreement_multitask`: Validates multi-task conflict aggregation.
8. `test_priority_scoring_hierarchy`: Confirms high priority for conflicts and near-zero demotion for reviewed records.
9. `test_active_learning_queue_filtering_and_sorting`: Validates queue filters (Pending, Ambiguous, Priority levels).
10. `test_feedback_persistence_and_deduplication`: Verifies deduplication hashing on repeated saves.
11. `test_feedback_unicode_preservation`: Validates Devanagari Hindi text preservation in feedback logs.
12. `test_feedback_analytics_computation`: Validates correction metrics and transition counting.
13. `test_controlled_learning_preparation_and_splits`: Validates 70/15/15 partitioning and fingerprinting.
14. `test_data_leakage_detection`: Validates cross-split duplicate text detection.

---

## 10. Limitations & Future Improvements

1. **Model Retraining Sequestered:** In strict compliance with the development rules, no automatic model retraining or fine-tuning was performed in Part 9. Training pipelines, adapter tuning (LoRA), and model version management will be addressed in Part 10.
2. **Cold-Start Disagreements:** On small datasets with uniform predictions, model disagreements may be rare. Margin sampling and lowest-confidence sampling provide the primary prioritization signal under these conditions.
3. **Dynamic Re-Prioritization:** When an annotator approves an item, the item is demoted immediately in the active session; refreshing the queue updates the global counts and ordering.


# Model Training & Continuous Improvement Guide

LinguaLabel features a built-in **Model Training & Continuous Improvement Center** accessible to the website owner. This guide explains how to curate data, train candidate models, compare them against active production baselines, and promote models safely.

---

## 1. Accessing the Training Center

1. Open LinguaLabel in your browser (`http://localhost:8501`).
2. Click **🔐 Owner Analytics** at the bottom of the sidebar (or navigate to `/?owner=true`).
3. Enter your configured `LINGUALABEL_ADMIN_KEY` from your `.env` file.
4. Select the **🧠 Model Training & Improvement Center** view from the navigation menu.

---

## 2. Training Data Sources

The Training Center supports two complementary sources of training data:

### Source A: Labeled Dataset Uploads
- Supported formats: CSV or Excel (`.xlsx`, `.xls`).
- The system automatically detects sentence text and label columns.
- The **Dataset Quality Validator** checks:
  - Total valid samples and missing values.
  - Duplicate texts.
  - Incompatible label values.
  - Class imbalance ratios.

### Source B: Human Annotator Feedback
- Annotators' human corrections made during labeling sessions are recorded to `active_learning_feedback/`.
- In the **Human Feedback Curation** tab, the owner can:
  - Review each correction alongside the model's suggested label and annotator notes.
  - Approve or reject individual corrections.
  - Bulk-approve all pending corrections.

---

## 3. Dataset Preparation & Leakage-Safe Splitting

1. Navigate to the **Dataset Preparation** tab.
2. Select one or more uploaded datasets and optionally include approved human feedback.
3. Configure train/validation/test ratios (e.g., 70% train / 15% validation / 15% sequestered test).
4. Set a random seed for complete reproducibility.
5. Click **⚡ Generate Leakage-Safe Training Split**.
   - The engine dedupes records across sources, resolves conflicting labels by giving precedence to approved human feedback, and generates a stratified split.

---

## 4. Launching a Training Job

1. Navigate to the **Model Training Execution** tab.
2. Select your prepared training split.
3. Choose an architecture:
   - **TF-IDF + Calibrated Logistic Regression**: Fast, highly interpretable, calibrated output probabilities.
   - **Linear SGD Classifier**: High-dimensional, L2-regularized linear classifier.
4. Configure hyperparameters (regularization `C`, max n-gram features).
5. Click **🔥 Launch Model Training Job**.
   - Training runs in an isolated temporary scratch directory.
   - Models are serialized with joblib only after full validation and registered as candidate models in `trained_models/model_registry.json`.

---

## 5. Model Evaluation & Comparison

1. Navigate to the **Evaluation & Comparison** tab.
2. Select a candidate model.
3. Choose the evaluation source:
   - **Fixed Benchmark Suite** (`evaluation_datasets/complaints_eval.csv` or `evaluation_datasets/abuse_eval.csv`).
   - **Sequestered Test Split** from the training split.
4. Click **🔬 Run Comprehensive Evaluation & Comparison**.
5. The comparison table displays side-by-side metrics:
   - Overall Accuracy and Macro-F1.
   - Per-class Precision, Recall, and F1.
   - Language-stratified performance across English, Hindi, and Hinglish.
   - False-positive and false-negative regression diagnostic checks.

---

## 6. Promotion & Rollback

- **Promotion**: When a candidate outperforms production without causing language regressions, click **🚀 Promote to Production Now**. The model is atomically activated in `trained_models/active_production.json` and immediately reloaded by the live inference engine without restarting the server.
- **Instant Rollback**: If unexpected behavior is observed in production, select **↩️ Execute Safe Rollback** to instantly revert to a previous archived model version or reset to the default system baseline.


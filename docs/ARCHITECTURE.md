# LinguaLabel System Architecture

This document describes the architectural layout, data flow, and safety boundaries of the LinguaLabel platform.

```
+-----------------------------------------------------------------------------------+
|                                Streamlit Frontend                                 |
|  [Home]  [Import]  [Analyse & Label]  [Review Queue]  [Results Explorer] [Export] |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                                   Core Routing                                    |
|   app.py (Session Router, First-Party Anonymous Visitor Token & Navigation Gate)  |
+-----------------------------------------------------------------------------------+
       |                                      |                                |
       v                                      v                                v
+-----------------------+   +------------------------------------+   +---------------------+
|      Data Manager     |   |          ML Engine Coordinator     |   |  Anonymous Telemetry|
|  data_manager.py      |   |  ml/engine.py                      |   |  db/analytics.py    |
|  - UTF-8-SIG Sniffing |   |  - Unified 5-Dimension Inference   |   |  - UUID Tokens      |
|  - Column Detection   |   |  - Language Detector               |   |  - 30-min Sessions  |
|  - ID Generation      |   |  - Sentiment Detector              |   |  - Zero Text Logged |
|  - Export Generation  |   |  - Text Type & Complaint Detector  |   |  - SQLite/Postgres  |
+-----------------------+   |  - Abuse & Severity Classifier     |   +---------------------+
                            |  - Promoted Model Predictors       |
                            |  - Optional Transformer Backends   |
                            +------------------------------------+
                                               |
                                               v
                            +------------------------------------+
                            |   Training & Improvement Center    |
                            |   ui_training.py / ml/training/    |
                            |   - Dataset Validation Engine      |
                            |   - Feedback Curation Store        |
                            |   - Stratified Split Generator     |
                            |   - Isolated Scratch Training      |
                            |   - Benchmark Evaluator            |
                            |   - Model Registry & Promotion     |
                            +------------------------------------+
```

---

## 1. Architectural Principles

1. **Anonymous-by-Default Public Experience**:
   No user accounts, registrations, or passwords exist on the public web surface. Any visitor can immediately import text, run AI analysis, review suggestions, and export data without authentication.
2. **Private Administrative Operations**:
   Sensitive administrative functions—specifically website owner analytics and the continuous model training center—are gated behind constant-time cryptographic secret verification (`hmac.compare_digest`).
3. **Non-Destructive AI Advisory**:
   AI suggestions do not commit to disk or overwrite human annotations without explicit annotator confirmation.
4. **Data Isolation & Provenance**:
   User datasets are isolated by deterministic dataset keys, ensuring that sample files and user-uploaded datasets never collide.

---

## 2. Ingestion & Persistence Pipeline

- **File Reading**: `data_manager.py` uses heuristic byte sniffing to determine character encoding (`utf-8-sig`, `utf-8`, `latin-1`, `cp1252`) and delimiter (`comma`, `tab`, `semicolon`).
- **Dataset Key**: Every dataset is keyed by an MD5 hash of its initial content or name, storing persistent progress in `saved_sessions/annotated_<dataset_key>.csv`.
- **Export Integrity**: Export files are encoded with `utf-8-sig` (UTF-8 with Byte Order Mark) to ensure native rendering in Microsoft Excel without corrupting Devanagari script.

---

## 3. Machine Learning Inference Pipeline

When a sentence is presented in `ui_analyse.py`:
1. `ml.engine.MLEngine.predict_all(text)` is invoked.
2. **Promoted Model Check**: For each task (`sentiment`, `text_type`, `complaint`, `abuse`), the engine checks `trained_models/active_production.json` for a promoted owner-trained model.
3. **Safety Fallback**: If no promoted model is registered, or if a promoted model fails or is unavailable, inference safely falls back to:
   - Transformer backend (if configured and installed).
   - Baseline heuristic and lexicon rule detector (deterministic fallback).
4. **Abuse Severity Safety Guardrail**:
   - If the abuse category is `Not abusive`, the severity is guaranteed to be `None`.
   - The severity detector cannot independently declare content abusive if the primary classifier found it non-abusive.


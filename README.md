# LinguaLabel

> **AI-Assisted Multilingual Text Analysis, Quality Annotation & Continuous Model Training Platform**

LinguaLabel is an open-source, human-in-the-loop annotation and text analytics platform designed specifically for multilingual, code-mixed, and transliterated customer text. It unifies high-speed heuristic and deep transformer inference with an ergonomic Streamlit annotation interface, active learning review queues, continuous model retraining, and privacy-first website telemetry.

LinguaLabel is built for researchers, data scientists, and content moderators analysing multilingual feedback, customer support queries, social media comments, and user-generated content.

---

## 1. Project Overview

Modern NLP systems often struggle with code-mixed and romanized text—such as Hinglish (Hindi written in the Latin alphabet)—and complex intent categories like subtle complaints or profanity. LinguaLabel addresses these challenges by combining:

- **AI-Assisted Multi-Dimensional Tagging**: Generates real-time suggestions across five dimensions (Language, Sentiment, Intent/Text-Type, Complaining, and Abuse/Profanity with Severity).
- **Human-in-the-Loop Sovereignty**: Machine predictions are strictly advisory; every final label requires human review, ensuring 100% human decision immutability upon export.
- **Continuous Learning & Training Center**: Allows platform owners to upload new domain datasets, curate human annotator corrections, train custom scikit-learn models with leakage-safe stratified splitting, evaluate candidate models against baselines, and promote models to live production with instant 1-click rollback.
- **Privacy-Conscious Architecture**: The public application is completely open and anonymous—zero logins, zero registrations, and zero user profiling. Telemetry aggregates usage counts and visits without ever storing user sentence text or dataset payloads.

---

## 2. Key Features

- **Multi-Format Ingestion**: Ingest data via CSV upload, Excel spreadsheets (`.xlsx`, `.xls`), raw text pasting (multi-sentence batching), single-sentence queries, or pre-loaded multilingual benchmarks.
- **Robust UTF-8 Encoding & Byte Sniffing**: Automatic detection of UTF-8, UTF-8-SIG, Latin-1, and Windows-1252 encodings, ensuring Devanagari Hindi text renders without corruption.
- **Deterministic Record Fingerprinting**: Retains source dataset identifiers or generates reproducible MD5/SHA256 content hashes for sentence tracking across sessions.
- **Active Learning & Smart Review Queue**: Automatic prioritization of ambiguous, low-confidence (< 60%), and model-disagreement sentences into a dedicated triage queue.
- **Interactive Analytics & Distribution Charts**: Dynamic Altair and Plotly visualizations for language distribution, sentiment polarity, and intent breakdowns.
- **UTF-8 with BOM Export**: 1-click download of verified datasets in CSV (`utf-8-sig`) and native Microsoft Excel (`.xlsx`) format.
- **Private Website Owner Telemetry**: Discreet, key-gated analytics dashboard providing real-time traffic analysis, unique visitor deduplication (30-minute inactivity boundary), and feature usage metrics.

---

## 3. Supported Languages

LinguaLabel natively supports three linguistic modalities:

1. **English**: Full syntactic token matching, sentiment lexicons, and transformer coverage.
2. **Hindi (Devanagari Script, `hi`)**: Unicode block evaluation (`U+0900..U+097F`), Devanagari normalizer, bidirectional negation parsing (`नहीं`, `मत`, `ना`), and Hindi sentiment lexicons.
3. **Hinglish (Romanized Hindi / Code-Mixed, `hi-Latn`)**: Custom transliterated token dictionaries, phoneme heuristics, mixed-script token splitting, leetspeak unmasking, and Hindi-English code-mixed profanity/complaint detection.

---

## 4. AI-Assisted Multi-Dimensional Classification

The inference engine coordinates multi-dimensional classification using either fast lightweight baselines or deep transformer checkpoints:

| Dimension | Canonical Labels | Default Technique |
|---|---|---|
| **Language** | `English`, `Hindi`, `Hinglish`, `Other`, `Uncertain` | Unicode script heuristic + romanized n-gram lexicon matching |
| **Sentiment** | `Positive`, `Negative`, `Neutral`, `Uncertain` | Bilingual polarity lexicons with bidirectional negation parsing |
| **Sentence Type / Intent** | `Statement`, `Question`, `Request / Instruction`, `Greeting`, `Opinion`, `Other` | Syntactic token matcher + regular expressions + TF-IDF intent classifier |
| **Complaining** | `Complaining`, `Not complaining`, `Uncertain` | Complaint keyword matching, service dissatisfaction patterns, and negative sentiment alignment |
| **Abuse & Profanity** | `Not abusive`, `Profanity`, `Insult`, `Harassment`, `Hate Speech`, `Threat`, `Other abuse` | Multilingual regex lexicon, character substitution deobfuscation, and intent mapping |
| **Abuse Severity** | `None`, `Low`, `Medium`, `High`, `Critical` | Keyword-level severity weights, violent threat triggers, and target harassment rules |

---

## 5. Human Annotation & Review Workflow

LinguaLabel implements strict human-in-the-loop safeguards:

1. **Non-Destructive Advisory Contract**: AI predictions populate form controls only when the user clicks **Use AI** or **Accept All AI Suggestions**. AI predictions never silently overwrite saved human decisions.
2. **Mandatory Explanatory Notes**: Any sentence marked with an `Uncertain` label requires an explanatory annotator note before the review can be approved.
3. **Four-Stage Verification Status**: Sentences transition from *Unreviewed* to *Reviewed & approved*, *Needs second review*, or *Flagged for ambiguity*.
4. **Smart Review Queue**: Sentences where models express low confidence (< 0.60) or where baseline rules and transformers disagree are automatically triaged for rapid human verification.

---

## 6. Abuse Detection & Severity Classification Guardrails

Content moderation requires high precision and strict safeguards:

- **Deobfuscation & Leetspeak Normalization**: Unmasks disguised profanities (e.g., `@$$hole`, `b!tch`, `f**k`, character repetition like `crap!!!!!`).
- **Devanagari Normalization**: Collapses repeated matras, unifies nukta forms, and strips zero-width non-joiners (`ZWNJ`/`ZWJ`).
- **Severity Invariance Guardrail**: If content is classified as `Not abusive`, its severity is strictly constrained to `None`. Conversely, severity inference can never independently declare benign content abusive.
- **Multi-Label Aggregation**: Detects co-occurring abuse patterns (e.g., simultaneous personal insult and violent threat).

---

## 7. Model Training & Continuous Improvement Center

Accessible to the platform owner via private administrative authentication, the Training Center enables complete lifecycle management:

- **Owner Dataset Ingestion**: Upload labeled CSV/Excel datasets for specific tasks with automatic schema validation, missing-value checks, and class imbalance detection.
- **Human Feedback Curation**: Curate and filter annotator corrections gathered from live labeling sessions; approve or reject suggestions for the training pool.
- **Leakage-Safe Stratified Splitting**: Generates reproducible, stratified train/validation/sequestered-test splits with full provenance tracking and deduplication.
- **Scratch-Isolated Training Jobs**: Trains calibrated scikit-learn classifiers (Logistic Regression, Linear SGD) in isolated runtime directories with hyperparameter controls.
- **Side-by-Side Model Evaluation**: Evaluates candidate models against active production baselines across accuracy, macro-F1, per-class F1, language-stratified metrics, and false-positive diagnostic regressions.
- **Atomic Promotion & Instant Rollback**: 1-click model activation with immediate runtime reload and safe rollback to previous versions or system baselines.

---

## 8. Installation Instructions (Linux)

### Prerequisites
- Python 3.10 to 3.14
- Git
- SQLite (built-in) or PostgreSQL

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/Arshhh26/LinguaLabel.git
cd LinguaLabel

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) Install deep learning transformer dependencies
# Required only if using transformer backends (mDeBERTa, XLM-RoBERTa):
pip install -r requirements-transformers.txt
```

---

## 9. Environment Configuration

Copy the example configuration template and configure your local settings:

```bash
cp .env.example .env
```

Edit `.env` with appropriate values:

```env
# Application Environment
APPLICATION_ENV=production

# Database Connection (SQLite default)
DATABASE_URL=sqlite:///lingualabel_local.db

# Private Owner Secret Key (required to access Owner Analytics & Training Center)
# Generate a secure 32+ character key: openssl rand -hex 32
LINGUALABEL_ADMIN_KEY=your_secure_32_character_admin_key_here

# Session Secret Key
AUTH_SECRET_KEY=your_secure_32_character_session_key_here

# Session Inactivity Timeout (minutes)
SESSION_EXPIRATION_MINUTES=120
```

---

## 10. Application Startup

Run the Streamlit application from the project root:

```bash
streamlit run app.py
```

The web application will be accessible at:
```text
http://localhost:8501
```

To access the private **Owner Analytics & Continuous Training Center**, click the discreet **🔐 Owner Analytics** button at the bottom of the sidebar or navigate to `http://localhost:8501/?owner=true` and enter your configured `LINGUALABEL_ADMIN_KEY`.

---

## 11. Testing & Quality Assurance

Run the comprehensive automated test suite (209 tests across 18 test modules):

```bash
# Run all tests
PYTHONPATH=. pytest -v

# Run a specific test suite
PYTHONPATH=. pytest tests/test_part11_4_inference_integration.py -v

# Run with short summary output
PYTHONPATH=. pytest -q
```

All 209 tests validate:
- UTF-8 file reading, byte sniffing, and dataset stabilization.
- Language, sentiment, intent, complaining, and abuse classifiers.
- Multi-backend fallback and transformer integration.
- Active learning priority scoring and disagreement algorithms.
- Feedback recording and dataset management.
- Continuous model training, evaluation, and version registry.
- Privacy sanitization and anonymous telemetry.

---

## 12. Project Structure

```text
LinguaLabel/
├── app.py                      # Main application entry point & view router
├── data_manager.py             # Dataset I/O, UTF-8 sniffing, column detection, ID hashing
├── dashboard.py                # Research Mode model performance & evaluation dashboard
├── active_learning_dashboard.py# Priority review queue & active learning analytics
├── evaluate.py                 # CLI model evaluation wrapper
├── README.md                   # Public documentation
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contributor guidelines
├── CHANGELOG.md                # Release version history
├── requirements.txt            # Core production dependencies
├── requirements-transformers.txt # Optional transformer dependencies
├── .env.example                # Environment configuration template
├── .gitignore                  # Git exclusion rules
│
├── ui_home.py                  # Home view & dataset summary
├── ui_import.py                # Dataset ingestion (CSV, Excel, paste, examples)
├── ui_analyse.py               # 5-dimension AI-assisted labeling interface
├── ui_review.py                # Triage queue for uncertain predictions
├── ui_results.py               # Results data explorer & distribution charts
├── ui_export.py                # Export engine (CSV UTF-8-SIG, Excel)
├── ui_settings.py              # Application settings & mode toggle
├── ui_admin.py                 # Private website owner analytics dashboard
├── ui_training.py              # Model Training & Continuous Improvement Center
├── ui_mappings.py              # Reversible label mappings & plain descriptions
│
├── ml/                         # Machine learning subsystem
│   ├── engine.py               # Unified inference coordinator
│   ├── base.py                 # PredictionResult dataclass contract
│   ├── normalization.py        # Deobfuscation & Devanagari text normalizer
│   ├── language_detector.py    # Script & n-gram language classifier
│   ├── sentiment_detector.py   # Lexicon & negation sentiment analyzer
│   ├── text_type_detector.py   # Syntactic & intent classifier
│   ├── abuse_detector.py       # Multilingual abuse, profanity & threat classifier
│   ├── benchmark.py            # Comparative evaluation & confusion matrices
│   ├── model_config.py         # Hardware & threshold configuration
│   ├── model_registry.py       # Singleton model caching
│   ├── lexicons/               # Multilingual lexicons & keyword dictionaries
│   ├── active_learning/        # Priority scoring, uncertainty & disagreement
│   ├── training/               # Model training, dataset manager, versioning & evaluation
│   └── transformers/           # XLM-RoBERTa & mDeBERTa pipeline loaders
│
├── db/                         # Database & analytics subsystem
│   ├── config.py               # Environment configuration loader
│   ├── session.py              # SQLAlchemy session lifecycle
│   ├── models.py               # ORM schemas (telemetry & datasets)
│   └── analytics.py            # Anonymous visitor tracking & usage telemetry
│
├── alembic/                    # Database migration scripts
│   ├── env.py
│   └── versions/               # Versioned migration files
│
├── evaluation_datasets/        # Canonical benchmark test datasets
│   ├── abuse_eval.csv
│   └── complaints_eval.csv
│
├── trained_models/             # Production model registry & active manifests
│   ├── active_production.json
│   └── model_registry.json
│
├── training_datasets/          # Ingested training datasets & prepared splits
│   └── datasets_manifest.json
│
├── sample_data.csv             # Curated 20-sentence multilingual seed dataset
└── tests/                      # 18 automated pytest suites (209 tests)
```

---

## 13. Privacy & Security Information

LinguaLabel is built with privacy and security as foundational requirements:

- **100% Anonymous Public Usage**: Visitors do not create accounts, log in, or provide personal details.
- **Zero Content Logging**: The analytics engine strictly sanitizes event metadata against an explicit whitelist. Actual sentence text, user inputs, and uploaded dataset contents are **never** logged to the telemetry database.
- **First-Party Anonymous Tokens**: Uses randomly generated UUIDs with a 30-minute inactivity boundary for session deduplication. Zero third-party trackers, invasive fingerprinting, or cross-site cookies.
- **Owner Access Isolation**: Owner controls and training operations are protected behind constant-time cryptographic verification (`hmac.compare_digest`) using an environment-stored secret key (`LINGUALABEL_ADMIN_KEY`).
- **Local Data Sovereignty**: All annotations and uploaded files remain local to your deployment instance.

---

## 14. Known Limitations

- **Transformer Resource Consumption**: Multilingual transformer models (mDeBERTa, XLM-RoBERTa) require substantial RAM (>= 8 GB) or an NVIDIA GPU for low-latency batch inference. On low-resource CPUs, the default rule-based baseline is recommended.
- **Dialectal Variation in Hinglish**: Romanized Hindi lacks standardized spelling conventions (e.g., `accha`, `achha`, `acha`). While LinguaLabel's normalizer handles common variants, highly idiosyncratic spellings may exhibit reduced confidence.
- **Context-Dependent Sarcasm**: Sarcastic expressions without explicit syntactic indicators or polarity contradictions remain challenging for lexicon-based classifiers and may require human annotator triage.

---

## 15. License

This project is licensed under the terms of the [MIT License](LICENSE).

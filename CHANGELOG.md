# Changelog

All notable changes to the LinguaLabel project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-06

### Added
- **5-Dimensional AI Annotation Pipeline**:
  - Script & n-gram Language Identification (English, Hindi, Hinglish).
  - Lexicon & negation-aware Sentiment Analysis (Positive, Negative, Neutral).
  - Syntactic Intent & Sentence-Type Classification (Statement, Question, Request, Greeting, Opinion).
  - Multilingual Complaining & Dissatisfaction Detection.
  - Multilingual Abuse, Profanity, Harassment, Hate Speech, and Threat Detection with 5-tier Severity scoring.
- **Continuous Improvement & Model Training Center**:
  - Labeled dataset upload and validation with class-balance metrics.
  - Annotator feedback curation store with bulk approval/rejection.
  - Leakage-safe stratified dataset split engine with provenance tracking.
  - Isolated training jobs for scikit-learn classifiers with hyperparameter controls.
  - Side-by-side comparative benchmarking against production models.
  - Atomic model promotion and instant 1-click version rollback.
- **Active Learning & Prioritized Review**:
  - Uncertainty threshold triage (< 0.60 confidence).
  - Multi-backend disagreement detection (Rule-based vs Transformer).
  - Smart review queues and priority ranking.
- **Privacy-Conscious Telemetry**:
  - Anonymous first-party UUID visitor tracking with 30-minute session boundary.
  - Zero-PII event logging with strict metadata whitelisting.
  - Private key-gated Owner Analytics dashboard.
- **Multi-Format Ingestion & Export**:
  - CSV upload with automatic delimiter and encoding detection (UTF-8, UTF-8-SIG, Latin-1, CP1252).
  - Excel spreadsheet support (`.xlsx`, `.xls`).
  - Batch paste and single sentence query testing.
  - UTF-8 with BOM export in CSV and native Excel.

### Optimized
- Modernized Streamlit UI parameters to `width="stretch"` across all views, eliminating deprecation warnings.
- Fully consolidated benchmark evaluation datasets to canonical `evaluation_datasets/`.
- Streamlined repository structure for public open-source release.

### Verified
- 209 automated unit and regression tests passing with 100% success rate.


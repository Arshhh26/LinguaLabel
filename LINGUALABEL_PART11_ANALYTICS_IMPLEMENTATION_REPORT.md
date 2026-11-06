# LinguaLabel — Part 11: Private Website Analytics & Admin Dashboard Implementation Report

**Local Path:** `/home/arshhh/Downloads/LinguaLabel`  
**Framework:** Python 3.14 + Streamlit + SQLAlchemy + PostgreSQL / SQLite  
**Milestone:** Part 11 — Private Website Analytics & Admin Dashboard  
**Status:** Completed & Fully Verified (148 passing tests, zero regressions)

---

## 1. Executive Summary & Core Product Direction

In Part 11, LinguaLabel has been designed with a clear, privacy-first product principle:

> **Public Website = Completely Open (Zero Login, Zero Registration)**  
> **Analytics Dashboard = Private & Restricted Exclusively to Website Owner**

### Key Product Pillars:
1. **No User Login or Registration System for the Public:**
   - Visitors land directly on the LinguaLabel application.
   - No user registration, no signup page, no login screen, no user profiles, no email or phone verification.
   - Anyone can type or paste sentences, upload CSV and Excel spreadsheets, run AI-assisted analyses, review and correct suggestions, save their annotations locally, and export results with zero friction.
2. **Anonymous, Non-Invasive Visitor Analytics:**
   - First-party randomly generated anonymous visitor tokens (RFC 4122 UUID v4).
   - Zero invasive browser fingerprinting (no canvas hashing, audio profiling, or hardware sniffing).
   - Session deduplication enforcing a 30-minute inactivity boundary. Streamlit component reruns on the same page do not inflate visits or pageviews.
   - Rigorous distinction between **Page Views**, **Visits / Sessions**, and **Unique Visitors**.
   - Explicitly documented limitation: A visitor who clears their browser storage, uses incognito mode, or switches devices will be counted as a new visitor.
3. **Aggregate Usage Analytics & Strict Zero-PII Policy:**
   - Aggregate telemetry tracking: Number of sentence analyses, datasets uploaded, CSV imports, Excel imports, exports (CSV and Excel), and most-used features.
   - **Zero-PII Guarantee:** Actual sentence texts and uploaded file contents are **strictly forbidden** from being collected or stored in analytics.
   - Automatic metadata sanitization enforcing an explicit whitelist of safe fields (`format`, `row_count`, `task`, `status`, etc.).
4. **Private Owner Analytics Dashboard:**
   - Restricted exclusively to the website owner via a discreet access gate and constant-time secret key verification (`LINGUALABEL_ADMIN_KEY`).
   - Four interactive dashboard tabs: Visitor Analytics, Feature & Usage Telemetry, 14-Day Activity Trends, and Live Event Stream.

---

## 2. Architecture & Database Relational Schema

Four dedicated analytics tables were introduced in `db/models.py` and applied via Alembic migration (`188edf74dc11_add_analytics_models.py`):

```
+-------------------------------------------------------------+
|                          VISITORS                           |
+-------------------------------------------------------------+
| id (PK, String(64))           - Anonymous First-Party UUID  |
| first_seen_at (DateTime)      - First visit timestamp       |
| last_seen_at (DateTime)       - Most recent visit timestamp |
| visit_count (Integer)         - Distinct session count      |
| pageview_count (Integer)      - Total pageviews             |
+------------------------------+------------------------------+
                               | 1:N
                               v
+-------------------------------------------------------------+
|                      VISITOR_SESSIONS                       |
+-------------------------------------------------------------+
| id (PK, String(64))           - Unique session identifier   |
| visitor_id (FK, String(64))   - References visitors.id      |
| started_at (DateTime)         - Session start timestamp     |
| last_active_at (DateTime)     - Last interaction timestamp  |
| pageview_count (Integer)      - Pageviews within session    |
| duration_seconds (Integer)    - Total active time in session|
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                         PAGE_VIEWS                          |
+-------------------------------------------------------------+
| id (PK, String(36))           - UUID identifier             |
| visitor_id (String(64))       - Anonymous visitor UUID      |
| session_id (String(64))       - Session identifier          |
| page_name (String(100))       - View name (Home, Label, etc)|
| timestamp (DateTime)          - Transition timestamp        |
+-------------------------------------------------------------+

+-------------------------------------------------------------+
|                        USAGE_EVENTS                         |
+-------------------------------------------------------------+
| id (PK, String(36))           - UUID identifier             |
| visitor_id (String(64))       - Anonymous visitor UUID      |
| session_id (String(64))       - Session identifier          |
| event_category (String(50))   - analysis / import / export  |
| event_action (String(100))    - sentence_analysis, etc.     |
| event_count (Integer)         - Multiplier/count            |
| metadata_json (JSON)          - Sanitized safe metadata     |
| timestamp (DateTime)          - Event timestamp             |
+-------------------------------------------------------------+
```

---

## 3. Detailed Component Implementations

### 3.1 Anonymous Tracking Engine (`db/analytics.py`)
- **`generate_anonymous_id()`**: Generates a random 32-character hexadecimal UUID string.
- **`get_or_create_visitor(visitor_id)`**: Retrieves or registers a new visitor without asking for personal information.
- **`register_or_resume_session(visitor_id, session_id)`**:
  - Enforces a 30-minute (`1800s`) inactivity boundary.
  - If the visitor is active within 30 minutes, touches `last_active_at` and updates `duration_seconds`.
  - If inactive for > 30 minutes, starts a new session and increments `visitor.visit_count += 1`.
- **`record_page_view(visitor_id, session_id, page_name)`**:
  - Increments pageview counters on visitor and session records.
  - Inserts a `PageView` audit record.
- **`record_usage_event(...)`**:
  - Captures usage telemetry while guaranteeing zero text or sensitive data collection.
- **`sanitize_event_metadata(raw_metadata)`**:
  - Whitelists only safe operational keys: `{"format", "source", "row_count", "column_count", "task", "status", "category", "scope", "mode", "view", "count", "batch_size", "duration_ms", "file_type"}`.
  - Disallows any user input, sentence strings, or dataset rows.
- **`get_visitor_analytics()`**:
  - Aggregates Total Visits, Unique Visitors, New Visitors (`visit_count == 1`), Returning Visitors (`visit_count > 1`), Daily Visitors (last 24 hours), Weekly Visitors (last 7 days), Monthly Visitors (last 30 days), and Total Pageviews.
- **`get_usage_analytics()`**:
  - Aggregates Sentence Analyses, Datasets Uploaded, CSV Imports, Excel Imports, CSV Exports, Excel Exports, and Top Features.
- **`get_activity_trends(days=14)`**:
  - Generates day-by-day counts of visitors, visits, analyses, imports, and exports for visualization.
- **`verify_admin_key(entered_key)`**:
  - Compares the entered key against `ADMIN_SECRET_KEY` using constant-time `hmac.compare_digest`.

### 3.2 Public Website Experience (`app.py`)
- Removed all authentication gates, login forms, registration screens, and guest mode banners.
- App initializes anonymous visitor tracking on first load via `st.session_state` and optional query param `_vid`.
- Clean sidebar navigation:
  1. `Home`
  2. `Import Sentences`
  3. `Analyse & Label`
  4. `Review Suggestions`
  5. `My Results`
  6. `Export Data`
  7. `Settings`
- Added discreet **"🔐 Owner Analytics"** button in sidebar footer for the website owner.
- Pageview deduplication: Only records a `PageView` when `view != st.session_state["_last_tracked_view"]`.

### 3.3 Event Telemetry Integrations
- **`ui_import.py`**:
  - Records `csv_import` and `dataset_upload` on CSV upload.
  - Records `excel_import` and `dataset_upload` on Excel upload (.xlsx / .xls).
  - Records `paste_import` and `dataset_upload` on pasted sentences.
  - Records `single_sentence_entry` on manual test sentence.
- **`ui_analyse.py`**:
  - Records `sentence_analysis` once per row per session when AI predictions run.
  - Records `accept_ai_suggestions` when user clicks "Accept All AI Suggestions".
  - Records `save_annotation` when user saves an annotation.
- **`ui_export.py`**:
  - Records `export_csv` when downloading CSV.
  - Records `export_excel` when downloading Excel workbook.
- **`ui_review.py`**:
  - Records `active_learning_review` when generating or refreshing uncertainty suggestions.

### 3.4 Private Website Owner Analytics Dashboard (`ui_admin.py`)
- **Access Gate (`render_owner_access_gate`)**:
  - Prompt for Owner Secret Key with constant-time verification.
  - Informs ordinary visitors that public features require no login.
- **Tab 1: 👥 Visitor Analytics**:
  - 8 real-time KPI metrics cards: Total Unique Visitors, Total Visits, New Visitors, Returning Visitors, Returning Retention Ratio, Daily (24h), Weekly (7d), Monthly (30d), and Total Pageviews.
  - Pageviews by Section bar chart.
  - Concept explanation banner clarifying Page Views vs. Visits vs. Unique Visitors.
- **Tab 2: 🚀 Feature & Usage Telemetry**:
  - 5 core usage metrics: Sentence Analyses, Datasets Uploaded, CSV Imports, Excel Imports, and Total Exports.
  - Most-used feature breakdown table.
  - Ingestion vs. Export channel distribution chart.
- **Tab 3: 📈 14-Day Activity Trends**:
  - Dual time-series charts for Daily Visitors/Visits and Daily Workflows (analyses, imports, exports).
  - Day-by-day breakdown table.
- **Tab 4: 📜 Live Event Stream**:
  - Live audit table of recent aggregate events with timestamp, category, action, and verified safe metadata.

---

## 4. Verification & Testing

### Complete Test Suite Execution:
```bash
PYTHONPATH=. .venv/bin/pytest
```
**Results:**
```
collected 148 items

tests/test_active_learning.py ..............                             [  9%]
tests/test_audit_fixes_regression.py ............                        [ 17%]
tests/test_benchmark_dashboard.py ..............                         [ 27%]
tests/test_degradation.py ......                                         [ 31%]
tests/test_ml_language.py .....                                          [ 34%]
tests/test_ml_sentiment.py .......                                       [ 39%]
tests/test_ml_text_type.py .....                                         [ 42%]
tests/test_ml_transformers.py .................                          [ 54%]
tests/test_part10_1_audit.py ............                                [ 62%]
tests/test_part10_simplification.py ............                         [ 70%]
tests/test_part11_analytics.py ............                              [ 78%]
tests/test_part11_multiuser.py ..........                                [ 85%]
tests/test_part1_integration.py ..                                       [ 86%]
tests/test_upload_and_ids.py .........                                   [ 92%]
tests/test_workflow.py ...........                                       [100%]

======================= 148 passed, 1 warning in 35.50s ========================
```

### Dedicated Analytics Test Coverage (`tests/test_part11_analytics.py`):
1. `test_anonymous_id_generation`: Validates RFC 4122 UUID v4 formatting.
2. `test_get_or_create_visitor`: Validates new vs returning visitor retrieval.
3. `test_session_resumption_within_inactivity_window`: Verifies session resumption and rerun deduplication.
4. `test_session_inactivity_timeout_and_returning_visitor`: Verifies 30-minute timeout boundary and `visit_count` increment.
5. `test_page_view_tracking`: Verifies pageview counter increments across views.
6. `test_privacy_sanitization_strips_sentence_and_dataset_text`: Verifies that actual text and content are stripped.
7. `test_record_usage_event_privacy`: Verifies database-level metadata sanitization.
8. `test_usage_and_feature_analytics`: Verifies aggregate metric calculation.
9. `test_visitor_summary_metrics`: Verifies daily, weekly, monthly, new, and returning visitor metrics.
10. `test_activity_trends_and_recent_log`: Verifies 14-day trend calculations.
11. `test_admin_key_verification`: Verifies constant-time owner key authentication.
12. `test_public_unauthenticated_workflow`: Verifies that single sentence queries, pasted sentences, CSV loading, and ML analysis execute with zero authentication required.

---

## 5. Privacy Notice & Documented Limitations

### Privacy-First Guarantees:
- **No Invasive Fingerprinting:** LinguaLabel does not use canvas fingerprinting, audio fingerprinting, WebGL hashing, or hardware profiling.
- **Strict Zero-PII Policy:** User sentences, uploaded dataset text, and file contents are never stored in telemetry tables.
- **Anonymity:** First-party visitor UUIDs are completely disconnected from personal identifiers.

### Documented Limitation:
- A visitor who clears their browser storage/cookies, browses in incognito/private mode, or accesses LinguaLabel from a different device will be assigned a new anonymous identifier and counted as a new visitor.

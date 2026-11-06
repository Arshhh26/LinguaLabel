# LinguaLabel — Part 11 Implementation Report
## User Accounts, Personal Workspaces & Multi-User Data Isolation

**Project Path:** `/home/arshhh/Downloads/LinguaLabel`  
**Completion Date:** November 6, 2026  
**Auditor & Implementation Agent:** Antigravity (Advanced Agentic Pair Programmer)  
**Status:** Successfully Implemented & 136/136 Automated Tests Passing  

---

## 1. Executive Summary

LinguaLabel has transitioned from a single-user local tool into a secure, multi-user application. Part 11 introduces:
- **Relational Multi-User Storage:** Implemented PostgreSQL support using SQLAlchemy 2.0 and Alembic schema migrations, with zero-config SQLite support for local development and isolated unit testing.
- **Argon2id Authentication & Cryptographic Sessions:** RFC 9106 Argon2id password hashing ($64\text{MB}$ memory, $3$ iterations, $4$ threads), input sanitization, generic error messages, account active status checks, and HMAC-SHA256 timestamped session tokens.
- **Zero-Trust Multi-User Data Isolation:** Enforced strictly at the service layer (`db/service.py`) across Workspaces, Datasets, Sentences, Feedback, and Exports. User B cannot view, query, mutate, delete, or export any resource owned by User A.
- **Personal Dashboard:** Dedicated landing hub displaying isolated metrics (total workspaces, datasets, sentences, pending reviews, completed datasets) and recent activity.
- **Workspace & Dataset Management:** Full lifecycle controls (create, rename, duplicate, delete with confirmation, reopen and resume incomplete annotations).
- **Non-Destructive Storage Migration Tooling:** Automated repeat-safe importer for legacy `saved_sessions/*.csv` and `active_learning_feedback/*.jsonl` files with pre-migration timestamped backups and original ID preservation.
- **Comprehensive Verification:** Added 10 new regression, isolation, and security tests in `tests/test_part11_multiuser.py`, expanding the test suite to **136 passing tests** (136 passed, 0 failed).

---

## 2. Architecture & Design

```mermaid
graph TD
    UI[Streamlit UI Layer] --> AuthGate{Auth Gateway}
    AuthGate -->|Authenticated| MainApp[Personal Dashboard & Navigation]
    AuthGate -->|Unauthenticated| AuthView[Sign In / Register]
    
    MainApp --> ServiceLayer[Service & Authorization Layer<br/>db/service.py]
    ServiceLayer -->|Strict user_id scoping| DB[(SQLAlchemy Engine<br/>PostgreSQL / SQLite)]
    
    subgraph Database Models
        Users[Users] -->|1:N| Workspaces[Workspaces]
        Workspaces -->|1:N| Datasets[Datasets]
        Users -->|1:N| Datasets
        Datasets -->|1:N| Sentences[Sentences]
        Users -->|1:N| Feedbacks[Feedbacks]
        Datasets -->|1:N| Feedbacks
        Sentences -->|1:N| Feedbacks
    end
    
    ServiceLayer --> DB
    MainApp --> MLEngine[Advisory ML Engine]
```

---

## 3. Database Schema Specification

All primary keys use UUID v4 strings (`String(36)`), guaranteeing consistent behavior across PostgreSQL and SQLite without driver-specific casting quirks.

### 3.1 Table Definitions

1. **`users`**
   - `id`: `String(36)` PRIMARY KEY (UUID v4)
   - `email`: `String(255)` UNIQUE, INDEXED, NOT NULL (case-insensitive checks)
   - `username`: `String(100)` UNIQUE, INDEXED, NOT NULL
   - `password_hash`: `String(255)` NOT NULL (Argon2id encoded hash)
   - `is_active`: `Boolean` DEFAULT TRUE, NOT NULL
   - `created_at`: `DateTime(timezone=True)` NOT NULL
   - `updated_at`: `DateTime(timezone=True)` NOT NULL

2. **`workspaces`**
   - `id`: `String(36)` PRIMARY KEY (UUID v4)
   - `owner_id`: `String(36)` FOREIGN KEY (`users.id` ON DELETE CASCADE), INDEXED, NOT NULL
   - `name`: `String(150)` NOT NULL
   - `description`: `Text` NULLABLE
   - `created_at`: `DateTime(timezone=True)` NOT NULL
   - `updated_at`: `DateTime(timezone=True)` NOT NULL
   - *Index:* `ix_workspaces_owner_name` (`owner_id`, `name`)

3. **`datasets`**
   - `id`: `String(36)` PRIMARY KEY (UUID v4)
   - `workspace_id`: `String(36)` FOREIGN KEY (`workspaces.id` ON DELETE CASCADE), INDEXED, NOT NULL
   - `owner_id`: `String(36)` FOREIGN KEY (`users.id` ON DELETE CASCADE), INDEXED, NOT NULL
   - `name`: `String(200)` NOT NULL
   - `original_filename`: `String(255)` NULLABLE
   - `dataset_key`: `String(64)` INDEXED, NULLABLE (for legacy file mapping)
   - `total_sentences`: `Integer` DEFAULT 0, NOT NULL
   - `processing_status`: `String(50)` DEFAULT "ready", NOT NULL
   - `created_at`: `DateTime(timezone=True)` NOT NULL
   - `updated_at`: `DateTime(timezone=True)` NOT NULL
   - *Index:* `ix_datasets_owner_workspace` (`owner_id`, `workspace_id`)

4. **`sentences`**
   - `id`: `String(64)` PRIMARY KEY (UUID or stable deterministic ID)
   - `dataset_id`: `String(36)` FOREIGN KEY (`datasets.id` ON DELETE CASCADE), INDEXED, NOT NULL
   - `sequence_number`: `Integer` DEFAULT 0, NOT NULL
   - `original_text`: `Text` NOT NULL
   - `language`: `String(50)` NULLABLE (`English`, `Hindi`, `Hinglish`, `Other`, `Uncertain`)
   - `sentiment`: `String(50)` NULLABLE (`Positive`, `Negative`, `Neutral`, `Not applicable`, `Uncertain`)
   - `sentence_type`: `String(50)` NULLABLE (`Statement`, `Question`, `Request / Instruction`, `Greeting`, `Other`, `Uncertain`)
   - `review_status`: `String(50)` DEFAULT "Pending review", NOT NULL
   - `annotator_notes`: `Text` NULLABLE
   - `annotation_metadata`: `JSON` NULLABLE
   - `created_at`: `DateTime(timezone=True)` NOT NULL
   - `updated_at`: `DateTime(timezone=True)` NOT NULL
   - *Indexes:* `ix_sentences_dataset_seq` (`dataset_id`, `sequence_number`), `ix_sentences_dataset_status` (`dataset_id`, `review_status`)

5. **`feedbacks`**
   - `id`: `String(36)` PRIMARY KEY (UUID v4)
   - `owner_id`: `String(36)` FOREIGN KEY (`users.id` ON DELETE CASCADE), INDEXED, NOT NULL
   - `dataset_id`: `String(36)` FOREIGN KEY (`datasets.id` ON DELETE CASCADE), INDEXED, NULLABLE
   - `sentence_id`: `String(64)` FOREIGN KEY (`sentences.id` ON DELETE CASCADE), INDEXED, NULLABLE
   - `task`: `String(50)` NOT NULL (`language`, `sentiment`, `text_type`)
   - `original_prediction`: `JSON` NULLABLE
   - `corrected_label`: `String(100)` NOT NULL
   - `annotator_notes`: `Text` NULLABLE
   - `created_at`: `DateTime(timezone=True)` NOT NULL
   - *Index:* `ix_feedbacks_owner_task` (`owner_id`, `task`)

---

## 4. Authentication & Security Architecture

1. **Password Hashing:**
   - Implemented using `argon2-cffi` adhering to RFC 9106 guidelines:
     - Algorithm: Argon2id
     - Time cost ($t$): $3$ iterations
     - Memory cost ($m$): $65536\text{ KB}$ ($64\text{ MB}$)
     - Parallelism ($p$): $4$ threads
     - Salt length: $16$ bytes, Hash length: $32$ bytes.
   - Plaintext passwords are never logged, cached, or persisted.

2. **Session Token Security:**
   - Session tokens follow the tamper-proof format:
     $$\text{Token} = \langle\text{user\_id}\rangle.\langle\text{timestamp}\rangle.\text{HMAC-SHA256}(\text{AUTH\_SECRET\_KEY}, \langle\text{user\_id}\rangle.\langle\text{timestamp}\rangle)$$
   - Verification uses constant-time comparison (`hmac.compare_digest`) to prevent timing attacks.
   - Tokens expire automatically after `SESSION_EXPIRATION_MINUTES` (default $120$ minutes).

3. **Protection against Account Enumeration:**
   - Invalid password or non-existent email/username returns a generic: `"Invalid email or password."`
   - Account deactivation is explicitly enforced: deactivated accounts are blocked with a clear notice.

---

## 5. Multi-User Isolation & Authorization

Every data access query is mediated through `db/service.py` requiring the caller to supply the authenticated `user_id`.

| Operation | Service Method | Enforcement Mechanism |
| :--- | :--- | :--- |
| **List Workspaces** | `list_workspaces(user_id)` | Filters `Workspace.owner_id == user_id` |
| **Get Workspace** | `get_workspace(user_id, ws_id)` | Filters `Workspace.id == ws_id` AND `Workspace.owner_id == user_id` |
| **Delete Workspace** | `delete_workspace(user_id, ws_id)` | Verifies ownership and prevents deletion of the last remaining workspace |
| **List Datasets** | `list_datasets(user_id, ws_id)` | Filters `Dataset.owner_id == user_id` |
| **Fetch Sentences** | `get_dataset_sentences(user_id, ds_id)` | Joins `Dataset`, verifying `Dataset.owner_id == user_id` |
| **Update Annotation** | `update_sentence_annotation(user_id, ds_id, sent_id)` | Verifies `Dataset.owner_id == user_id` before committing updates |
| **Record Feedback** | `record_feedback(user_id, ds_id, sent_id)` | Verifies dataset ownership and binds `Feedback.owner_id = user_id` |
| **Export Dataset** | `export_user_dataset_df(user_id, ds_id)` | Blocks export and returns `(None, "Unauthorized")` if not owned |

Automated test `test_cross_user_isolation` in `tests/test_part11_multiuser.py` verifies all cross-user boundaries.

---

## 6. Personal Dashboard & Workspace Management

1. **Dashboard Interface (`ui_dashboard.py`):**
   - Displays real-time isolated statistics: Workspaces, Datasets, Sentences, Pending Reviews, and Completed Datasets.
   - Summarizes the user's latest 5 modified datasets in a clean activity feed with quick-launch shortcuts.

2. **Workspace Management (`ui_workspaces.py`):**
   - Allows users to switch workspaces, create new project workspaces, rename existing ones, or delete empty/unneeded workspaces.
   - For safety, workspace deletion requires checking an explicit confirmation box.
   - Datasets inside the selected workspace display real-time review progress counters, with quick actions to **Open & Annotate**, **Duplicate**, **Rename**, **Delete**, or **Download (CSV & Excel)**.

---

## 7. Safe Legacy Data Migration Tooling

The migration tool (`db/migration_tool.py`) preserves all historical local data:
- **Pre-Flight Preview:** Inspects `saved_sessions/` and `active_learning_feedback/` to report sentence and feedback counts.
- **Non-Destructive Backups:** Creates a timestamped directory under `legacy_backups/backup_<timestamp>/` copying all source files before database insertion.
- **Original Fidelity:** Preserves original sentence IDs, human-approved labels, and annotator notes.
- **Repeat-Safe (Idempotent):** Skips datasets that have already been imported for the target user.
- **No Local Deletions:** Source files remain untouched on disk.

---

## 8. Configuration & Environment Variables

A `.env.example` file has been provided in the repository root:

```ini
# Application Environment: 'development', 'testing', or 'production'
APPLICATION_ENV=development

# Production PostgreSQL Connection:
# DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/lingualabel_db

# Local Development / Testing Fallback (default):
DATABASE_URL=sqlite:///lingualabel_local.db

# Cryptographic Secret Key:
AUTH_SECRET_KEY=lingualabel-dev-secret-replace-with-random-hex-in-production-2026

# Session Inactivity Expiration (in minutes):
SESSION_EXPIRATION_MINUTES=120
```

`.env` is registered in `.gitignore` to prevent credential exposure.

---

## 9. Test Suite Verification Results

Running pytest across the complete repository:

```bash
============================= test session starts ==============================
platform linux -- Python 3.14.10, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/arshhh/Downloads/LinguaLabel
plugins: anyio-4.15.1
collected 136 items

tests/test_active_learning.py ..............                             [ 10%]
tests/test_audit_fixes_regression.py ............                        [ 19%]
tests/test_benchmark_dashboard.py ..............                         [ 29%]
tests/test_degradation.py ......                                         [ 33%]
tests/test_ml_language.py .....                                          [ 37%]
tests/test_ml_sentiment.py .......                                       [ 42%]
tests/test_ml_text_type.py .....                                         [ 46%]
tests/test_ml_transformers.py .................                          [ 58%]
tests/test_part10_1_audit.py ............                                [ 67%]
tests/test_part10_simplification.py ............                         [ 76%]
tests/test_part11_multiuser.py ..........                                [ 83%]
tests/test_part1_integration.py ..                                       [ 85%]
tests/test_upload_and_ids.py .........                                   [ 91%]
tests/test_workflow.py ...........                                       [100%]

======================= 136 passed, 1 warning in 33.48s ========================
```

**Verification Summary:**
- Total Tests: **136**
- Passed: **136**
- Failed: **0**
- Regressions: **None**

---

## 10. Exact Fedora Setup & Run Commands

### 10.1 Running with Local SQLite (Zero-Config Development)
```bash
cd /home/arshhh/Downloads/LinguaLabel

# Run Alembic migrations
.venv/bin/alembic upgrade head

# Launch Streamlit application
.venv/bin/streamlit run app.py --server.port 8501
```

### 10.2 Running with PostgreSQL (Production)
```bash
# 1. Install PostgreSQL and start the service (if not already installed)
sudo dnf install -y postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl enable --now postgresql

# 2. Create the production database and user
sudo -u postgres psql -c "CREATE USER lingualabel WITH PASSWORD 'lingualabel_secret';"
sudo -u postgres psql -c "CREATE DATABASE lingualabel_db OWNER lingualabel;"

# 3. Configure .env
echo "DATABASE_URL=postgresql+psycopg://lingualabel:lingualabel_secret@localhost:5432/lingualabel_db" > .env
echo "AUTH_SECRET_KEY=$(openssl rand -hex 32)" >> .env
echo "APPLICATION_ENV=production" >> .env

# 4. Run Alembic migrations against PostgreSQL
.venv/bin/alembic upgrade head

# 5. Launch LinguaLabel
.venv/bin/streamlit run app.py --server.port 8501
```

---

## 11. Known Limitations & Next Steps

1. **Email Confirmation / Password Reset:** The current authentication implementation validates email format and duplicate uniqueness, but does not send external SMTP email verifications.
2. **Role-Based Collaboration:** Workspaces are currently isolated strictly per individual user. Shared team workspaces can be added in future iterations if collaborative multi-annotator review workflows are needed.
3. **Database Concurrency:** SQLite handles concurrent reads well, but production multi-user deployments with many simultaneous writers should use the provided PostgreSQL backend.


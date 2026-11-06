
"""
LinguaLabel — Database & Application Configuration

Loads and validates environment variables for:
- Application environment
- Database connectivity
- Cryptographic session signing
- Private website owner authentication
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the LinguaLabel project root.
# Existing system environment variables take precedence.
load_dotenv(BASE_DIR / ".env", override=False)


# ============================================================
# APPLICATION ENVIRONMENT
# ============================================================

APPLICATION_ENV = (
    os.getenv("APPLICATION_ENV", "development")
    .strip()
    .lower()
)

VALID_ENVIRONMENTS = {"development", "testing", "production"}

if APPLICATION_ENV not in VALID_ENVIRONMENTS:
    raise ValueError(
        f"Invalid APPLICATION_ENV: {APPLICATION_ENV}. "
        f"Expected one of: {', '.join(sorted(VALID_ENVIRONMENTS))}"
    )


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(BASE_DIR / 'lingualabel_local.db').as_posix()}",
).strip()

if not DATABASE_URL:
    raise ValueError("DATABASE_URL cannot be empty.")


# ============================================================
# AUTHENTICATION / SESSION SECRET
# ============================================================

AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "").strip()

if not AUTH_SECRET_KEY:
    raise RuntimeError(
        "Authentication is not configured. "
        "Set the AUTH_SECRET_KEY environment variable in .env."
    )

if len(AUTH_SECRET_KEY) < 32:
    raise ValueError(
        "AUTH_SECRET_KEY must contain at least 32 characters."
    )


# ============================================================
# SESSION CONFIGURATION
# ============================================================

try:
    SESSION_EXPIRATION_MINUTES = int(
        os.getenv("SESSION_EXPIRATION_MINUTES", "120")
    )
except ValueError as exc:
    raise ValueError(
        "SESSION_EXPIRATION_MINUTES must be an integer."
    ) from exc

if SESSION_EXPIRATION_MINUTES <= 0:
    raise ValueError(
        "SESSION_EXPIRATION_MINUTES must be greater than zero."
    )


# ============================================================
# PRIVATE WEBSITE OWNER / ADMIN AUTHENTICATION
# ============================================================

# Only the website owner should know this key.
# Never use a hardcoded fallback.
ADMIN_SECRET_KEY = (
    os.getenv("LINGUALABEL_ADMIN_KEY")
    or os.getenv("ADMIN_SECRET_KEY")
    or ""
).strip()

if not ADMIN_SECRET_KEY:
    raise RuntimeError(
        "Owner authentication is not configured. "
        "Set the LINGUALABEL_ADMIN_KEY environment variable."
    )

if len(ADMIN_SECRET_KEY) < 32:
    raise ValueError(
        "LINGUALABEL_ADMIN_KEY must contain at least 32 characters."
    )


# ============================================================
# CONFIGURATION VALIDATION COMPLETE
# ============================================================

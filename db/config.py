"""
LinguaLabel — Database & Application Configuration

Loads and validates configuration from environment variables, .env,
and Streamlit Cloud secrets (st.secrets) for:
- Application environment
- Database connectivity
- Cryptographic session signing
- Private website owner authentication
"""

import os
from pathlib import Path
import secrets
from typing import Optional

from dotenv import load_dotenv


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from the LinguaLabel project root if present.
# System environment variables take precedence.
load_dotenv(BASE_DIR / ".env", override=False)


# ============================================================
# SECRET RESOLUTION HELPER (ENV + STREAMLIT CLOUD SECRETS)
# ============================================================

def _get_secret_or_env(key: str, default: str = "") -> str:
    """
    Retrieves configuration from environment variables or Streamlit secrets (st.secrets).
    Falls back to default if neither is present.
    """
    # 1. Check OS environment variable
    val = os.getenv(key)
    if val:
        return val.strip()

    # 2. Check Streamlit secrets (for Streamlit Community Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            try:
                if key in st.secrets:
                    v = st.secrets[key]
                    if v is not None and str(v).strip():
                        return str(v).strip()
            except Exception:
                pass
    except Exception:
        pass

    return default


# ============================================================
# APPLICATION ENVIRONMENT
# ============================================================

APPLICATION_ENV = (
    _get_secret_or_env("APPLICATION_ENV", "production")
    .strip()
    .lower()
)

VALID_ENVIRONMENTS = {"development", "testing", "production"}

if APPLICATION_ENV not in VALID_ENVIRONMENTS:
    APPLICATION_ENV = "production"


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = _get_secret_or_env(
    "DATABASE_URL",
    f"sqlite:///{(BASE_DIR / 'lingualabel_local.db').as_posix()}",
).strip()

if not DATABASE_URL:
    DATABASE_URL = f"sqlite:///{(BASE_DIR / 'lingualabel_local.db').as_posix()}"


# ============================================================
# AUTHENTICATION / SESSION SECRET
# ============================================================

AUTH_SECRET_KEY = _get_secret_or_env("AUTH_SECRET_KEY")

if not AUTH_SECRET_KEY:
    # On cloud deployments or fresh environments without a .env file,
    # generate an ephemeral, cryptographically secure key.
    # This ensures seamless zero-configuration deployment on Streamlit Cloud.
    AUTH_SECRET_KEY = secrets.token_hex(32)

if len(AUTH_SECRET_KEY) < 32:
    AUTH_SECRET_KEY = (AUTH_SECRET_KEY * 32)[:64]


# ============================================================
# SESSION CONFIGURATION
# ============================================================

try:
    SESSION_EXPIRATION_MINUTES = int(
        _get_secret_or_env("SESSION_EXPIRATION_MINUTES", "120")
    )
except ValueError:
    SESSION_EXPIRATION_MINUTES = 120

if SESSION_EXPIRATION_MINUTES <= 0:
    SESSION_EXPIRATION_MINUTES = 120


# ============================================================
# PRIVATE WEBSITE OWNER / ADMIN AUTHENTICATION
# ============================================================

# Static random fallback generated per server process so owner access
# is never unauthenticated by default if no secret is configured.
_STATIC_FALLBACK_ADMIN_KEY = secrets.token_hex(32)


def get_admin_secret_key() -> str:
    """Returns the current owner admin key from env, Streamlit secrets, or cached fallback."""
    key = (
        _get_secret_or_env("LINGUALABEL_ADMIN_KEY")
        or _get_secret_or_env("ADMIN_SECRET_KEY")
    )
    if key and len(key) >= 32:
        return key
    elif key:
        return (key * 32)[:64]
    return _STATIC_FALLBACK_ADMIN_KEY


# Check whether a custom owner key was explicitly configured
_configured_key = (
    _get_secret_or_env("LINGUALABEL_ADMIN_KEY")
    or _get_secret_or_env("ADMIN_SECRET_KEY")
)
OWNER_KEY_CONFIGURED: bool = bool(_configured_key and len(_configured_key) >= 8)

ADMIN_SECRET_KEY: str = get_admin_secret_key()

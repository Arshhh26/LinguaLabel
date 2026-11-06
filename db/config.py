"""
LinguaLabel — Database & Application Configuration (Part 11)

Loads and validates environment variables for database connectivity,
cryptographic session signing, and environment modes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the LinguaLabel project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from project root if present
load_dotenv(BASE_DIR / ".env")

APPLICATION_ENV = os.getenv("APPLICATION_ENV", "development").strip().lower()

# Database URL: defaults to local SQLite file for zero-config dev/testing,
# and supports PostgreSQL in production via postgresql+psycopg://
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'lingualabel_local.db'}"
).strip()

# Cryptographic secret for signing session tokens and state
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "").strip()
if not AUTH_SECRET_KEY:
    if APPLICATION_ENV == "production":
        raise ValueError("AUTH_SECRET_KEY environment variable is mandatory in production!")
    AUTH_SECRET_KEY = "dev-insecure-secret-key-replace-in-production-lingualabel"

# Session expiration in minutes (default 120 minutes / 2 hours)
try:
    SESSION_EXPIRATION_MINUTES = int(os.getenv("SESSION_EXPIRATION_MINUTES", "120"))
except ValueError:
    SESSION_EXPIRATION_MINUTES = 120


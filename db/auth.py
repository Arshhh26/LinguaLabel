"""
LinguaLabel — Authentication, Cryptography & Session Management (Part 11)

Implements:
- Argon2id password hashing and verification.
- HMAC-SHA256 cryptographically signed session tokens with timestamp expiration.
- Input validation (email, username, password complexity).
- Generic authentication error messages to mitigate user enumeration.
- Account deactivation checks.
- Automatic personal default workspace creation upon registration.
"""

import hmac
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from db.config import AUTH_SECRET_KEY, SESSION_EXPIRATION_MINUTES
from db.models import User, Workspace
from db.session import get_db_session

# RFC 9106 Argon2id password hasher (64MB memory, 3 iterations, 4 parallelism threads)
_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]{3,30}$")


def hash_password(password: str) -> str:
    """Hashes a plaintext password using Argon2id."""
    if not password:
        raise ValueError("Password cannot be empty.")
    return _PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verifies a plaintext password against an Argon2id hash."""
    if not password or not password_hash:
        return False
    try:
        return _PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, Exception):
        return False


def validate_registration_inputs(email: str, username: str, password: str) -> Tuple[bool, Optional[str]]:
    """Validates registration input criteria."""
    clean_email = (email or "").strip()
    clean_username = (username or "").strip()

    if not clean_email or not EMAIL_REGEX.match(clean_email):
        return False, "Please enter a valid email address."

    if not clean_username or not USERNAME_REGEX.match(clean_username):
        return False, "Username must be 3-30 characters long and contain only letters, numbers, underscores, or hyphens."

    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long."

    if not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password):
        return False, "Password must contain at least one letter and one number."

    return True, None


def create_session_token(user_id: str) -> str:
    """
    Creates a tamper-proof, timestamped session token signed with HMAC-SHA256.
    Format: <user_id>.<timestamp>.<signature_hex>
    """
    now_ts = int(time.time())
    payload = f"{user_id}.{now_ts}"
    signature = hmac.new(
        AUTH_SECRET_KEY.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}.{signature}"


def verify_session_token(token: str) -> Optional[str]:
    """
    Verifies the HMAC signature and expiration of a session token.
    Returns the user_id if valid and unexpired; otherwise returns None.
    """
    if not token or not isinstance(token, str):
        return None

    parts = token.split(".")
    if len(parts) != 3:
        return None

    user_id, ts_str, signature = parts
    payload = f"{user_id}.{ts_str}"
    expected_sig = hmac.new(
        AUTH_SECRET_KEY.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_sig):
        return None

    try:
        ts = int(ts_str)
    except ValueError:
        return None

    # Check expiration (in minutes)
    max_age_seconds = SESSION_EXPIRATION_MINUTES * 60
    if time.time() - ts > max_age_seconds:
        return None

    return user_id


def register_user(
    email: str,
    username: str,
    password: str,
    session: Optional[Session] = None,
) -> Tuple[Optional[User], Optional[str]]:
    """
    Registers a new user, validates inputs, prevents duplicate emails/usernames,
    and initializes a default personal workspace.
    """
    valid, err = validate_registration_inputs(email, username, password)
    if not valid:
        return None, err

    clean_email = email.strip().lower()
    clean_username = username.strip()

    def _do_register(sess: Session) -> Tuple[Optional[User], Optional[str]]:
        # Check duplicate email
        existing_email = sess.query(User).filter(func.lower(User.email) == clean_email).first()
        if existing_email:
            return None, "An account with this email address already exists."

        # Check duplicate username
        existing_user = sess.query(User).filter(func.lower(User.username) == clean_username.lower()).first()
        if existing_user:
            return None, "This username is already taken. Please choose another."

        p_hash = hash_password(password)
        new_user = User(
            email=clean_email,
            username=clean_username,
            password_hash=p_hash,
            is_active=True,
        )
        sess.add(new_user)
        sess.flush()  # assign user.id

        # Provision personal default workspace
        default_ws = Workspace(
            owner_id=new_user.id,
            name="Personal Workspace",
            description="Your default workspace for NLP annotation projects.",
        )
        sess.add(default_ws)
        return new_user, None

    if session is not None:
        return _do_register(session)
    else:
        with get_db_session() as sess:
            return _do_register(sess)


def authenticate_user(
    email_or_username: str,
    password: str,
    session: Optional[Session] = None,
) -> Tuple[Optional[User], Optional[str]]:
    """
    Authenticates a user by email or username with generic failure messages.
    Checks account active status.
    """
    ident = (email_or_username or "").strip()
    if not ident or not password:
        return None, "Invalid email or password."

    def _do_auth(sess: Session) -> Tuple[Optional[User], Optional[str]]:
        user = sess.query(User).filter(
            or_(
                func.lower(User.email) == ident.lower(),
                func.lower(User.username) == ident.lower(),
            )
        ).first()

        # Generic authentication failure message to prevent username enumeration
        if not user or not verify_password(password, user.password_hash):
            return None, "Invalid email or password."

        if not user.is_active:
            return None, "Your account has been deactivated. Please contact an administrator."

        return user, None

    if session is not None:
        return _do_auth(session)
    else:
        with get_db_session() as sess:
            return _do_auth(sess)


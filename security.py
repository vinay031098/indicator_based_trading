"""
Security helpers (Tasks 3, 8): dashboard password verification and
encrypted-at-rest persistence for the Fyers access token.
"""

from __future__ import annotations

import hmac
import logging
import os

from werkzeug.security import check_password_hash, generate_password_hash

from config import settings

logger = logging.getLogger(__name__)

_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fyers_token.enc")


# ─── Dashboard password ────────────────────────────────────────────

def verify_dashboard_login(username: str, password: str) -> bool:
    """Check username + password (plain env or werkzeug hash)."""
    if username != settings.dashboard_user:
        return False
    plain = (settings.dashboard_password or "").strip()
    if plain:
        return hmac.compare_digest(password, plain)
    pwd_hash = (settings.dashboard_password_hash or "").strip()
    if not pwd_hash:
        logger.warning("Dashboard login disabled: set DASHBOARD_PASSWORD or DASHBOARD_PASSWORD_HASH.")
        return False
    return check_password_hash(pwd_hash, password)


def make_password_hash(password: str) -> str:
    """Utility for generating a hash to put in DASHBOARD_PASSWORD_HASH.

    Uses pbkdf2:sha256 (universally available) rather than scrypt, which is
    missing on some OpenSSL/LibreSSL builds.
    """
    return generate_password_hash(password, method="pbkdf2:sha256")


# ─── Encrypted Fyers token storage ─────────────────────────────────

def _fernet():
    key = settings.token_encryption_key
    if not key:
        return None
    try:
        from cryptography.fernet import Fernet

        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as exc:  # pragma: no cover
        logger.error("Invalid TOKEN_ENCRYPTION_KEY: %s", exc)
        return None


def save_fyers_token(token: str) -> None:
    """Persist the Fyers access token, encrypted when a key is configured."""
    if not token:
        return
    f = _fernet()
    try:
        if f is not None:
            data = f.encrypt(token.encode())
            with open(_TOKEN_FILE, "wb") as fh:
                fh.write(data)
        else:
            # Dev fallback (no key): store plaintext but warn loudly.
            logger.warning("TOKEN_ENCRYPTION_KEY not set; storing token unencrypted (dev only).")
            with open(_TOKEN_FILE, "w") as fh:
                fh.write(token)
    except OSError as exc:
        logger.error("Failed to persist Fyers token: %s", exc)


def load_fyers_token() -> str | None:
    """Load the persisted Fyers access token, decrypting if needed."""
    if not os.path.exists(_TOKEN_FILE):
        return None
    f = _fernet()
    try:
        if f is not None:
            with open(_TOKEN_FILE, "rb") as fh:
                return f.decrypt(fh.read()).decode()
        with open(_TOKEN_FILE, "r") as fh:
            return fh.read().strip()
    except Exception as exc:
        logger.error("Failed to load Fyers token: %s", exc)
        return None


def clear_fyers_token() -> None:
    try:
        if os.path.exists(_TOKEN_FILE):
            os.remove(_TOKEN_FILE)
    except OSError:
        pass


def generate_encryption_key() -> str:
    """Generate a fresh Fernet key (for TOKEN_ENCRYPTION_KEY)."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()

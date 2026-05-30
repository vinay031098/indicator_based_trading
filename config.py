"""
Central, typed configuration (Task 14).

All environment access goes through the single `settings` object so the rest of
the codebase never touches os.environ directly. In production, required secrets
must be present or the app refuses to start (fail-fast).
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env from the project root (not the process cwd — gunicorn/cron often start elsewhere).
_PROJECT_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE) if _ENV_FILE.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─── Runtime ───────────────────────────────────────────────────
    production: bool = Field(default=False, alias="PRODUCTION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    domain: str = Field(default="localhost:5000", alias="DOMAIN")

    # ─── Flask ─────────────────────────────────────────────────────
    secret_key: str = Field(default="dev-only-insecure-key-change-me", alias="SECRET_KEY")

    # ─── Dashboard auth (server-side) ──────────────────────────────
    # Store ONLY a werkzeug password hash here, never a plaintext password.
    dashboard_user: str = Field(default="Trader", alias="DASHBOARD_USER")
    dashboard_password_hash: str = Field(default="", alias="DASHBOARD_PASSWORD_HASH")

    # ─── Fyers ─────────────────────────────────────────────────────
    fyers_app_id: str = Field(default="", alias="FYERS_APP_ID")
    fyers_secret_id: str = Field(default="", alias="FYERS_SECRET_ID")
    fyers_redirect_uri: str = Field(
        default="", alias="FYERS_REDIRECT_URI"
    )
    # Fyers v3 API — live apps use api-t1 for OAuth, trading, and market data.
    # api.fyers.in returns HTTP 500 "Invalid Request, please provide valid method".
    fyers_api_base: str = Field(
        default="https://api-t1.fyers.in/api/v3", alias="FYERS_API_BASE"
    )
    fyers_data_api_base: str = Field(
        default="https://api-t1.fyers.in/data", alias="FYERS_DATA_API_BASE"
    )
    fyers_auth_api_base: str = Field(
        default="https://api-t1.fyers.in/api/v3", alias="FYERS_AUTH_API_BASE"
    )
    # Key used to encrypt the persisted Fyers token at rest (Fernet key, base64).
    token_encryption_key: str = Field(default="", alias="TOKEN_ENCRYPTION_KEY")

    # ─── LLM ───────────────────────────────────────────────────────
    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # ─── Persistence / infra ───────────────────────────────────────
    database_url: str = Field(
        default="sqlite:///analysis_data.db", alias="DATABASE_URL"
    )
    redis_url: str = Field(default="", alias="REDIS_URL")

    # ─── Analysis defaults ─────────────────────────────────────────
    min_score_threshold: int = Field(default=2, alias="MIN_SCORE_THRESHOLD")
    analysis_period_days: int = Field(default=365, alias="ANALYSIS_PERIOD_DAYS")

    @property
    def base_url(self) -> str:
        scheme = "https" if self.production else "http"
        return f"{scheme}://{self.domain}"

    @property
    def jobs_enabled(self) -> bool:
        return bool(self.redis_url)

    @property
    def effective_fyers_redirect_uri(self) -> str:
        """OAuth redirect URI — must match EXACTLY what is registered in the Fyers app dashboard.

        Production (belezabrasileiro.com): ``https://<domain>/auth/callback`` (see deploy/deploy.sh).
        Override any time with ``FYERS_REDIRECT_URI`` in .env.
        """
        explicit = (self.fyers_redirect_uri or "").strip()
        if explicit:
            return explicit
        if self.production:
            return f"{self.base_url.rstrip('/')}/auth/callback"
        # Local dev fallback when not configured (paste-auth-code flow).
        return "https://fyersapiapp.com"

    def validate_production(self) -> list[str]:
        """Return a list of misconfiguration errors for production mode."""
        problems: list[str] = []
        if not self.production:
            return problems
        if self.secret_key == "dev-only-insecure-key-change-me":
            problems.append("SECRET_KEY must be set to a strong random value in production.")
        if not self.dashboard_password_hash:
            problems.append("DASHBOARD_PASSWORD_HASH must be set in production.")
        if not self.fyers_app_id or not self.fyers_secret_id:
            problems.append("FYERS_APP_ID and FYERS_SECRET_ID must be set in production.")
        if not self.token_encryption_key:
            problems.append("TOKEN_ENCRYPTION_KEY must be set in production.")
        return problems


@lru_cache
def get_settings() -> Settings:
    try:
        s = Settings()
    except ValidationError as exc:  # pragma: no cover - startup guard
        print(f"FATAL: invalid configuration:\n{exc}", file=sys.stderr)
        raise

    problems = s.validate_production()
    if problems:
        msg = "FATAL: production configuration errors:\n" + "\n".join(
            f"  - {p}" for p in problems
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)
    return s


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once (Task 15)."""
    lvl = (level or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Eagerly available singleton for convenient imports.
settings = get_settings()

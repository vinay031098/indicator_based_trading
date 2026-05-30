"""Meta routes: landing page and health check (Task 13)."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, render_template

from app.auth import is_authenticated
from config import settings
from fyers_integration import provider

meta_bp = Blueprint("meta", __name__)


@meta_bp.route("/")
def index():
    return render_template(
        "index.html",
        authenticated=provider.is_connected(),
        production=settings.production,
        logged_in=is_authenticated(),
        today=datetime.now().strftime("%Y-%m-%d"),
    )


@meta_bp.route("/healthz")
def healthz():
    """Public liveness/readiness probe."""
    db_ok = True
    try:
        from db import get_available_dates

        get_available_dates()
    except Exception:
        db_ok = False
    return jsonify(
        {
            "status": "ok" if db_ok else "degraded",
            "db": db_ok,
            "fyers_token": provider.is_connected(),
            "fyers_configured": bool(
                (settings.fyers_app_id or "").strip()
                and (settings.fyers_secret_id or "").strip()
            ),
            "dashboard_auth": bool(
                (settings.dashboard_password or "").strip()
                or (settings.dashboard_password_hash or "").strip()
            ),
            "jobs": "redis" if settings.jobs_enabled else "in-process",
            "time": datetime.now().isoformat(),
        }
    ), (200 if db_ok else 503)

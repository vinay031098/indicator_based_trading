"""Flask application factory (Task 10)."""

from __future__ import annotations

import logging
import os
from datetime import timedelta

from flask import Flask

from config import configure_logging, settings

logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_app() -> Flask:
    configure_logging()

    app = Flask(
        __name__,
        template_folder=os.path.join(_BASE_DIR, "templates"),
        static_folder=os.path.join(_BASE_DIR, "static"),
    )
    app.config.update(
        SECRET_KEY=settings.secret_key,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.production,
        JSON_SORT_KEYS=False,
        WTF_CSRF_TIME_LIMIT=None,
    )

    _init_extensions(app)
    _register_blueprints(app)

    from app.errors import register_error_handlers

    register_error_handlers(app)

    # Ensure DB schema exists.
    from db import init_db

    init_db()

    logger.info("App created (production=%s, jobs=%s)", settings.production, settings.jobs_enabled)
    return app


def _init_extensions(app: Flask) -> None:
    from app.extensions import HAS_CACHE, HAS_CSRF, HAS_LIMITER, cache, csrf, limiter

    if HAS_LIMITER and limiter is not None:
        # Always in-memory for the web app. Render's REDIS_URL is for RQ workers only;
        # using Redis here breaks rate-limited routes (e.g. /auth/login) when Redis is
        # unavailable or misconfigured on free tier.
        app.config["RATELIMIT_STORAGE_URI"] = "memory://"
        limiter.init_app(app)

    if HAS_CSRF and csrf is not None:
        csrf.init_app(app)
        # CSRF is enforced for forms; our JSON API uses the X-CSRFToken header.

    if HAS_CACHE and cache is not None:
        if settings.redis_url:
            try:
                cache.init_app(
                    app,
                    config={"CACHE_TYPE": "RedisCache", "CACHE_REDIS_URL": settings.redis_url},
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis cache unavailable (%s); using SimpleCache.", exc)
                cache.init_app(app)
        else:
            cache.init_app(app)

    # Security headers (Task 6) — only enforce HTTPS/HSTS in production.
    try:
        from flask_talisman import Talisman

        csp = {
            "default-src": "'self'",
            "script-src": ["'self'", "'unsafe-inline'", "https://unpkg.com", "https://cdn.jsdelivr.net"],
            "style-src": ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
            "font-src": ["'self'", "https://fonts.gstatic.com"],
            "img-src": ["'self'", "data:"],
            "connect-src": "'self'",
        }
        Talisman(
            app,
            force_https=settings.production,
            strict_transport_security=settings.production,
            session_cookie_secure=settings.production,
            content_security_policy=csp,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Talisman not enabled: %s", exc)


def _register_blueprints(app: Flask) -> None:
    from app.routes import ALL_BLUEPRINTS

    for bp in ALL_BLUEPRINTS:
        app.register_blueprint(bp)

    # Exempt the JSON API from form-CSRF (it uses the X-CSRFToken header instead),
    # keeping CSRF protection available for any future server-rendered forms.
    from app.extensions import HAS_CSRF, csrf

    if HAS_CSRF and csrf is not None:
        for bp in ALL_BLUEPRINTS:
            csrf.exempt(bp)

"""WSGI entry point for Gunicorn (production) and `python wsgi.py` (local)."""

from __future__ import annotations

from app import create_app

app = create_app()


if __name__ == "__main__":
    from config import settings

    app.run(host="0.0.0.0", port=5000, debug=not settings.production)

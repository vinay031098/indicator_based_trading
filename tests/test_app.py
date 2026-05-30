"""
Web-layer smoke tests for the Flask app factory, auth gating, and stored data.

These set required env vars BEFORE importing the app so the config singleton picks
them up, then exercise the public HTTP contract via Flask's test client.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# Configure a minimal, deterministic environment before importing app/config.
# Use an isolated temp-file SQLite DB (not :memory:, which is per-connection).
_DB_FD, _DB_PATH = tempfile.mkstemp(suffix=".db")
os.environ.setdefault("PRODUCTION", "0")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_DB_PATH}")

from werkzeug.security import generate_password_hash  # noqa: E402

_TEST_PASSWORD = "pytestpass123"
os.environ["DASHBOARD_USER"] = "Trader"
os.environ["DASHBOARD_PASSWORD_HASH"] = generate_password_hash(
    _TEST_PASSWORD, method="pbkdf2:sha256"
)

from app import create_app  # noqa: E402


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as c:
        yield c


def _login(client):
    return client.post(
        "/auth/dashboard-login",
        json={"username": "Trader", "password": _TEST_PASSWORD},
    )


def test_healthz_is_public(client):
    r = client.get("/healthz")
    assert r.status_code in (200, 503)
    assert "status" in r.get_json()


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert len(r.data) > 0


def test_api_requires_auth(client):
    r = client.get("/api/stored-dates")
    assert r.status_code == 401
    assert r.get_json()["code"] == "auth_required"


def test_login_rejects_bad_password(client):
    r = client.post("/auth/dashboard-login", json={"username": "Trader", "password": "nope"})
    assert r.status_code == 401


def test_login_then_access_and_logout(client):
    assert _login(client).status_code == 200
    r = client.get("/api/stored-dates")
    assert r.status_code == 200
    assert "dates" in r.get_json()
    assert client.post("/auth/logout").status_code == 200
    assert client.get("/api/stored-dates").status_code == 401


def test_status_endpoint(client):
    _login(client)
    r = client.get("/api/status")
    assert r.status_code == 200
    assert "authenticated" in r.get_json()

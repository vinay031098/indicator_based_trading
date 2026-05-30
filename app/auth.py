"""
Authentication helpers (Tasks 3, 4): server-side session login + guard decorator.
"""

from __future__ import annotations

from functools import wraps

from flask import session

from app.errors import AuthError

_SESSION_KEY = "dashboard_authed"


def login_user() -> None:
    session[_SESSION_KEY] = True
    session.permanent = True


def logout_user() -> None:
    session.pop(_SESSION_KEY, None)


def is_authenticated() -> bool:
    return bool(session.get(_SESSION_KEY))


def login_required(fn):
    """Guard API routes; returns JSON 401 when no valid session (Task 4)."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            raise AuthError("Authentication required. Please log in.")
        return fn(*args, **kwargs)

    return wrapper

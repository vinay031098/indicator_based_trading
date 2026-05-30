"""Auth routes: server-side dashboard login + Fyers OAuth (Tasks 3, 4, 5)."""

from __future__ import annotations

import logging
import urllib.parse

from flask import Blueprint, jsonify, redirect, request, url_for

from app.auth import is_authenticated, login_required, login_user, logout_user
from app.errors import AuthError, ValidationError
from app.extensions import rate_limit
from config import settings
from fyers_integration import provider
from security import verify_dashboard_login

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


# ─── Dashboard login (server-side) ─────────────────────────────────
@auth_bp.route("/auth/dashboard-login", methods=["POST"])
@rate_limit("10 per minute")
def dashboard_login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if verify_dashboard_login(username, password):
        login_user()
        return jsonify({"success": True})
    raise AuthError("Invalid username or password.")


@auth_bp.route("/auth/logout", methods=["POST"])
def dashboard_logout():
    logout_user()
    return jsonify({"success": True})


@auth_bp.route("/auth/session")
def session_status():
    return jsonify({"logged_in": is_authenticated()})


# ─── Fyers OAuth (public endpoints — same as pre-refactor app.py) ───
@auth_bp.route("/auth/login")
@rate_limit("10 per minute")
def fyers_login():
    """Return Fyers login URL. Frontend opens it in a new tab (no dashboard gate)."""
    try:
        client = provider.build_auth_client()
        return jsonify({
            "auth_url": client.generate_auth_url(),
            "redirect_uri": settings.effective_fyers_redirect_uri,
            "client_id": settings.fyers_app_id.strip(),
        })
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


@auth_bp.route("/auth/callback")
def fyers_callback():
    """Production redirect target from Fyers (when redirect URI is your domain)."""
    auth_code = request.args.get("auth_code", "")
    status = request.args.get("s", "")
    if status != "ok" or not auth_code:
        return redirect(url_for("meta.index"))
    if provider.connect_with_auth_code(auth_code):
        logger.info("Fyers connected via callback.")
    return redirect(url_for("meta.index"))


@auth_bp.route("/auth/connect", methods=["POST"])
@rate_limit("10 per minute")
def fyers_connect():
    """Accept auth_code (or full redirect URL) from frontend — public like old app."""
    data = request.get_json(silent=True) or {}
    auth_input = (data.get("auth_code") or "").strip()
    if "auth_code=" in auth_input:
        parsed = urllib.parse.urlparse(auth_input)
        params = urllib.parse.parse_qs(parsed.query)
        auth_input = params.get("auth_code", [""])[0]
    if not auth_input:
        raise ValidationError("No auth_code found.")
    try:
        success = provider.connect_with_auth_code(auth_input)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if not success:
        return jsonify({"success": False, "error": "Token exchange failed. Check auth_code and Fyers app credentials."})
    return jsonify({"success": True})


@auth_bp.route("/api/status")
@login_required
def api_status():
    return jsonify({"authenticated": provider.is_connected()})

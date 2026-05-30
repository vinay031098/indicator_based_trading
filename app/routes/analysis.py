"""Analysis routes: live analyze, background run-and-store, job polling (Task 12)."""

from __future__ import annotations

from datetime import datetime

from flask import Blueprint, jsonify, request

from app.auth import login_required
from app.errors import AuthError, NotFoundError, ValidationError
from app.extensions import rate_limit
from analysis_service import analyze
from db import get_run_by_date
from fyers_integration import provider
from jobs import enqueue_analysis, get_job

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/api/analyze", methods=["POST"])
@login_required
@rate_limit("6 per minute")
def api_analyze():
    """Synchronous live analysis (best for small universes like NIFTY 50/100)."""
    client = provider.get_client()
    if client is None:
        raise AuthError("Not connected to Fyers. Please connect first.")

    data = request.get_json(silent=True) or {}
    analysis_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    min_score = int(data.get("min_score", 2))
    category = data.get("category", "nifty50")
    try:
        payload = analyze(client, analysis_date, category, min_score)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return jsonify(payload)


@analysis_bp.route("/api/run-daily", methods=["POST"])
@login_required
@rate_limit("3 per minute")
def api_run_daily():
    """Enqueue a full analyze+AI+store job. Returns a job_id to poll."""
    client = provider.get_client()
    if client is None:
        raise AuthError("Not connected to Fyers. Please connect first.")

    data = request.get_json(silent=True) or {}
    analysis_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    category = data.get("category", "all")
    min_score = int(data.get("min_score", 2))
    skip_ai = bool(data.get("skip_ai", False))

    existing = get_run_by_date(analysis_date, category)
    if existing and not data.get("force", False):
        return jsonify({"exists": True, "message": f"Analysis already exists for {analysis_date}."})

    job_id = enqueue_analysis(analysis_date, category, min_score, skip_ai)
    return jsonify({"job_id": job_id, "status": "queued"})


@analysis_bp.route("/api/jobs/<job_id>")
@login_required
def api_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise NotFoundError("Job not found.")
    return jsonify(job)

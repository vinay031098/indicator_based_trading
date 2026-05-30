"""
Custom exceptions and centralized error handling (Task 16).

All API errors share one JSON shape: {"error": <message>, "code": <slug>}.
Unexpected exceptions return 500 with a safe message; details go to logs only.
"""

from __future__ import annotations

import logging

from flask import jsonify

logger = logging.getLogger(__name__)


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class AuthError(AppError):
    status_code = 401
    code = "auth_required"


class ValidationError(AppError):
    status_code = 400
    code = "validation_error"


class FyersError(AppError):
    status_code = 502
    code = "fyers_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


def register_error_handlers(app) -> None:
    @app.errorhandler(AppError)
    def _handle_app_error(err: AppError):
        if err.status_code >= 500:
            logger.error("AppError: %s", err.message)
        return jsonify({"error": err.message, "code": err.code}), err.status_code

    @app.errorhandler(404)
    def _handle_404(err):
        return jsonify({"error": "Not found", "code": "not_found"}), 404

    @app.errorhandler(429)
    def _handle_429(err):
        return (
            jsonify({"error": "Too many requests. Please slow down.", "code": "rate_limited"}),
            429,
        )

    @app.errorhandler(Exception)
    def _handle_unexpected(err):
        logger.exception("Unhandled exception: %s", err)
        return (
            jsonify({"error": "An unexpected error occurred.", "code": "internal_error"}),
            500,
        )

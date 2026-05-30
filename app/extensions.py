"""
Flask extension singletons (Tasks 5, 6, 7, 33).

Extensions are created here unbound and initialised in the app factory so they
can be imported anywhere without circular imports. All are optional/defensive:
if an extension package is missing, a no-op stub is used so the app still boots.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ─── Rate limiting (Task 5) ────────────────────────────────────────
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    HAS_LIMITER = True
except Exception:  # pragma: no cover
    limiter = None
    HAS_LIMITER = False
    logger.warning("flask-limiter not installed; rate limiting disabled.")


# ─── CSRF (Task 7) ─────────────────────────────────────────────────
try:
    from flask_wtf import CSRFProtect

    csrf = CSRFProtect()
    HAS_CSRF = True
except Exception:  # pragma: no cover
    csrf = None
    HAS_CSRF = False
    logger.warning("flask-wtf not installed; CSRF protection disabled.")


# ─── Caching (Task 33) ─────────────────────────────────────────────
try:
    from flask_caching import Cache

    cache = Cache(config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 60})
    HAS_CACHE = True
except Exception:  # pragma: no cover
    cache = None
    HAS_CACHE = False
    logger.warning("flask-caching not installed; caching disabled.")


def cached(timeout: int = 60, **kwargs):
    """Decorator that no-ops when caching is unavailable."""
    if HAS_CACHE and cache is not None:
        return cache.cached(timeout=timeout, **kwargs)

    def _noop(fn):
        return fn

    return _noop


def rate_limit(*args, **kwargs):
    """Decorator that no-ops when limiter is unavailable."""
    if HAS_LIMITER and limiter is not None:
        return limiter.limit(*args, **kwargs)

    def _noop(fn):
        return fn

    return _noop

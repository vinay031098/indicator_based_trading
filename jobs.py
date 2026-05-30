"""
Background job manager (Task 12).

Uses RQ + Redis when REDIS_URL is configured; otherwise falls back to running
the job in a daemon thread and tracking progress in memory. Either way the API
returns a job_id immediately and the frontend polls /api/jobs/<id> for progress.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, Optional

from config import settings

logger = logging.getLogger(__name__)

# In-memory job registry for the fallback path.
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


def _set(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def get_job(job_id: str) -> Optional[Dict]:
    # Try RQ first when enabled.
    if settings.jobs_enabled:
        try:
            from redis import Redis
            from rq.job import Job

            conn = Redis.from_url(settings.redis_url)
            job = Job.fetch(job_id, connection=conn)
            meta = job.meta or {}
            return {
                "id": job_id,
                "status": job.get_status(),
                "progress": meta.get("progress", 0),
                "total": meta.get("total", 0),
                "found": meta.get("found", 0),
                "result": job.result if job.is_finished else None,
                "error": str(job.exc_info) if job.is_failed else None,
            }
        except Exception as exc:
            logger.debug("RQ fetch failed, checking memory: %s", exc)
    with _jobs_lock:
        return _jobs.get(job_id)


def enqueue_analysis(analysis_date: str, category: str, min_score: int, skip_ai: bool) -> str:
    """Start a run-and-store job; returns a job_id."""
    if settings.jobs_enabled:
        try:
            from redis import Redis
            from rq import Queue

            conn = Redis.from_url(settings.redis_url)
            q = Queue("analysis", connection=conn)
            job = q.enqueue(
                "jobs.worker_run_and_store",
                analysis_date, category, min_score, skip_ai,
                job_timeout=3600,
            )
            return job.id
        except Exception as exc:
            logger.warning("Falling back to in-process job (RQ unavailable): %s", exc)

    job_id = uuid.uuid4().hex
    _set(job_id, id=job_id, status="queued", progress=0, total=0, found=0, result=None, error=None)
    t = threading.Thread(
        target=_run_inline, args=(job_id, analysis_date, category, min_score, skip_ai), daemon=True
    )
    t.start()
    return job_id


def _run_inline(job_id: str, analysis_date: str, category: str, min_score: int, skip_ai: bool) -> None:
    _set(job_id, status="started")

    def progress_cb(done, total, found):
        _set(job_id, progress=done, total=total, found=found)

    try:
        from analysis_service import run_and_store
        from fyers_integration import provider

        client = provider.get_client()
        if client is None:
            _set(job_id, status="failed", error="Not connected to Fyers.")
            return
        result = run_and_store(client, analysis_date, category, min_score, skip_ai, progress_cb)
        _set(job_id, status="finished", result=result, progress=_jobs[job_id].get("total", 0))
    except Exception as exc:
        logger.exception("Inline job %s failed", job_id)
        _set(job_id, status="failed", error=str(exc))


# ─── RQ worker entrypoint ──────────────────────────────────────────
def worker_run_and_store(analysis_date: str, category: str, min_score: int, skip_ai: bool) -> Dict:
    from rq import get_current_job

    from analysis_service import run_and_store
    from fyers_integration import provider

    job = get_current_job()

    def progress_cb(done, total, found):
        if job:
            job.meta.update({"progress": done, "total": total, "found": found})
            job.save_meta()

    client = provider.get_client()
    if client is None:
        raise RuntimeError("Not connected to Fyers.")
    return run_and_store(client, analysis_date, category, min_score, skip_ai, progress_cb)

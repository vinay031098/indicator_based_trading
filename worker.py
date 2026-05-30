#!/usr/bin/env python3
"""
RQ worker entry point (Task 12).

Run alongside the web service when REDIS_URL is set:
  python worker.py
"""

from __future__ import annotations

import sys

from config import configure_logging, settings


def main() -> int:
    configure_logging()
    if not settings.redis_url:
        print("REDIS_URL not set — background worker not needed (in-process fallback in use).")
        return 0
    from redis import Redis
    from rq import Queue, Worker

    conn = Redis.from_url(settings.redis_url)
    worker = Worker([Queue("analysis", connection=conn)], connection=conn)
    worker.work(with_scheduler=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

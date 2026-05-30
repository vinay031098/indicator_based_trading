#!/usr/bin/env python3
"""
Daily Runner — full analysis + AI recommendations for an NSE universe, stored in DB.

Usage:
  python daily_runner.py                     # ALL NSE stocks for today
  python daily_runner.py --date 2026-02-18   # specific date
  python daily_runner.py --category nifty50  # specific universe
  python daily_runner.py --skip-ai           # indicators only
  python daily_runner.py --force             # overwrite existing run

Cron (after market hours, weekdays):
  30 16 * * 1-5 cd /path/to/project && python daily_runner.py --force >> logs/daily.log 2>&1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime

from config import configure_logging
from analysis_service import run_and_store
from db import get_run_by_date
from fyers_integration import provider

logger = logging.getLogger(__name__)


def run_daily_analysis(analysis_date: str, category: str = "all", min_score: int = 2,
                       skip_ai: bool = False, force: bool = False) -> None:
    start = time.time()
    logger.info("Daily analysis: date=%s category=%s skip_ai=%s", analysis_date, category, skip_ai)

    existing = get_run_by_date(analysis_date, category)
    if existing and not force:
        resp = input(f"Analysis exists for {analysis_date} ({category}). Overwrite? (y/N): ").strip().lower()
        if resp != "y":
            logger.info("Skipped.")
            return

    client = provider.get_client()
    if client is None:
        logger.error("No saved Fyers token. Connect via the dashboard first.")
        return

    def progress(done, total, found):
        logger.info("  [%d/%d] fetched (%d ok)", done, total, found)

    result = run_and_store(client, analysis_date, category, min_score, skip_ai, progress)
    elapsed = int(time.time() - start)
    ai = result.get("ai_stats")
    logger.info(
        "Done in %dm%ds — run_id=%s, %d stocks, %d qualified, AI=%s",
        elapsed // 60, elapsed % 60, result.get("run_id"),
        result.get("total_stocks"), result.get("qualified_count"),
        f"{ai['buy']}B/{ai['hold']}H/{ai['avoid']}A" if ai else "skipped",
    )


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Daily NSE Stock Analysis Runner")
    parser.add_argument("--date", "-d", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--category", "-c", default="all",
                        choices=["nifty50", "nifty100", "nifty200", "nifty500", "all"])
    parser.add_argument("--min-score", "-m", type=int, default=2)
    parser.add_argument("--skip-ai", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format: %s (use YYYY-MM-DD)", args.date)
        return 1

    run_daily_analysis(args.date, args.category, args.min_score, args.skip_ai, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

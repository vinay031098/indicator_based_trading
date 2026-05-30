"""
Analysis pipeline shared by the web routes, the background worker, and the CLI
daily runner (avoids duplicating the orchestration logic in three places).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

from db import (
    create_run,
    save_ai_recommendations,
    save_stock_analysis,
    update_run_status,
)
from fyers_integration import FyersClient, get_symbols_for_category
from indicators import analyze_stock

logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "nifty50": "NIFTY 50", "nifty100": "NIFTY 100", "nifty200": "NIFTY 200",
    "nifty500": "NIFTY 500", "all": "All NSE",
}


def category_label(category: str, count: int = 0) -> str:
    if category == "all":
        return f"All NSE ({count})" if count else "All NSE"
    return CATEGORY_LABELS.get(category, category)


def analyze(client: FyersClient, analysis_date: str, category: str = "nifty50",
            min_score: int = 2, progress_cb: Optional[Callable] = None) -> Dict:
    """Run indicator analysis for a date/category. Returns the full result payload."""
    try:
        target_date = datetime.strptime(analysis_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.") from exc

    days_from_now = (datetime.now() - target_date).days
    total_days = days_from_now + 365

    symbols = get_symbols_for_category(category)
    label = category_label(category, len(symbols))
    logger.info("Analyzing %s (%d symbols) for %s", label, len(symbols), analysis_date)

    all_data = client.get_stock_data(symbols, resolution="D", days=total_days, progress_cb=progress_cb)
    if not all_data:
        raise RuntimeError("No stock data retrieved. Fyers API may be down or token expired.")

    results: List[Dict] = []
    skipped: List[str] = []
    errors = 0
    for symbol, df in all_data.items():
        try:
            filtered = df[df.index <= pd.Timestamp(analysis_date)]
            if len(filtered) < 50:
                skipped.append(symbol.replace("NSE:", "").replace("-EQ", ""))
                continue
            result = analyze_stock(symbol, filtered)
            if result:
                results.append(result)
        except Exception as exc:
            errors += 1
            logger.debug("Error analyzing %s: %s", symbol, exc)

    results.sort(key=lambda x: x["score"], reverse=True)
    qualified = [r for r in results if r["score"] >= min_score]
    unqualified = [r for r in results if r["score"] < min_score]
    logger.info("Analysis done: %d analyzed, %d qualified, %d errors", len(results), len(qualified), errors)

    return {
        "date": analysis_date, "category": category, "category_label": label,
        "total_stocks": len(results), "qualified_count": len(qualified),
        "min_score": min_score, "qualified": qualified, "unqualified": unqualified,
        "skipped": skipped, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_and_store(client: FyersClient, analysis_date: str, category: str = "all",
                  min_score: int = 2, skip_ai: bool = False,
                  progress_cb: Optional[Callable] = None) -> Dict:
    """Full pipeline: analyze, optionally run AI, persist everything to the DB."""
    from llm_analyzer import analyze_with_llm  # local import keeps web import light

    run_id = create_run(analysis_date, category, min_score)
    try:
        payload = analyze(client, analysis_date, category, min_score, progress_cb=progress_cb)
        results = payload["qualified"] + payload["unqualified"]
        save_stock_analysis(run_id, results)

        ai_buy = ai_hold = ai_avoid = ai_completed = 0
        if not skip_ai and results:
            ai_result = analyze_with_llm(results, fyers_client=client)
            recs = ai_result.get("recommendations", {}) if isinstance(ai_result, dict) else {}
            if recs:
                save_ai_recommendations(run_id, recs)
                ai_buy = sum(1 for r in recs.values() if r.get("action") == "BUY")
                ai_hold = sum(1 for r in recs.values() if r.get("action") == "HOLD")
                ai_avoid = sum(1 for r in recs.values() if r.get("action") == "AVOID")
                ai_completed = 1

        update_run_status(
            run_id, "completed", total_stocks=len(results),
            qualified_count=payload["qualified_count"], ai_completed=ai_completed,
            ai_buy=ai_buy, ai_hold=ai_hold, ai_avoid=ai_avoid,
        )

        # Evaluate alert rules against the fresh results (Task 46). Best-effort.
        try:
            from alerts import evaluate_and_notify

            evaluate_and_notify(results)
        except Exception as exc:  # pragma: no cover
            logger.debug("Alert evaluation skipped: %s", exc)

        return {"run_id": run_id, "ok": True, **payload,
                "ai_stats": {"buy": ai_buy, "hold": ai_hold, "avoid": ai_avoid} if ai_completed else None}
    except Exception:
        update_run_status(run_id, "failed")
        raise

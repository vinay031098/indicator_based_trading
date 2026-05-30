"""
Backtesting module (Task 45) — historically validate the scoring strategy.

The engine performs a **walk-forward** simulation over a date range:

  * For every trading day ``d`` in ``[start_date, end_date]`` and every symbol,
    score the stock with :func:`indicators.analyze_stock` using **only** candles
    dated ``<= d`` (no look-ahead).
  * If the bullish ``score`` is ``>= min_score`` an entry is taken at the **next**
    day's open (falling back to the signal day's close if no next bar exists).
  * The position exits on whichever comes first:
        - price reaches the target (``+target_pct%``),
        - price reaches the stop (``-stop_pct%``),
        - ``hold_days`` bars elapse (exit at the last available close).
  * Only one open position per symbol at a time (a cooldown until the prior
    trade exits) to avoid overlapping/duplicated trades.

Returned metrics: trade count, wins/losses, win-rate, average & total return,
max drawdown and an equity curve (starting at 100), plus an optional per-symbol
breakdown.

Performance notes:
  * For broad universes (``nifty500`` / ``all``) the symbol set is capped at
    :data:`MAX_SYMBOLS` so a backtest finishes in a sane amount of time.
  * Scoring on each simulated day only looks at the trailing
    :data:`SCORING_LOOKBACK` candles, which is enough for every indicator
    (incl. the 200-SMA and 52-week window) while bounding per-call cost.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from fyers_integration import get_symbols_for_category
from indicators import analyze_stock

logger = logging.getLogger(__name__)

# Cap the universe for broad categories to keep runtime reasonable.
MAX_SYMBOLS = 150
# Trailing candles fed to analyze_stock on each simulated day. 260 trading days
# (~1y) covers the 200-SMA and 52-week high/low windows.
SCORING_LOOKBACK = 260
# Minimum history (in candles) required before a symbol can be scored.
MIN_HISTORY = 50
# Extra calendar days of history fetched before start_date so indicators warm up.
WARMUP_DAYS = 400


def _parse_date(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError:
        logger.warning("Backtest: invalid date %r (expected YYYY-MM-DD)", value)
        return None


def _simulate_trade(df: pd.DataFrame, signal_pos: int, target_pct: float,
                    stop_pct: float, hold_days: int) -> Optional[Dict]:
    """Simulate a single trade entered the bar **after** ``signal_pos``.

    Returns a trade dict, or ``None`` if there is no bar available to enter on.
    """
    entry_pos = signal_pos + 1
    n = len(df)
    if entry_pos >= n:
        # No next bar: fall back to entering at the signal day's close.
        entry_pos = signal_pos
        entry_price = float(df.iloc[signal_pos]["Close"])
    else:
        entry_price = float(df.iloc[entry_pos]["Open"])
    if entry_price <= 0:
        return None

    target_price = entry_price * (1.0 + target_pct / 100.0)
    stop_price = entry_price * (1.0 - stop_pct / 100.0)

    last_pos = min(entry_pos + hold_days, n - 1)
    exit_price = float(df.iloc[last_pos]["Close"])
    exit_pos = last_pos
    reason = "time"

    for pos in range(entry_pos, last_pos + 1):
        bar = df.iloc[pos]
        low = float(bar["Low"])
        high = float(bar["High"])
        # Conservative tie-break: assume the stop is touched before the target
        # when a single bar spans both levels.
        if low <= stop_price:
            exit_price = stop_price
            exit_pos = pos
            reason = "stop"
            break
        if high >= target_price:
            exit_price = target_price
            exit_pos = pos
            reason = "target"
            break

    return_pct = (exit_price - entry_price) / entry_price * 100.0
    return {
        "symbol": df.attrs.get("symbol", ""),
        "entry_date": df.index[entry_pos].date().isoformat(),
        "exit_date": df.index[exit_pos].date().isoformat(),
        "entry_price": round(entry_price, 2),
        "exit_price": round(exit_price, 2),
        "return_pct": round(return_pct, 3),
        "exit_reason": reason,
        "exit_pos": exit_pos,
    }


def _backtest_symbol(symbol: str, df: pd.DataFrame, start: datetime, end: datetime,
                     min_score: int, hold_days: int, target_pct: float,
                     stop_pct: float) -> List[Dict]:
    """Walk forward over one symbol's history and collect its trades."""
    if df is None or df.empty:
        return []
    df = df.sort_index()
    df.attrs["symbol"] = symbol
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    trades: List[Dict] = []
    n = len(df)
    cooldown_until = -1  # positions <= this are inside an open trade

    for pos in range(n):
        ts = df.index[pos]
        if ts < start_ts or ts > end_ts:
            continue
        if pos <= cooldown_until:
            continue
        if pos + 1 < MIN_HISTORY:  # not enough trailing candles to score
            continue

        window_start = max(0, pos + 1 - SCORING_LOOKBACK)
        window = df.iloc[window_start: pos + 1]
        try:
            res = analyze_stock(symbol, window)
        except Exception as exc:  # pragma: no cover - analyze_stock is defensive
            logger.debug("Backtest scoring failed for %s @ %s: %s", symbol, ts, exc)
            continue
        if not res or res.get("score", 0) < min_score:
            continue

        trade = _simulate_trade(df, pos, target_pct, stop_pct, hold_days)
        if trade is None:
            continue
        trades.append(trade)
        cooldown_until = trade.pop("exit_pos")

    return trades


def _equity_and_drawdown(trades: List[Dict], start_label: str):
    """Build an equity curve (compounding full allocation per trade) + max DD."""
    ordered = sorted(trades, key=lambda t: t["exit_date"])
    equity = 100.0
    curve = [{"date": start_label, "equity": 100.0}]
    peak = 100.0
    max_dd = 0.0
    for t in ordered:
        equity *= 1.0 + t["return_pct"] / 100.0
        curve.append({"date": t["exit_date"], "equity": round(equity, 4)})
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            max_dd = max(max_dd, dd)
    return curve, equity, max_dd


def _by_symbol(trades: List[Dict], limit: int = 20) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for t in trades:
        grouped.setdefault(t["symbol"], []).append(t)
    rows = []
    for sym, ts in grouped.items():
        rets = [t["return_pct"] for t in ts]
        wins = sum(1 for r in rets if r > 0)
        rows.append({
            "symbol": sym.replace("NSE:", "").replace("-EQ", ""),
            "trades": len(ts),
            "win_rate": round(wins / len(ts) * 100.0, 2) if ts else 0.0,
            "avg_return_pct": round(sum(rets) / len(rets), 3) if rets else 0.0,
        })
    rows.sort(key=lambda r: (r["trades"], r["avg_return_pct"]), reverse=True)
    return rows[:limit]


def run_backtest(category, start_date, end_date, min_score, hold_days,
                 target_pct, stop_pct, client=None) -> dict:
    """Backtest the scoring strategy over a date range.

    See module docstring for the strategy details. Returns a metrics dict, or
    ``{"error": ...}`` when it cannot run (no client / no data / bad inputs).
    """
    params = {
        "category": category,
        "start_date": start_date,
        "end_date": end_date,
        "min_score": min_score,
        "hold_days": hold_days,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
    }

    if client is None:
        return {"error": "Not connected to Fyers; backtesting requires live history access."}

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start is None or end is None:
        return {"error": "Invalid or missing start_date/end_date (use YYYY-MM-DD)."}
    if start > end:
        return {"error": "start_date must be on or before end_date."}

    try:
        hold_days = max(1, int(hold_days))
        min_score = int(min_score)
        target_pct = float(target_pct)
        stop_pct = float(stop_pct)
    except (TypeError, ValueError):
        return {"error": "min_score/hold_days/target_pct/stop_pct must be numeric."}

    symbols = get_symbols_for_category(category)
    if not symbols:
        return {"error": f"No symbols for category {category!r}."}
    if len(symbols) > MAX_SYMBOLS:
        logger.info("Backtest: capping universe %s from %d to %d symbols",
                    category, len(symbols), MAX_SYMBOLS)
        symbols = symbols[:MAX_SYMBOLS]

    # Fetch enough history to cover the warm-up window before start_date through
    # end_date. get_history fetches back `days` from "now", so size accordingly.
    days = (datetime.now() - start).days + WARMUP_DAYS
    days = max(days, WARMUP_DAYS)
    logger.info("Backtest: fetching %d symbols (%d days of history)", len(symbols), days)

    try:
        stock_data = client.get_stock_data(symbols, resolution="D", days=days)
    except Exception as exc:
        logger.exception("Backtest: data fetch failed: %s", exc)
        return {"error": f"Failed to fetch history: {exc}"}

    if not stock_data:
        return {"error": "No historical data retrieved (Fyers may be down or token expired)."}

    all_trades: List[Dict] = []
    for symbol, df in stock_data.items():
        try:
            all_trades.extend(
                _backtest_symbol(symbol, df, start, end, min_score,
                                 hold_days, target_pct, stop_pct)
            )
        except Exception as exc:
            logger.debug("Backtest: symbol %s failed: %s", symbol, exc)

    start_label = start.date().isoformat()
    curve, final_equity, max_dd = _equity_and_drawdown(all_trades, start_label)

    n_trades = len(all_trades)
    returns = [t["return_pct"] for t in all_trades]
    wins = sum(1 for r in returns if r > 0)
    losses = n_trades - wins

    result = {
        "trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n_trades * 100.0, 2) if n_trades else 0.0,
        "avg_return_pct": round(sum(returns) / n_trades, 3) if n_trades else 0.0,
        "total_return_pct": round(final_equity - 100.0, 3),
        "max_drawdown_pct": round(max_dd, 3),
        "equity_curve": curve,
        "by_symbol": _by_symbol(all_trades),
        "params": params,
    }
    logger.info(
        "Backtest done: %d trades, %.1f%% win-rate, %.2f%% total return, %.2f%% max DD",
        n_trades, result["win_rate"], result["total_return_pct"], result["max_drawdown_pct"],
    )
    return result

"""Data routes: stored analysis, history for charts, export, watchlist, alerts, backtest."""

from __future__ import annotations

import csv
import io

from flask import Blueprint, Response, jsonify, request

from app.auth import login_required
from app.errors import NotFoundError, ValidationError
from app.extensions import cached
from db import (
    add_alert_rule,
    add_to_watchlist,
    delete_alert_rule,
    get_available_dates,
    get_run_by_date,
    get_stored_analysis,
    list_alert_rules,
    list_watchlist,
    remove_from_watchlist,
)
from fyers_integration import provider

data_bp = Blueprint("data", __name__)


# ─── Stored analysis ───────────────────────────────────────────────
@data_bp.route("/api/stored-dates")
@login_required
@cached(timeout=30)
def api_stored_dates():
    return jsonify({"dates": get_available_dates()})


@data_bp.route("/api/stored-data", methods=["POST"])
@login_required
def api_stored_data():
    data = request.get_json(silent=True) or {}
    run_date = data.get("date", "")
    category = data.get("category", "all")
    min_score = int(data.get("min_score", 0))
    if not run_date:
        raise ValidationError("No date provided.")
    run = get_run_by_date(run_date, category)
    if not run:
        raise NotFoundError(f"No stored analysis for {run_date} ({category}).")
    return jsonify(get_stored_analysis(run["id"], min_score))


# ─── History for charts (Task 36) ──────────────────────────────────
@data_bp.route("/api/history/<path:symbol>")
@login_required
def api_history(symbol: str):
    client = provider.get_client()
    if client is None:
        raise NotFoundError("Not connected to Fyers; live history unavailable.")
    resolution = request.args.get("resolution", "D")
    days = int(request.args.get("days", 365))
    df = client.get_history(symbol, resolution=resolution, days=days)
    if df is None or df.empty:
        raise NotFoundError(f"No history for {symbol}.")
    candles = [
        {
            "time": int(idx.timestamp()),
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        for idx, row in df.iterrows()
    ]
    return jsonify({"symbol": symbol, "candles": candles})


# ─── Export (Task 47) ──────────────────────────────────────────────
@data_bp.route("/api/export")
@login_required
def api_export():
    run_date = request.args.get("date", "")
    category = request.args.get("category", "all")
    fmt = request.args.get("format", "csv")
    if not run_date:
        raise ValidationError("No date provided.")
    run = get_run_by_date(run_date, category)
    if not run:
        raise NotFoundError(f"No stored analysis for {run_date} ({category}).")
    payload = get_stored_analysis(run["id"], 0)
    stocks = payload["qualified"] + payload["unqualified"]
    if not stocks:
        raise NotFoundError("No stocks to export.")

    cols = ["name", "symbol", "price", "change_pct", "score", "bear_score", "signal",
            "rsi", "macd_hist", "adx", "mfi", "vol_ratio"]
    if fmt == "xlsx":
        return _export_xlsx(stocks, cols, run_date, category)
    return _export_csv(stocks, cols, run_date, category)


def _export_csv(stocks, cols, run_date, category):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for s in stocks:
        writer.writerow({c: s.get(c, "") for c in cols})
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=analysis_{category}_{run_date}.csv"},
    )


def _export_xlsx(stocks, cols, run_date, category):
    try:
        from openpyxl import Workbook
    except Exception:
        raise ValidationError("XLSX export unavailable (openpyxl not installed).")
    wb = Workbook()
    ws = wb.active
    ws.title = "Analysis"
    ws.append(cols)
    for s in stocks:
        ws.append([s.get(c, "") for c in cols])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return Response(
        out.read(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=analysis_{category}_{run_date}.xlsx"},
    )


# ─── Watchlist (Task 46) ───────────────────────────────────────────
@data_bp.route("/api/watchlist", methods=["GET"])
@login_required
def api_watchlist_get():
    return jsonify({"watchlist": list_watchlist()})


@data_bp.route("/api/watchlist", methods=["POST"])
@login_required
def api_watchlist_add():
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip()
    name = (data.get("name") or symbol).strip()
    if not symbol:
        raise ValidationError("symbol is required.")
    add_to_watchlist(symbol, name)
    return jsonify({"success": True})


@data_bp.route("/api/watchlist/<path:symbol>", methods=["DELETE"])
@login_required
def api_watchlist_remove(symbol: str):
    remove_from_watchlist(symbol)
    return jsonify({"success": True})


# ─── Alert rules (Task 46) ─────────────────────────────────────────
@data_bp.route("/api/alerts", methods=["GET"])
@login_required
def api_alerts_get():
    return jsonify({"alerts": list_alert_rules()})


@data_bp.route("/api/alerts", methods=["POST"])
@login_required
def api_alerts_add():
    data = request.get_json(silent=True) or {}
    metric = data.get("metric", "score")
    operator = data.get("operator", ">=")
    if operator not in (">=", "<=", "==", ">", "<"):
        raise ValidationError("Invalid operator.")
    rule_id = add_alert_rule(
        name=data.get("name", ""), metric=metric, operator=operator,
        threshold=float(data.get("threshold", 0)), channel=data.get("channel", "email"),
        target=data.get("target", ""),
    )
    return jsonify({"success": True, "id": rule_id})


@data_bp.route("/api/alerts/<int:rule_id>", methods=["DELETE"])
@login_required
def api_alerts_delete(rule_id: int):
    delete_alert_rule(rule_id)
    return jsonify({"success": True})


# ─── Backtest (Task 45) ────────────────────────────────────────────
@data_bp.route("/api/backtest", methods=["POST"])
@login_required
def api_backtest():
    try:
        from backtest import run_backtest
    except Exception:
        raise ValidationError("Backtest module unavailable.")
    data = request.get_json(silent=True) or {}
    result = run_backtest(
        category=data.get("category", "nifty50"),
        start_date=data.get("start_date"),
        end_date=data.get("end_date"),
        min_score=int(data.get("min_score", 8)),
        hold_days=int(data.get("hold_days", 5)),
        target_pct=float(data.get("target_pct", 5)),
        stop_pct=float(data.get("stop_pct", 3)),
        client=provider.get_client(),
    )
    return jsonify(result)

"""
Data Store — SQLite-based persistence for daily stock analysis & AI recommendations.
Stores complete analysis results so the dashboard can display historical data
without needing a live Fyers connection.

Tables:
  daily_runs: metadata for each analysis run (date, category, counts, timestamp)
  stock_analysis: full indicator data for each stock per run
  ai_recommendations: LLM buy/hold/avoid recommendations per stock per run
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_data.db")


def _get_conn():
    """Get SQLite connection with WAL mode for better concurrency."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'all',
            total_stocks INTEGER DEFAULT 0,
            qualified_count INTEGER DEFAULT 0,
            min_score INTEGER DEFAULT 2,
            ai_completed INTEGER DEFAULT 0,
            ai_buy_count INTEGER DEFAULT 0,
            ai_hold_count INTEGER DEFAULT 0,
            ai_avoid_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'running',
            UNIQUE(run_date, category)
        );

        CREATE TABLE IF NOT EXISTS stock_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            price REAL,
            change_val REAL,
            change_pct REAL,
            score INTEGER,
            rsi REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            sma20 REAL,
            sma50 REAL,
            sma200 REAL,
            ema9 REAL,
            ema21 REAL,
            stoch_k REAL,
            stoch_d REAL,
            bb_upper REAL,
            bb_lower REAL,
            bb_width REAL,
            w52_high REAL,
            w52_low REAL,
            dist_52w REAL,
            volume INTEGER,
            avg_volume INTEGER,
            vol_ratio REAL,
            atr REAL,
            atr_pct REAL,
            adx REAL,
            plus_di REAL,
            minus_di REAL,
            cci REAL,
            williams_r REAL,
            mfi REAL,
            vwap REAL,
            roc REAL,
            cmf REAL,
            ichimoku_tenkan REAL,
            ichimoku_kijun REAL,
            pivot REAL,
            pivot_s1 REAL,
            pivot_r1 REAL,
            reasons_json TEXT,
            FOREIGN KEY (run_id) REFERENCES daily_runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ai_recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            stock_name TEXT NOT NULL,
            action TEXT,
            confidence TEXT,
            reason TEXT,
            target REAL,
            stoploss REAL,
            risk_reward TEXT,
            FOREIGN KEY (run_id) REFERENCES daily_runs(id) ON DELETE CASCADE,
            UNIQUE(run_id, stock_name)
        );

        CREATE INDEX IF NOT EXISTS idx_stock_run ON stock_analysis(run_id);
        CREATE INDEX IF NOT EXISTS idx_stock_name ON stock_analysis(name);
        CREATE INDEX IF NOT EXISTS idx_ai_run ON ai_recommendations(run_id);
        CREATE INDEX IF NOT EXISTS idx_runs_date ON daily_runs(run_date);
    """)
    conn.commit()
    conn.close()


# ─── Write Operations ─────────────────────────────────────────────

def create_run(run_date: str, category: str = "all", min_score: int = 2) -> int:
    """Create a new daily run record. Returns run_id."""
    conn = _get_conn()
    try:
        # Delete existing run for same date+category (replace)
        conn.execute("DELETE FROM daily_runs WHERE run_date=? AND category=?", (run_date, category))
        cur = conn.execute(
            "INSERT INTO daily_runs (run_date, category, min_score, created_at, status) VALUES (?, ?, ?, ?, ?)",
            (run_date, category, min_score, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "running")
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def save_stock_analysis(run_id: int, stocks: List[Dict]):
    """Save analyzed stock data for a run."""
    conn = _get_conn()
    try:
        for s in stocks:
            conn.execute("""
                INSERT INTO stock_analysis (
                    run_id, symbol, name, price, change_val, change_pct, score,
                    rsi, macd, macd_signal, macd_hist,
                    sma20, sma50, sma200, ema9, ema21,
                    stoch_k, stoch_d, bb_upper, bb_lower, bb_width,
                    w52_high, w52_low, dist_52w,
                    volume, avg_volume, vol_ratio,
                    atr, atr_pct, adx, plus_di, minus_di,
                    cci, williams_r, mfi, vwap, roc, cmf,
                    ichimoku_tenkan, ichimoku_kijun,
                    pivot, pivot_s1, pivot_r1,
                    reasons_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_id, s.get('symbol', ''), s.get('name', ''),
                s.get('price', 0), s.get('change', 0), s.get('change_pct', 0), s.get('score', 0),
                s.get('rsi', 0), s.get('macd', 0), s.get('macd_signal', 0), s.get('macd_hist', 0),
                s.get('sma20', 0), s.get('sma50', 0), s.get('sma200', 0),
                s.get('ema9', 0), s.get('ema21', 0),
                s.get('stoch_k', 0), s.get('stoch_d', 0),
                s.get('bb_upper', 0), s.get('bb_lower', 0), s.get('bb_width', 0),
                s.get('w52_high', 0), s.get('w52_low', 0), s.get('dist_52w', 0),
                s.get('volume', 0), s.get('avg_volume', 0), s.get('vol_ratio', 0),
                s.get('atr', 0), s.get('atr_pct', 0),
                s.get('adx', 0), s.get('plus_di', 0), s.get('minus_di', 0),
                s.get('cci', 0), s.get('williams_r', 0), s.get('mfi', 0),
                s.get('vwap', 0), s.get('roc', 0), s.get('cmf', 0),
                s.get('ichimoku_tenkan', 0), s.get('ichimoku_kijun', 0),
                s.get('pivot', 0), s.get('pivot_s1', 0), s.get('pivot_r1', 0),
                json.dumps(s.get('reasons', []))
            ))
        conn.commit()
    finally:
        conn.close()


def save_ai_recommendations(run_id: int, recommendations: Dict):
    """Save AI recommendations for a run. recommendations = {name: {action, confidence, ...}}"""
    conn = _get_conn()
    try:
        for name, rec in recommendations.items():
            conn.execute("""
                INSERT OR REPLACE INTO ai_recommendations
                (run_id, stock_name, action, confidence, reason, target, stoploss, risk_reward)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, name,
                rec.get('action', 'HOLD'),
                rec.get('confidence', 'LOW'),
                rec.get('reason', ''),
                rec.get('target', 0),
                rec.get('stoploss', 0),
                rec.get('risk_reward', 'N/A')
            ))
        conn.commit()
    finally:
        conn.close()


def update_run_status(run_id: int, status: str, total_stocks: int = 0,
                      qualified_count: int = 0, ai_completed: int = 0,
                      ai_buy: int = 0, ai_hold: int = 0, ai_avoid: int = 0):
    """Update run metadata after completion."""
    conn = _get_conn()
    try:
        conn.execute("""
            UPDATE daily_runs SET
                status=?, total_stocks=?, qualified_count=?,
                ai_completed=?, ai_buy_count=?, ai_hold_count=?, ai_avoid_count=?
            WHERE id=?
        """, (status, total_stocks, qualified_count, ai_completed, ai_buy, ai_hold, ai_avoid, run_id))
        conn.commit()
    finally:
        conn.close()


# ─── Read Operations ──────────────────────────────────────────────

def get_available_dates() -> List[Dict]:
    """Get list of all completed analysis runs with metadata."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT id, run_date, category, total_stocks, qualified_count,
                   ai_completed, ai_buy_count, ai_hold_count, ai_avoid_count,
                   created_at, status
            FROM daily_runs
            WHERE status = 'completed'
            ORDER BY run_date DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_run_by_date(run_date: str, category: str = "all") -> Optional[Dict]:
    """Get a specific run by date and category."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM daily_runs WHERE run_date=? AND category=? AND status='completed'",
            (run_date, category)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_stored_analysis(run_id: int, min_score: int = 0) -> Dict:
    """Get full analysis data for a run — stocks + AI recommendations."""
    conn = _get_conn()
    try:
        # Get run metadata
        run = conn.execute("SELECT * FROM daily_runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            return {"error": "Run not found"}
        run = dict(run)

        # Get all stocks for this run
        stock_rows = conn.execute(
            "SELECT * FROM stock_analysis WHERE run_id=? ORDER BY score DESC",
            (run_id,)
        ).fetchall()

        stocks = []
        for row in stock_rows:
            s = dict(row)
            # Parse reasons JSON back to list
            s['reasons'] = json.loads(s.get('reasons_json', '[]'))
            del s['reasons_json']
            # Rename change_val back to change
            s['change'] = s.pop('change_val', 0)
            # Remove internal IDs
            s.pop('id', None)
            s.pop('run_id', None)
            stocks.append(s)

        # Get AI recommendations
        ai_rows = conn.execute(
            "SELECT stock_name, action, confidence, reason, target, stoploss, risk_reward FROM ai_recommendations WHERE run_id=?",
            (run_id,)
        ).fetchall()

        ai_recs = {}
        for row in ai_rows:
            r = dict(row)
            name = r.pop('stock_name')
            ai_recs[name] = r

        # Split qualified / unqualified
        effective_min = min_score if min_score > 0 else run.get('min_score', 2)
        qualified = [s for s in stocks if s['score'] >= effective_min]
        unqualified = [s for s in stocks if s['score'] < effective_min]

        return {
            "date": run['run_date'],
            "category": run['category'],
            "category_label": _category_label(run['category'], len(stocks)),
            "total_stocks": len(stocks),
            "qualified_count": len(qualified),
            "min_score": effective_min,
            "qualified": qualified,
            "unqualified": unqualified,
            "skipped": [],
            "timestamp": run['created_at'],
            "stored": True,
            "ai_recommendations": ai_recs if ai_recs else None,
            "ai_stats": {
                "buy": run.get('ai_buy_count', 0),
                "hold": run.get('ai_hold_count', 0),
                "avoid": run.get('ai_avoid_count', 0),
            } if run.get('ai_completed') else None,
        }
    finally:
        conn.close()


def _category_label(category: str, count: int) -> str:
    labels = {
        'nifty50': 'NIFTY 50',
        'nifty100': 'NIFTY 100',
        'nifty200': 'NIFTY 200',
        'nifty500': 'NIFTY 500',
        'all': f'All NSE ({count})',
    }
    return labels.get(category, f'{category} ({count})')


# Initialize DB on import
init_db()

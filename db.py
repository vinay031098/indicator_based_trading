"""
Persistence layer (Tasks 30, 31, 32) — SQLAlchemy models + read/write helpers.

Supports any database via DATABASE_URL:
  - dev default: sqlite:///analysis_data.db
  - production:  postgresql+psycopg2://...

Replaces the old hand-rolled sqlite3 module (data_store.py), which now re-exports
from here for backward compatibility.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from config import settings
from strategy import STRATEGY

logger = logging.getLogger(__name__)

_engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class DailyRun(Base):
    __tablename__ = "daily_runs"
    __table_args__ = (UniqueConstraint("run_date", "category", name="uq_run_date_category"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_date: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(20), nullable=False, default="all")
    total_stocks: Mapped[int] = mapped_column(Integer, default=0)
    qualified_count: Mapped[int] = mapped_column(Integer, default=0)
    min_score: Mapped[int] = mapped_column(Integer, default=2)
    ai_completed: Mapped[int] = mapped_column(Integer, default=0)
    ai_buy_count: Mapped[int] = mapped_column(Integer, default=0)
    ai_hold_count: Mapped[int] = mapped_column(Integer, default=0)
    ai_avoid_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running")

    stocks: Mapped[List["StockAnalysis"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    recommendations: Mapped[List["AiRecommendation"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class StockAnalysis(Base):
    __tablename__ = "stock_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("daily_runs.id", ondelete="CASCADE"), index=True)
    symbol: Mapped[str] = mapped_column(String(40))
    name: Mapped[str] = mapped_column(String(40), index=True)
    price: Mapped[float] = mapped_column(Float, default=0)
    change_val: Mapped[float] = mapped_column(Float, default=0)
    change_pct: Mapped[float] = mapped_column(Float, default=0)
    score: Mapped[int] = mapped_column(Integer, default=0)
    bear_score: Mapped[int] = mapped_column(Integer, default=0)
    signal: Mapped[str] = mapped_column(String(10), default="NEUTRAL")
    rsi: Mapped[float] = mapped_column(Float, default=0)
    macd: Mapped[float] = mapped_column(Float, default=0)
    macd_signal: Mapped[float] = mapped_column(Float, default=0)
    macd_hist: Mapped[float] = mapped_column(Float, default=0)
    sma20: Mapped[float] = mapped_column(Float, default=0)
    sma50: Mapped[float] = mapped_column(Float, default=0)
    sma200: Mapped[float] = mapped_column(Float, default=0)
    ema9: Mapped[float] = mapped_column(Float, default=0)
    ema21: Mapped[float] = mapped_column(Float, default=0)
    stoch_k: Mapped[float] = mapped_column(Float, default=0)
    stoch_d: Mapped[float] = mapped_column(Float, default=0)
    bb_upper: Mapped[float] = mapped_column(Float, default=0)
    bb_lower: Mapped[float] = mapped_column(Float, default=0)
    bb_width: Mapped[float] = mapped_column(Float, default=0)
    w52_high: Mapped[float] = mapped_column(Float, default=0)
    w52_low: Mapped[float] = mapped_column(Float, default=0)
    dist_52w: Mapped[float] = mapped_column(Float, default=0)
    volume: Mapped[int] = mapped_column(Integer, default=0)
    avg_volume: Mapped[int] = mapped_column(Integer, default=0)
    vol_ratio: Mapped[float] = mapped_column(Float, default=0)
    atr: Mapped[float] = mapped_column(Float, default=0)
    atr_pct: Mapped[float] = mapped_column(Float, default=0)
    adx: Mapped[float] = mapped_column(Float, default=0)
    plus_di: Mapped[float] = mapped_column(Float, default=0)
    minus_di: Mapped[float] = mapped_column(Float, default=0)
    cci: Mapped[float] = mapped_column(Float, default=0)
    williams_r: Mapped[float] = mapped_column(Float, default=0)
    mfi: Mapped[float] = mapped_column(Float, default=0)
    vwap: Mapped[float] = mapped_column(Float, default=0)
    roc: Mapped[float] = mapped_column(Float, default=0)
    cmf: Mapped[float] = mapped_column(Float, default=0)
    ichimoku_tenkan: Mapped[float] = mapped_column(Float, default=0)
    ichimoku_kijun: Mapped[float] = mapped_column(Float, default=0)
    pivot: Mapped[float] = mapped_column(Float, default=0)
    pivot_s1: Mapped[float] = mapped_column(Float, default=0)
    pivot_r1: Mapped[float] = mapped_column(Float, default=0)
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")

    run: Mapped[DailyRun] = relationship(back_populates="stocks")


class AiRecommendation(Base):
    __tablename__ = "ai_recommendations"
    __table_args__ = (UniqueConstraint("run_id", "stock_name", name="uq_run_stock"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("daily_runs.id", ondelete="CASCADE"), index=True)
    stock_name: Mapped[str] = mapped_column(String(40))
    action: Mapped[str] = mapped_column(String(10), default="HOLD")
    confidence: Mapped[str] = mapped_column(String(10), default="LOW")
    reason: Mapped[str] = mapped_column(Text, default="")
    target: Mapped[float] = mapped_column(Float, default=0)
    stoploss: Mapped[float] = mapped_column(Float, default=0)
    risk_reward: Mapped[str] = mapped_column(String(20), default="N/A")

    run: Mapped[DailyRun] = relationship(back_populates="recommendations")


class Watchlist(Base):
    __tablename__ = "watchlist"
    __table_args__ = (UniqueConstraint("symbol", name="uq_watchlist_symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(40))
    name: Mapped[str] = mapped_column(String(40))
    added_at: Mapped[str] = mapped_column(String(40), nullable=False)


class AlertRule(Base):
    __tablename__ = "alert_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(80), default="")
    metric: Mapped[str] = mapped_column(String(20), default="score")  # score|rsi|signal
    operator: Mapped[str] = mapped_column(String(4), default=">=")     # >=,<=,==,>,<
    threshold: Mapped[float] = mapped_column(Float, default=0)
    channel: Mapped[str] = mapped_column(String(20), default="email")  # email|telegram
    target: Mapped[str] = mapped_column(String(200), default="")
    enabled: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)


_STOCK_COLS = [c.name for c in StockAnalysis.__table__.columns if c.name not in ("id", "run_id")]


@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    Base.metadata.create_all(_engine)
    _ensure_schema_upgrades()


def _ensure_schema_upgrades() -> None:
    """Lightweight, idempotent column additions for pre-existing databases.

    Older databases (created before the bull/bear scoring change) lack the
    `bear_score` and `signal` columns. Add them in place so stored data keeps
    loading without a destructive migration.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(_engine)
    try:
        existing = {c["name"] for c in inspector.get_columns("stock_analysis")}
    except Exception:
        return
    additions = []
    if "bear_score" not in existing:
        additions.append("ALTER TABLE stock_analysis ADD COLUMN bear_score INTEGER DEFAULT 0")
    if "signal" not in existing:
        additions.append("ALTER TABLE stock_analysis ADD COLUMN signal VARCHAR(10) DEFAULT 'NEUTRAL'")
    if not additions:
        return
    with _engine.begin() as conn:
        for stmt in additions:
            try:
                conn.execute(text(stmt))
                logger.info("Schema upgrade applied: %s", stmt)
            except Exception as exc:
                logger.warning("Schema upgrade skipped (%s): %s", stmt, exc)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─── Write operations ──────────────────────────────────────────────

def create_run(run_date: str, category: str = "all", min_score: int = 2) -> int:
    with session_scope() as s:
        existing = s.execute(
            select(DailyRun).where(DailyRun.run_date == run_date, DailyRun.category == category)
        ).scalar_one_or_none()
        if existing is not None:
            s.delete(existing)
            s.flush()
        run = DailyRun(
            run_date=run_date, category=category, min_score=min_score,
            created_at=_now(), status="running",
        )
        s.add(run)
        s.flush()
        return run.id


def save_stock_analysis(run_id: int, stocks: List[Dict]) -> None:
    """Bulk-insert stock rows in a single transaction (Task 32)."""
    rows = []
    for st in stocks:
        row = {"run_id": run_id}
        for col in _STOCK_COLS:
            if col == "change_val":
                row[col] = st.get("change", 0)
            elif col == "reasons_json":
                row[col] = json.dumps(st.get("reasons", []))
            else:
                row[col] = st.get(col, 0 if col not in ("symbol", "name", "signal") else "")
        row.setdefault("signal", st.get("signal", "NEUTRAL"))
        rows.append(row)
    if not rows:
        return
    with session_scope() as s:
        s.execute(StockAnalysis.__table__.insert(), rows)


def save_ai_recommendations(run_id: int, recommendations: Dict) -> None:
    with session_scope() as s:
        for name, rec in recommendations.items():
            s.merge(
                AiRecommendation(
                    run_id=run_id, stock_name=name,
                    action=rec.get("action", "HOLD"),
                    confidence=rec.get("confidence", "LOW"),
                    reason=rec.get("reason", ""),
                    target=rec.get("target", 0) or 0,
                    stoploss=rec.get("stoploss", 0) or 0,
                    risk_reward=str(rec.get("risk_reward", "N/A")),
                )
            )


def update_run_status(run_id: int, status: str, total_stocks: int = 0,
                      qualified_count: int = 0, ai_completed: int = 0,
                      ai_buy: int = 0, ai_hold: int = 0, ai_avoid: int = 0) -> None:
    with session_scope() as s:
        run = s.get(DailyRun, run_id)
        if not run:
            return
        run.status = status
        run.total_stocks = total_stocks
        run.qualified_count = qualified_count
        run.ai_completed = ai_completed
        run.ai_buy_count = ai_buy
        run.ai_hold_count = ai_hold
        run.ai_avoid_count = ai_avoid


# ─── Read operations ───────────────────────────────────────────────

def get_available_dates() -> List[Dict]:
    with session_scope() as s:
        runs = s.execute(
            select(DailyRun).where(DailyRun.status == "completed").order_by(DailyRun.run_date.desc())
        ).scalars().all()
        return [
            {
                "id": r.id, "run_date": r.run_date, "category": r.category,
                "total_stocks": r.total_stocks, "qualified_count": r.qualified_count,
                "ai_completed": r.ai_completed, "ai_buy_count": r.ai_buy_count,
                "ai_hold_count": r.ai_hold_count, "ai_avoid_count": r.ai_avoid_count,
                "created_at": r.created_at, "status": r.status,
            }
            for r in runs
        ]


def get_run_by_date(run_date: str, category: str = "all") -> Optional[Dict]:
    with session_scope() as s:
        r = s.execute(
            select(DailyRun).where(
                DailyRun.run_date == run_date,
                DailyRun.category == category,
                DailyRun.status == "completed",
            )
        ).scalar_one_or_none()
        return {"id": r.id, "total_stocks": r.total_stocks} if r else None


def _enrich_stock_fields(d: Dict) -> Dict:
    """Derive net_score and signal from bull/bear scores (always recompute)."""
    bull = int(d.get("score") or 0)
    bear = int(d.get("bear_score") or 0)
    net = bull - bear
    d["net_score"] = net
    th = STRATEGY.thresholds
    if net >= th.buy:
        d["signal"] = "BUY"
    elif net <= th.sell:
        d["signal"] = "SELL"
    else:
        d["signal"] = "NEUTRAL"
    return d


def _stock_to_dict(row: StockAnalysis) -> Dict:
    d = {c: getattr(row, c) for c in _STOCK_COLS}
    d["change"] = d.pop("change_val", 0)
    d["reasons"] = json.loads(d.pop("reasons_json", "[]") or "[]")
    return _enrich_stock_fields(d)


def get_stored_analysis(run_id: int, min_score: int = 0) -> Dict:
    with session_scope() as s:
        run = s.get(DailyRun, run_id)
        if not run:
            return {"error": "Run not found"}
        stock_rows = s.execute(
            select(StockAnalysis).where(StockAnalysis.run_id == run_id).order_by(StockAnalysis.score.desc())
        ).scalars().all()
        stocks = [_stock_to_dict(r) for r in stock_rows]
        ai_rows = s.execute(
            select(AiRecommendation).where(AiRecommendation.run_id == run_id)
        ).scalars().all()
        ai_recs = {
            r.stock_name: {
                "action": r.action, "confidence": r.confidence, "reason": r.reason,
                "target": r.target, "stoploss": r.stoploss, "risk_reward": r.risk_reward,
            }
            for r in ai_rows
        }
        effective_min = min_score if min_score > 0 else run.min_score
        qualified = [st for st in stocks if st["score"] >= effective_min]
        unqualified = [st for st in stocks if st["score"] < effective_min]
        return {
            "date": run.run_date,
            "category": run.category,
            "category_label": _category_label(run.category, len(stocks)),
            "total_stocks": len(stocks),
            "qualified_count": len(qualified),
            "min_score": effective_min,
            "qualified": qualified,
            "unqualified": unqualified,
            "skipped": [],
            "timestamp": run.created_at,
            "stored": True,
            "ai_recommendations": ai_recs or None,
            "ai_stats": {
                "buy": run.ai_buy_count, "hold": run.ai_hold_count, "avoid": run.ai_avoid_count,
            } if run.ai_completed else None,
        }


def _category_label(category: str, count: int) -> str:
    labels = {
        "nifty50": "NIFTY 50", "nifty100": "NIFTY 100", "nifty200": "NIFTY 200",
        "nifty500": "NIFTY 500", "all": f"All NSE ({count})",
    }
    return labels.get(category, f"{category} ({count})")


# ─── Watchlist / alerts (Task 46) ──────────────────────────────────

def list_watchlist() -> List[Dict]:
    with session_scope() as s:
        rows = s.execute(select(Watchlist).order_by(Watchlist.added_at.desc())).scalars().all()
        return [{"symbol": r.symbol, "name": r.name, "added_at": r.added_at} for r in rows]


def add_to_watchlist(symbol: str, name: str) -> None:
    with session_scope() as s:
        s.merge(Watchlist(symbol=symbol, name=name, added_at=_now()))


def remove_from_watchlist(symbol: str) -> None:
    with session_scope() as s:
        row = s.execute(select(Watchlist).where(Watchlist.symbol == symbol)).scalar_one_or_none()
        if row:
            s.delete(row)


def list_alert_rules(enabled_only: bool = False) -> List[Dict]:
    with session_scope() as s:
        stmt = select(AlertRule)
        if enabled_only:
            stmt = stmt.where(AlertRule.enabled == 1)
        rows = s.execute(stmt).scalars().all()
        return [
            {
                "id": r.id, "name": r.name, "metric": r.metric, "operator": r.operator,
                "threshold": r.threshold, "channel": r.channel, "target": r.target,
                "enabled": r.enabled,
            }
            for r in rows
        ]


def add_alert_rule(name: str, metric: str, operator: str, threshold: float,
                   channel: str, target: str) -> int:
    with session_scope() as s:
        rule = AlertRule(
            name=name, metric=metric, operator=operator, threshold=threshold,
            channel=channel, target=target, created_at=_now(),
        )
        s.add(rule)
        s.flush()
        return rule.id


def delete_alert_rule(rule_id: int) -> None:
    with session_scope() as s:
        row = s.get(AlertRule, rule_id)
        if row:
            s.delete(row)


# Initialize schema on import (safe/idempotent).
init_db()

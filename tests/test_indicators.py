"""
Unit tests for the indicator engine (Task 27).

Strategy:
* Hand-computed reference values for the simple indicators (RSI, ATR,
  Stochastic, Bollinger, VWAP) on tiny fixed fixtures — the arithmetic is
  spelled out in the comments so a regression is obvious.
* An independent, transparent loop-based reference implementation for the
  Wilder-smoothed indicators (RSI/ATR/ADX) and MACD, used to prove that the
  vectorized production code (Task 26) matches the explicit recursion.
* Behavioural tests for the scoring engine: a clearly bullish fixture scores
  net-positive (BUY) and a clearly bearish fixture scores net-negative (SELL),
  contradictory reasons are mutually exclusive, the public dict contract is
  preserved, and failures return ``None`` while being logged.

Run with: ``python3 -m pytest tests/test_indicators.py -v``
"""

import logging
import math

import numpy as np
import pandas as pd
import pytest

import indicators as ind
from strategy import STRATEGY, Strategy, IndicatorPeriods
from tests.conftest import make_ohlcv


TOL = 1e-6  # tolerance for vectorized-vs-reference float comparisons


# ─── Independent reference implementations (plain Python loops) ──────────────

def ref_rma(values, period):
    """Wilder's RMA seeded by the SMA of the first ``period`` valid values.

    Mirrors the production algorithm (skips leading NaNs) but uses an explicit
    loop so it is an *independent* check of the vectorized version.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    out = [float("nan")] * n
    p0 = next((i for i, x in enumerate(vals) if not math.isnan(x)), None)
    if p0 is None or n - p0 < period:
        return out
    seed = sum(vals[p0:p0 + period]) / period
    out[p0 + period - 1] = seed
    for i in range(p0 + period, n):
        out[i] = (out[i - 1] * (period - 1) + vals[i]) / period
    return out


def ref_ema(values, period):
    k = 2.0 / (period + 1)
    out = [float(values[0])]
    for i in range(1, len(values)):
        out.append(values[i] * k + out[-1] * (1 - k))
    return out


def ref_rsi(close, period):
    delta = np.diff(np.asarray(close, dtype=float))
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = ref_rma(gain, period)
    al = ref_rma(loss, period)
    last_g, last_l = ag[-1], al[-1]
    if math.isnan(last_g):
        return float("nan")
    if last_l == 0:
        return 100.0
    rs = last_g / last_l
    return 100.0 - 100.0 / (1.0 + rs)


def ref_adx(high, low, close, period):
    h = [float(x) for x in high]
    l = [float(x) for x in low]
    c = [float(x) for x in close]
    n = len(c)
    tr = [h[0] - l[0]]
    plus_dm = [0.0]
    minus_dm = [0.0]
    for i in range(1, n):
        tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
    atr = ref_rma(tr, period)
    pdm = ref_rma(plus_dm, period)
    mdm = ref_rma(minus_dm, period)
    pdi, mdi, dx = [], [], []
    for i in range(n):
        if math.isnan(atr[i]) or atr[i] == 0:
            pdi.append(float("nan"))
            mdi.append(float("nan"))
            dx.append(float("nan"))
            continue
        p = 100.0 * pdm[i] / atr[i]
        m = 100.0 * mdm[i] / atr[i]
        pdi.append(p)
        mdi.append(m)
        dx.append(100.0 * abs(p - m) / (p + m) if (p + m) > 0 else float("nan"))
    adx = ref_rma(dx, period)
    return adx[-1], pdi[-1], mdi[-1]


# ─── Hand-computed indicator checks ──────────────────────────────────────────

def test_rsi_handcomputed():
    # close diffs: +1,+1,-1,+1,+1,-1,+1 ; period 3 Wilder.
    # avg_gain_last = 173/243, avg_loss_last = 70/243, rs = 173/70.
    # RSI = 100 - 100/(1 + 173/70) = 71.1935...
    close = [10, 11, 12, 11, 12, 13, 12, 13]
    rsi = ind._last(ind.rsi_series(close, 3))
    assert rsi == pytest.approx(71.1935, abs=1e-3)


def test_atr_wilder_handcomputed():
    # TR = [1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], period 3 Wilder.
    # ATR_last computed via the Wilder recursion = 1.478052...
    high = [10.5, 11.5, 12.5, 12.0, 12.5, 13.5, 13.0, 13.5]
    low = [9.5, 10.5, 11.5, 10.5, 11.0, 12.0, 11.5, 12.0]
    close = [10, 11, 12, 11, 12, 13, 12, 13]
    atr = ind._last(ind.atr_series(high, low, close, 3))
    assert atr == pytest.approx(1.478052, abs=1e-4)


def test_stochastic_handcomputed():
    # %K (k=3): [.., 75, 33.333, 75, 80] ; %D = 3-SMA of %K -> last = 62.7778.
    high = [11, 12, 13, 12, 14, 15]
    low = [9, 10, 11, 10, 11, 12]
    close = [10, 11, 12, 11, 13, 14]
    k, d = ind.stochastic(high, low, close, 3, 1, 3)
    assert ind._last(k) == pytest.approx(80.0, abs=1e-6)
    assert ind._last(d) == pytest.approx(62.7778, abs=1e-3)


def test_bollinger_handcomputed():
    # window [10,11,12,13,14], period 5, 2 std (population): mid=12,
    # std=sqrt(2)=1.41421, upper=14.82843, lower=9.17157.
    close = [10, 11, 12, 13, 14]
    mid, upper, lower = ind.bollinger(close, 5, 2.0)
    assert ind._last(mid) == pytest.approx(12.0, abs=1e-9)
    assert ind._last(upper) == pytest.approx(14.828427, abs=1e-5)
    assert ind._last(lower) == pytest.approx(9.171573, abs=1e-5)


def test_vwap_session_anchored_resets_daily():
    # Two sessions, two bars each, high==low==close so typical price == close.
    # Day1: C=10 (v=100), C=20 (v=300) -> vwap@bar2 = 7000/400 = 17.5
    # Day2 must RESET: vwap@bar3 = 30 (not blended with day 1),
    #                  vwap@bar4 = (30*100 + 40*100)/200 = 35.
    dates = pd.to_datetime([
        "2024-01-01 09:15", "2024-01-01 09:16",
        "2024-01-02 09:15", "2024-01-02 09:16",
    ])
    close = np.array([10.0, 20.0, 30.0, 40.0])
    vol = np.array([100.0, 300.0, 100.0, 100.0])
    vwap = ind.vwap_session(close, close, close, vol, dates)
    assert vwap[1] == pytest.approx(17.5, abs=1e-9)
    assert vwap[2] == pytest.approx(30.0, abs=1e-9)  # reset at new session
    assert vwap[3] == pytest.approx(35.0, abs=1e-9)
    # A naive cumulative VWAP would give bar3 = (1000+6000+3000)/500 = 20, so
    # the reset behaviour is what distinguishes session-anchored VWAP.
    assert vwap[2] != pytest.approx(20.0, abs=1e-6)


# ─── Vectorized vs independent reference (Task 19 + Task 26) ─────────────────

@pytest.fixture
def noisy_close():
    rng = np.random.default_rng(42)
    steps = rng.normal(0.0, 1.0, 120).cumsum()
    return 100.0 + steps


def test_wilder_rma_matches_reference(noisy_close):
    got = ind.wilder_rma(noisy_close, 14)
    exp = ref_rma(noisy_close, 14)
    assert ind._last(got) == pytest.approx(exp[-1], abs=TOL)


def test_rsi_matches_reference(noisy_close):
    got = ind._last(ind.rsi_series(noisy_close, 14))
    assert got == pytest.approx(ref_rsi(noisy_close, 14), abs=TOL)


def test_macd_matches_reference(noisy_close):
    macd_line, signal_line, hist = ind.macd(noisy_close, 12, 26, 9)
    ref_macd = np.array(ref_ema(noisy_close, 12)) - np.array(ref_ema(noisy_close, 26))
    ref_signal = ref_ema(ref_macd, 9)
    assert macd_line[-1] == pytest.approx(ref_macd[-1], abs=TOL)
    assert signal_line[-1] == pytest.approx(ref_signal[-1], abs=TOL)
    assert hist[-1] == pytest.approx(ref_macd[-1] - ref_signal[-1], abs=TOL)


def test_adx_matches_reference():
    rng = np.random.default_rng(7)
    close = 100.0 + rng.normal(0.0, 1.0, 120).cumsum()
    high = close + np.abs(rng.normal(0.5, 0.2, 120))
    low = close - np.abs(rng.normal(0.5, 0.2, 120))
    adx_s, pdi_s, mdi_s = ind.adx_components(high, low, close, 14)
    ref_adx_v, ref_pdi, ref_mdi = ref_adx(high, low, close, 14)
    assert ind._last(adx_s) == pytest.approx(ref_adx_v, abs=1e-5)
    assert ind._last(pdi_s) == pytest.approx(ref_pdi, abs=1e-5)
    assert ind._last(mdi_s) == pytest.approx(ref_mdi, abs=1e-5)


def test_adx_directionality(bullish_frame, bearish_frame):
    bh, bl, bc = bullish_frame["High"], bullish_frame["Low"], bullish_frame["Close"]
    adx_s, pdi_s, mdi_s = ind.adx_components(bh, bl, bc, 14)
    assert ind._last(pdi_s) > ind._last(mdi_s)        # +DI dominates in uptrend
    assert ind._last(adx_s) > 20                       # strong trend

    eh, el, ec = bearish_frame["High"], bearish_frame["Low"], bearish_frame["Close"]
    adx_s2, pdi_s2, mdi_s2 = ind.adx_components(eh, el, ec, 14)
    assert ind._last(mdi_s2) > ind._last(pdi_s2)      # -DI dominates in downtrend


def test_optional_pandas_ta_reference(noisy_close):
    """Cross-check RSI against pandas-ta when it is installed (skipped on numpy 2)."""
    pta = pytest.importorskip("pandas_ta")
    ref = pta.rsi(pd.Series(noisy_close), length=14)
    got = ind._last(ind.rsi_series(noisy_close, 14))
    assert got == pytest.approx(float(ref.iloc[-1]), abs=0.5)


# ─── Scoring engine (Tasks 20, 21) ───────────────────────────────────────────

PUBLIC_KEYS = {
    "symbol", "name", "price", "change", "change_pct", "score", "rsi", "macd",
    "macd_signal", "macd_hist", "sma20", "sma50", "sma200", "ema9", "ema21",
    "stoch_k", "stoch_d", "bb_upper", "bb_lower", "bb_width", "w52_high",
    "w52_low", "dist_52w", "volume", "avg_volume", "vol_ratio", "atr",
    "atr_pct", "adx", "plus_di", "minus_di", "cci", "williams_r", "mfi",
    "vwap", "roc", "cmf", "ichimoku_tenkan", "ichimoku_kijun", "pivot",
    "pivot_s1", "pivot_r1", "reasons",
}


def test_public_contract_preserved(bullish_frame):
    res = ind.analyze_stock("NSE:TEST-EQ", bullish_frame)
    assert res is not None
    # Every original key must still be present (new keys may be added).
    assert PUBLIC_KEYS.issubset(set(res.keys()))
    assert res["name"] == "TEST"
    # reasons is a list of {text, type, icon} dicts.
    assert isinstance(res["reasons"], list)
    for r in res["reasons"]:
        assert set(r.keys()) == {"text", "type", "icon"}
        assert r["type"] in {"bullish", "bearish", "info"}


def test_new_fields_present(bullish_frame):
    res = ind.analyze_stock("NSE:TEST-EQ", bullish_frame)
    for key in ("bear_score", "net_score", "signal"):
        assert key in res
    assert res["signal"] in {"BUY", "NEUTRAL", "SELL"}
    assert res["net_score"] == res["score"] - res["bear_score"]


def test_bullish_fixture_scores_high(bullish_frame):
    res = ind.analyze_stock("NSE:BULL-EQ", bullish_frame)
    assert res["net_score"] > 0
    assert res["score"] > res["bear_score"]
    assert res["signal"] == "BUY"


def test_bearish_fixture_scores_low(bearish_frame):
    res = ind.analyze_stock("NSE:BEAR-EQ", bearish_frame)
    assert res["net_score"] < 0
    assert res["bear_score"] > res["score"]
    assert res["signal"] == "SELL"


def test_bullish_beats_bearish(bullish_frame, bearish_frame):
    bull = ind.analyze_stock("NSE:BULL-EQ", bullish_frame)
    bear = ind.analyze_stock("NSE:BEAR-EQ", bearish_frame)
    assert bull["net_score"] > bear["net_score"]


# Mutually-exclusive contradictory reason pairs (Task 21).
CONTRADICTORY_PAIRS = [
    ("Near 52W High", "Near 52W Low"),
    ("MACD Bullish Crossover", "MACD Bearish"),
    ("Golden Cross", "Death Cross"),
    ("EMA 9/21 Bullish Crossover", "EMA 9/21 Bearish Crossover"),
    ("Above Ichimoku Cloud", "Below Ichimoku Cloud"),
    ("Ichimoku TK Cross Bullish", "Ichimoku TK Cross Bearish"),
    ("Below Bollinger Lower Band", "Above Bollinger Upper Band"),
    ("Above R1 Pivot", "Below S1 Support"),
    ("Green Candles in a row", "Red Candles in a row"),
]


def _random_frames(count=25, length=120):
    frames = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        closes = 100.0 + rng.normal(0.0, 1.5, length).cumsum()
        closes = np.clip(closes, 5.0, None)  # keep prices positive
        vols = rng.uniform(500, 4000, length)
        frames.append(make_ohlcv(closes, volume=vols))
    return frames


@pytest.mark.parametrize("frame", _random_frames())
def test_no_contradictory_reasons(frame):
    res = ind.analyze_stock("NSE:RND-EQ", frame)
    assert res is not None
    texts = [r["text"] for r in res["reasons"]]
    for a, b in CONTRADICTORY_PAIRS:
        has_a = any(a in t for t in texts)
        has_b = any(b in t for t in texts)
        assert not (has_a and has_b), f"Contradictory reasons fired together: {a!r} & {b!r}"


# ─── Configurable strategy (Task 25) ─────────────────────────────────────────

def test_strategy_validation_rejects_bad_periods():
    bad = Strategy(periods=IndicatorPeriods(rsi=0))
    with pytest.raises(ValueError):
        bad.validate()


def test_strategy_weight_change_affects_score(bullish_frame):
    base = ind.analyze_stock("NSE:TEST-EQ", bullish_frame, STRATEGY)
    # Double every bullish weight via an override and confirm the score grows.
    boosted_bull = {k: v * 2 for k, v in STRATEGY.weights.bull.items()}
    from strategy import ScoreWeights
    boosted = STRATEGY.with_overrides(
        weights=ScoreWeights(bull=boosted_bull, bear=STRATEGY.weights.bear)
    )
    tuned = ind.analyze_stock("NSE:TEST-EQ", bullish_frame, boosted)
    assert tuned["score"] > base["score"]


# ─── Exception handling / logging (Task 24) ──────────────────────────────────

def test_analyze_stock_logs_and_returns_none_on_bad_input(caplog):
    # One-row frame triggers the "insufficient data" guard.
    frame = make_ohlcv([100.0])
    with caplog.at_level(logging.ERROR, logger="indicators"):
        result = ind.analyze_stock("NSE:BAD-EQ", frame)
    assert result is None
    assert any("analyze_stock failed for NSE:BAD-EQ" in rec.message for rec in caplog.records)


def test_analyze_stock_missing_column_returns_none(caplog):
    frame = pd.DataFrame({"Close": [1, 2, 3]})  # missing High/Low/Volume
    with caplog.at_level(logging.ERROR, logger="indicators"):
        result = ind.analyze_stock("NSE:MISS-EQ", frame)
    assert result is None
    assert any("NSE:MISS-EQ" in rec.message for rec in caplog.records)

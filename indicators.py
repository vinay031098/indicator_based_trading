"""
Stock Analyzer — Technical Indicators + Bullish/Bearish Scoring Engine.

Extracted from app.py so it can be imported by both Flask and daily_runner.py
without initializing the Flask app, Fyers client, or browser opens.

Design notes (Phase 2 — Trading Logic & Correctness):

* RSI / ATR / ADX use **Wilder's smoothing** (RMA), implemented as an EWM with
  ``alpha = 1/period`` seeded by the simple average of the first ``period``
  values — the standard convention used by TA-Lib / pandas-ta.
* The indicator math is **vectorized** with NumPy / pandas; the only remaining
  Python loops are the short candle-streak counter (bounded, cheap).
* The Stochastic oscillator computes a clean ``%K`` series and derives
  ``%D`` as a simple 3-period moving average of ``%K``.
* VWAP is **session-anchored**: cumulative ``typical_price * volume`` resets at
  the start of each trading day (grouped on the DataFrame's date index) instead
  of accumulating over the entire history. For daily bars this means each bar's
  VWAP equals that bar's typical price; for intraday bars it resets every day.
* Scoring is **symmetric**: a bullish ``score`` (kept for backward compat), a
  ``bear_score``, a ``net_score`` and a categorical ``signal`` of
  BUY / NEUTRAL / SELL. Contradictory reasons (e.g. "Near 52W High" vs
  "Near 52W Low") are mutually exclusive — only the dominant one fires.
* All periods / weights / thresholds live in ``strategy.py`` (no magic numbers).
* ``analyze_stock`` logs failures via ``logging`` (with stack trace) instead of
  silently swallowing every exception.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from strategy import STRATEGY, Strategy

logger = logging.getLogger(__name__)


# ─── Low-level helpers ───────────────────────────────────────────────────────

def _as_array(data) -> np.ndarray:
    return np.asarray(data, dtype=float).flatten()


def _last(series, default: float = 0.0) -> float:
    """Last non-NaN value of an array/series, or ``default`` if none."""
    arr = np.asarray(series, dtype=float).flatten()
    arr = arr[~np.isnan(arr)]
    return float(arr[-1]) if arr.size else float(default)


def wilder_rma(values, period: int) -> np.ndarray:
    """Wilder's smoothed moving average (a.k.a. RMA / SMMA).

    Equivalent to an EWM with ``alpha = 1/period`` seeded by the simple mean of
    the first ``period`` valid observations. Leading entries (and anything
    before the seed) are NaN, matching TA-Lib's warm-up behaviour. Implemented
    with pandas' C-level ``ewm`` so it stays vectorized.
    """
    arr = _as_array(values)
    n = arr.size
    out = np.full(n, np.nan)
    valid = ~np.isnan(arr)
    if not valid.any():
        return out
    p0 = int(np.argmax(valid))  # index of first valid observation
    if n - p0 < period:
        # Not enough data to seed; degrade to an expanding mean of the tail.
        out[p0:] = pd.Series(arr[p0:]).expanding().mean().to_numpy()
        return out
    seed = float(np.nanmean(arr[p0:p0 + period]))
    sub = arr[p0:].copy()
    sub[:period - 1] = np.nan
    sub[period - 1] = seed
    out[p0:] = pd.Series(sub).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    return out


def ema_series(values, period: int) -> np.ndarray:
    """Standard exponential moving average (``adjust=False``, seeded with x0)."""
    arr = _as_array(values)
    if arr.size == 0:
        return arr
    return pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()


def sma_series(values, period: int) -> np.ndarray:
    arr = _as_array(values)
    return pd.Series(arr).rolling(period).mean().to_numpy()


# ─── Indicators ──────────────────────────────────────────────────────────────

def rsi_series(close, period: int = 14) -> np.ndarray:
    """Wilder's RSI as a full-length series."""
    arr = _as_array(close)
    if arr.size < 2:
        return np.full(arr.size, np.nan)
    delta = np.diff(arr)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (all gains) RSI is 100; when avg_gain == 0 it is 0.
    rsi = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, rsi)
    rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi)
    rsi = np.where((avg_gain == 0) & (avg_loss > 0), 0.0, rsi)
    # diff drops one element; pad front so length matches input.
    return np.concatenate([[np.nan], rsi])


def true_range(high, low, close) -> np.ndarray:
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    return tr


def atr_series(high, low, close, period: int = 14) -> np.ndarray:
    return wilder_rma(true_range(high, low, close), period)


def adx_components(high, low, close, period: int = 14):
    """Return (adx_series, plus_di_series, minus_di_series) using Wilder smoothing."""
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    n = c.size
    up = np.zeros(n)
    down = np.zeros(n)
    up[1:] = h[1:] - h[:-1]
    down[1:] = l[:-1] - l[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    atr = wilder_rma(true_range(h, l, c), period)
    plus_dm_s = wilder_rma(plus_dm, period)
    minus_dm_s = wilder_rma(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(atr > 0, 100.0 * plus_dm_s / atr, np.nan)
        minus_di = np.where(atr > 0, 100.0 * minus_dm_s / atr, np.nan)
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, np.nan)
    adx = wilder_rma(dx, period)
    return adx, plus_di, minus_di


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram) series."""
    arr = _as_array(close)
    macd_line = ema_series(arr, fast) - ema_series(arr, slow)
    signal_line = ema_series(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(high, low, close, k_period: int = 14, smooth_k: int = 1, d_period: int = 3):
    """Clean Stochastic oscillator.

    %K = 100 * (close - lowest_low) / (highest_high - lowest_low) over
    ``k_period`` bars (optionally smoothed by ``smooth_k``); %D = ``d_period``
    SMA of %K. Returns (%K series, %D series).
    """
    h = pd.Series(_as_array(high))
    l = pd.Series(_as_array(low))
    c = pd.Series(_as_array(close))
    lowest = l.rolling(k_period).min()
    highest = h.rolling(k_period).max()
    rng = highest - lowest
    raw_k = 100.0 * (c - lowest) / rng.where(rng != 0, np.nan)
    raw_k = raw_k.where(rng != 0, 50.0)  # flat range -> neutral 50
    k = raw_k.rolling(smooth_k).mean() if smooth_k > 1 else raw_k
    d = k.rolling(d_period).mean()
    return k.to_numpy(), d.to_numpy()


def bollinger(close, period: int = 20, num_std: float = 2.0):
    """Return (mid, upper, lower) bands. Uses population std (ddof=0)."""
    s = pd.Series(_as_array(close))
    mid = s.rolling(period).mean()
    std = s.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid.to_numpy(), upper.to_numpy(), lower.to_numpy()


def obv(close, volume) -> np.ndarray:
    """On-Balance Volume (vectorized)."""
    c = _as_array(close)
    v = _as_array(volume)
    if c.size == 0:
        return np.zeros(0)
    direction = np.sign(np.diff(c))
    out = np.zeros(c.size)
    out[1:] = np.cumsum(direction * v[1:])
    return out


def vwap_session(high, low, close, volume, dates=None) -> np.ndarray:
    """Session-anchored VWAP series.

    Cumulative ``typical_price * volume`` and cumulative volume reset at the
    start of each trading session. Sessions are derived from ``dates`` (a
    DatetimeIndex / array). If ``dates`` is None or non-datetime, the whole
    series is treated as one session.
    """
    tp = (_as_array(high) + _as_array(low) + _as_array(close)) / 3.0
    v = _as_array(volume)
    tpv = tp * v

    if dates is None:
        keys = np.zeros(tp.size, dtype=np.int64)
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="coerce"))
        if idx.isna().all():
            keys = np.zeros(tp.size, dtype=np.int64)
        else:
            keys = idx.normalize().asi8  # one key per calendar day

    df = pd.DataFrame({"tpv": tpv, "v": v, "key": keys})
    cum_tpv = df.groupby("key")["tpv"].cumsum()
    cum_v = df.groupby("key")["v"].cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cum_v.to_numpy() > 0, cum_tpv.to_numpy() / cum_v.to_numpy(), tp)
    return vwap


def cci(high, low, close, period: int = 20) -> float:
    tp = (_as_array(high) + _as_array(low) + _as_array(close)) / 3.0
    window = tp[-period:]
    sma = np.mean(window)
    md = np.mean(np.abs(window - sma))
    return float((tp[-1] - sma) / (0.015 * md)) if md != 0 else 0.0


def williams_r(high, low, close, period: int = 14) -> float:
    h, l, c = _as_array(high), _as_array(low), _as_array(close)
    hh = np.max(h[-period:])
    ll = np.min(l[-period:])
    return float(-100.0 * (hh - c[-1]) / (hh - ll)) if (hh - ll) != 0 else -50.0


def mfi(high, low, close, volume, period: int = 14) -> float:
    tp = (_as_array(high) + _as_array(low) + _as_array(close)) / 3.0
    mf = tp * _as_array(volume)
    if tp.size <= period:
        return 50.0
    delta = np.diff(tp)
    pos = np.where(delta > 0, mf[1:], 0.0)
    neg = np.where(delta < 0, mf[1:], 0.0)
    pos_mf = float(np.sum(pos[-period:]))
    neg_mf = float(np.sum(neg[-period:]))
    if neg_mf == 0:
        return 100.0
    mfr = pos_mf / neg_mf
    return float(100.0 - (100.0 / (1.0 + mfr)))


def roc(close, period: int = 12) -> float:
    arr = _as_array(close)
    if arr.size <= period or arr[-1 - period] == 0:
        return 0.0
    return float((arr[-1] - arr[-1 - period]) / arr[-1 - period] * 100.0)


def cmf(high, low, close, volume, period: int = 20) -> float:
    h, l, c, v = _as_array(high), _as_array(low), _as_array(close), _as_array(volume)
    rng = h - l
    mfm = np.where(rng != 0, ((c - l) - (h - c)) / rng, 0.0)
    mfv = mfm * v
    vol_sum = float(np.sum(v[-period:]))
    return float(np.sum(mfv[-period:]) / vol_sum) if vol_sum > 0 else 0.0


def ichimoku(high, low, close, tenkan_p=9, kijun_p=26, senkou_b_p=52):
    h, l = _as_array(high), _as_array(low)
    tenkan = (np.max(h[-tenkan_p:]) + np.min(l[-tenkan_p:])) / 2.0
    kijun = (np.max(h[-kijun_p:]) + np.min(l[-kijun_p:])) / 2.0
    senkou_a = (tenkan + kijun) / 2.0
    if h.size >= senkou_b_p:
        senkou_b = (np.max(h[-senkou_b_p:]) + np.min(l[-senkou_b_p:])) / 2.0
    else:
        senkou_b = (np.max(h) + np.min(l)) / 2.0
    return float(tenkan), float(kijun), float(senkou_a), float(senkou_b)


def pivot_points(high, low, close):
    h = float(_as_array(high)[-1])
    l = float(_as_array(low)[-1])
    c = float(_as_array(close)[-1])
    pp = (h + l + c) / 3.0
    s1 = 2 * pp - h
    r1 = 2 * pp - l
    s2 = pp - (h - l)
    r2 = pp + (h - l)
    return pp, s1, r1, s2, r2


# ─── Main analysis + scoring ─────────────────────────────────────────────────

def analyze_stock(symbol, data, strategy: Strategy = STRATEGY):
    """Analyze a single stock and return a dict of indicators + scoring.

    On any failure the error is logged (with stack trace) and ``None`` is
    returned, so a single bad symbol never aborts a full run while still being
    visible in the logs.
    """
    try:
        p = strategy.periods
        w_bull = strategy.weights.bull
        w_bear = strategy.weights.bear
        lv = strategy.levels

        close = _as_array(data["Close"])
        high = _as_array(data["High"])
        low = _as_array(data["Low"])
        volume = _as_array(data["Volume"])
        dates = data.index

        if close.size < 2:
            raise ValueError(f"insufficient data: {close.size} rows")

        price = float(close[-1])
        prev_close = float(close[-2])
        change = price - prev_close
        change_pct = (change / prev_close * 100.0) if prev_close != 0 else 0.0

        # ── Indicators ───────────────────────────────────────────────
        rsi = _last(rsi_series(close, p.rsi), 50.0)

        sma20 = _last(sma_series(close, p.sma_short), float(np.mean(close)))
        sma50 = float(np.mean(close[-p.sma_mid:])) if close.size >= p.sma_mid else float(np.mean(close))
        sma200 = float(np.mean(close[-p.sma_long:])) if close.size >= p.sma_long else float(np.mean(close))

        ema_fast_s = ema_series(close, p.ema_fast)
        ema_slow_s = ema_series(close, p.ema_slow)
        ema9 = float(ema_fast_s[-1])
        ema21 = float(ema_slow_s[-1])
        prev_ema9 = float(ema_fast_s[-2])
        prev_ema21 = float(ema_slow_s[-2])
        ema_cross_bull = (ema9 > ema21) and (prev_ema9 <= prev_ema21)
        ema_cross_bear = (ema9 < ema21) and (prev_ema9 >= prev_ema21)

        macd_line, signal_line, hist_line = macd(close, p.macd_fast, p.macd_slow, p.macd_signal)
        macd_val = float(macd_line[-1])
        macd_signal_val = float(signal_line[-1])
        macd_hist = macd_val - macd_signal_val
        prev_macd_hist = float(hist_line[-2]) if hist_line.size > 1 else 0.0
        macd_hist_rising = macd_hist > prev_macd_hist

        k_series, d_series = stochastic(high, low, close, p.stoch_k, p.stoch_smooth_k, p.stoch_d)
        stoch_k = _last(k_series, 50.0)
        stoch_d = _last(d_series, 50.0)

        bb_mid_s, bb_up_s, bb_low_s = bollinger(close, p.bb_period, p.bb_std)
        bb_sma = _last(bb_mid_s, float(np.mean(close[-p.bb_period:])))
        bb_upper = _last(bb_up_s, bb_sma)
        bb_lower = _last(bb_low_s, bb_sma)
        bb_width = (bb_upper - bb_lower) / bb_sma * 100.0 if bb_sma > 0 else 0.0

        w52_high = float(np.max(high[-p.w52:])) if high.size >= p.w52 else float(np.max(high))
        w52_low = float(np.min(low[-p.w52:])) if low.size >= p.w52 else float(np.min(low))
        dist_52w_high = ((w52_high - price) / w52_high) * 100.0 if w52_high > 0 else 0.0
        dist_52w_low = ((price - w52_low) / w52_low) * 100.0 if w52_low > 0 else 0.0

        avg_vol = float(np.mean(volume[-p.volume_avg:])) if volume.size >= p.volume_avg else float(np.mean(volume))
        curr_vol = float(volume[-1])
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

        atr_val = _last(atr_series(high, low, close, p.atr), 0.0)
        atr_pct = (atr_val / price * 100.0) if price > 0 else 0.0

        adx_s, plus_di_s, minus_di_s = adx_components(high, low, close, p.adx)
        adx_val = _last(adx_s, 0.0)
        plus_di = _last(plus_di_s, 0.0)
        minus_di = _last(minus_di_s, 0.0)

        cci_val = cci(high, low, close, p.cci)
        wr_val = williams_r(high, low, close, p.williams)
        mfi_val = mfi(high, low, close, volume, p.mfi)

        obv_arr = obv(close, volume)
        obv_slope = float(obv_arr[-1] - obv_arr[-1 - p.obv_slope]) if obv_arr.size > p.obv_slope else 0.0

        vwap_val = _last(vwap_session(high, low, close, volume, dates), price)
        roc_val = roc(close, p.roc)
        cmf_val = cmf(high, low, close, volume, p.cmf)

        tenkan, kijun, senkou_a, senkou_b = ichimoku(
            high, low, close, p.ichimoku_tenkan, p.ichimoku_kijun, p.ichimoku_senkou_b
        )
        pp, s1, r1, s2, r2 = pivot_points(high, low, close)

        # Consecutive green / red candles (short, bounded loop).
        green_count = 0
        red_count = 0
        for i in range(-1, -min(10, close.size), -1):
            if close[i] > close[i - 1]:
                if red_count:
                    break
                green_count += 1
            elif close[i] < close[i - 1]:
                if green_count:
                    break
                red_count += 1
            else:
                break

        dist_sma200 = ((price - sma200) / sma200 * 100.0) if sma200 > 0 else 0.0

        # ── Scoring (symmetric bull / bear) ──────────────────────────
        bull = 0
        bear = 0
        reasons = []

        def add_bull(key, text, icon):
            nonlocal bull
            bull += w_bull.get(key, 0)
            reasons.append({"text": text, "type": "bullish", "icon": icon})

        def add_bear(key, text, icon):
            nonlocal bear
            bear += w_bear.get(key, 0)
            reasons.append({"text": text, "type": "bearish", "icon": icon})

        def add_info(text, icon):
            reasons.append({"text": text, "type": "info", "icon": icon})

        # 1. RSI (oversold/overbought are mutually exclusive by threshold)
        if rsi < lv.rsi_oversold:
            add_bull("rsi_oversold", f"RSI Oversold ({rsi:.1f})", "📉")
        elif rsi < lv.rsi_low:
            add_bull("rsi_low", f"RSI Low ({rsi:.1f})", "📊")
        elif rsi > lv.rsi_overbought:
            add_bear("rsi_overbought", f"RSI Overbought ({rsi:.1f})", "⚠️")
        elif rsi > lv.rsi_high:
            add_bear("rsi_high", f"RSI High ({rsi:.1f})", "📊")

        # 2. MACD crossover (mutually exclusive)
        if macd_val > macd_signal_val:
            add_bull("macd_bull_cross", "MACD Bullish Crossover", "✅")
        else:
            add_bear("macd_bear_cross", "MACD Bearish", "❌")

        # 3. MACD histogram momentum (exclusive)
        if macd_hist > 0 and macd_hist_rising:
            add_bull("macd_hist_rising", "MACD Histogram Rising", "📈")
        elif macd_hist < 0 and not macd_hist_rising:
            add_bear("macd_hist_falling", "MACD Histogram Falling", "📉")

        # 4. Below 20-SMA (dip buy)
        if price < sma20:
            add_bull("below_sma20_dip", "Below 20-SMA (dip buy)", "⬇️")

        # 5. Stochastic (oversold/overbought exclusive)
        if stoch_k < lv.stoch_oversold and stoch_k > stoch_d:
            add_bull("stoch_bull_cross", f"Stochastic Bullish Cross ({stoch_k:.1f})", "🔄")
        elif stoch_k < lv.stoch_oversold:
            add_bull("stoch_oversold", f"Stochastic Oversold ({stoch_k:.1f})", "🔻")
        elif stoch_k > lv.stoch_overbought and stoch_k < stoch_d:
            add_bear("stoch_bear_cross", f"Stochastic Bearish Cross ({stoch_k:.1f})", "🔁")
        elif stoch_k > lv.stoch_overbought:
            add_bear("stoch_overbought", f"Stochastic Overbought ({stoch_k:.1f})", "🔺")

        # 6 + 22. 52-week proximity — dominant condition only (mutually exclusive)
        if dist_52w_high <= dist_52w_low:
            # Closer to the high than to the low → momentum / strength.
            if dist_52w_high < lv.near_52w_high_pct:
                add_bull("near_52w_high", f"Near 52W High ({dist_52w_high:.1f}% away)", "🚀")
            elif dist_52w_high < lv.close_52w_high_pct:
                add_info(f"Close to 52W High ({dist_52w_high:.1f}% away)", "📍")
        else:
            # Closer to the low than to the high.
            if dist_52w_low < lv.near_52w_low_pct:
                # Near 52W low is treated as a downside-risk (bearish) condition.
                add_bear("near_52w_low_risk", f"Near 52W Low ({dist_52w_low:.1f}% above)", "🛑")

        # 7. SMA alignment: golden cross vs death cross (exclusive)
        if price > sma20 > sma50 > sma200:
            add_bull("golden_cross", "Golden Cross (20>50>200 SMA)", "⭐")
        elif price < sma20 < sma50 < sma200:
            add_bear("death_cross", "Death Cross (20<50<200 SMA)", "💀")

        # 8. Bollinger band breaches (exclusive)
        if price < bb_lower:
            add_bull("below_bb_lower", "Below Bollinger Lower Band", "💎")
        elif price > bb_upper:
            add_bear("above_bb_upper", "Above Bollinger Upper Band", "🎈")

        # 9. EMA 9/21 (exclusive)
        if ema_cross_bull:
            add_bull("ema_bull_cross", "EMA 9/21 Bullish Crossover", "🔀")
        elif ema_cross_bear:
            add_bear("ema_bear_cross", "EMA 9/21 Bearish Crossover", "🔂")
        elif ema9 > ema21:
            add_bull("ema_fast_above_slow", "EMA 9 above EMA 21", "📐")
        elif ema9 < ema21:
            add_bear("ema_fast_below_slow", "EMA 9 below EMA 21", "📏")

        # 10. ADX trend strength + DI dominance (exclusive)
        if adx_val > lv.adx_strong and plus_di > minus_di:
            add_bull("adx_strong_up", f"Strong Uptrend ADX={adx_val:.0f} DI+>DI-", "💪")
        elif adx_val > lv.adx_moderate and plus_di > minus_di:
            add_bull("adx_moderate_up", f"Moderate Uptrend ADX={adx_val:.0f}", "📊")
        elif adx_val > lv.adx_strong and minus_di > plus_di:
            add_bear("adx_strong_down", f"Strong Downtrend ADX={adx_val:.0f} DI->DI+", "⛔")
        elif adx_val > lv.adx_moderate and minus_di > plus_di:
            add_bear("adx_moderate_down", f"Moderate Downtrend ADX={adx_val:.0f}", "📉")

        # 11. CCI (exclusive)
        if cci_val < lv.cci_oversold:
            add_bull("cci_oversold", f"CCI Oversold ({cci_val:.0f})", "🎯")
        elif cci_val > lv.cci_overbought:
            add_bear("cci_overbought", f"CCI Overbought ({cci_val:.0f})", "🔔")

        # 12. Williams %R (exclusive)
        if wr_val < lv.williams_oversold:
            add_bull("williams_oversold", f"Williams %R Oversold ({wr_val:.0f})", "🏷️")
        elif wr_val > lv.williams_overbought:
            add_bear("williams_overbought", f"Williams %R Overbought ({wr_val:.0f})", "🏴")

        # 13. MFI (exclusive)
        if mfi_val < lv.mfi_oversold:
            add_bull("mfi_oversold", f"MFI Oversold ({mfi_val:.0f}) — Money flowing in", "💰")
        elif mfi_val < lv.mfi_low:
            add_bull("mfi_low", f"MFI Low ({mfi_val:.0f})", "💵")
        elif mfi_val > lv.mfi_overbought:
            add_bear("mfi_overbought", f"MFI Overbought ({mfi_val:.0f})", "💸")
        elif mfi_val > lv.mfi_high:
            add_bear("mfi_high", f"MFI High ({mfi_val:.0f})", "💳")

        # 14. OBV vs price (exclusive)
        if obv_slope > 0 and change > 0:
            add_bull("obv_rising", "OBV Rising — Volume confirms uptrend", "📊")
        elif obv_slope < 0 and change < 0:
            add_bear("obv_falling", "OBV Falling — Volume confirms downtrend", "📉")

        # 15. Price vs VWAP (exclusive)
        if price < vwap_val:
            add_bull("below_vwap", f"Below VWAP (₹{vwap_val:.0f}) — Undervalued", "🎪")
        elif price > vwap_val:
            add_bear("above_vwap", f"Above VWAP (₹{vwap_val:.0f}) — Extended", "🎢")

        # 16. Rate of Change (exclusive)
        if roc_val > lv.roc_strong:
            add_bull("roc_strong", f"Strong Momentum ROC={roc_val:.1f}%", "⚡")
        elif roc_val < lv.roc_weak:
            add_bear("roc_weak", f"Weak Momentum ROC={roc_val:.1f}%", "🐌")

        # 17. Chaikin Money Flow (exclusive)
        if cmf_val > lv.cmf_positive:
            add_bull("cmf_positive", f"CMF Positive ({cmf_val:.2f}) — Buying pressure", "🏦")
        elif cmf_val < lv.cmf_negative:
            add_bear("cmf_negative", f"CMF Negative ({cmf_val:.2f}) — Selling pressure", "🔴")

        # 18. Ichimoku cloud (exclusive)
        if price > senkou_a and price > senkou_b:
            add_bull("above_cloud", "Above Ichimoku Cloud — Bullish", "☁️")
        elif price < senkou_a and price < senkou_b:
            add_bear("below_cloud", "Below Ichimoku Cloud — Bearish", "🌧️")

        # 19. Ichimoku TK cross (exclusive)
        if tenkan > kijun:
            add_bull("ichimoku_tk_bull", "Ichimoku TK Cross Bullish", "⛩️")
        elif tenkan < kijun:
            add_bear("ichimoku_tk_bear", "Ichimoku TK Cross Bearish", "🌫️")

        # 20. Pivot R1 / S1 (exclusive — price can only be on one side)
        if price > r1:
            add_bull("above_r1", f"Above R1 Pivot (₹{r1:.0f}) — Strong", "🏔️")
        elif price < s1:
            add_bear("below_s1", f"Below S1 Support (₹{s1:.0f})", "🕳️")

        # 21. Bollinger squeeze (low volatility, neutral-bullish breakout setup)
        if bb_width < lv.bb_squeeze_pct:
            add_bull("bb_squeeze", f"Bollinger Squeeze ({bb_width:.1f}%) — Breakout imminent", "🤏")

        # 23. Volume spike (exclusive based on price direction)
        if vol_ratio > lv.vol_spike_ratio and change > 0:
            add_bull("volume_spike_up", f"Volume Spike {vol_ratio:.1f}x + Price Up", "🔊")
        elif vol_ratio > lv.vol_spike_ratio and change < 0:
            add_info(f"High Volume on Down Day ({vol_ratio:.1f}x avg)", "🔊")
        elif vol_ratio > lv.vol_spike_ratio:
            add_info(f"High Volume ({vol_ratio:.1f}x avg)", "🔊")

        # 24. Candle streaks (green vs red are exclusive by construction)
        if green_count >= lv.green_candles_min:
            add_bull("green_candles", f"{green_count} Green Candles in a row", "🟢")
        elif red_count >= lv.red_candles_min:
            add_bear("red_candles", f"{red_count} Red Candles in a row", "🔴")

        # 25. Long-term trend vs SMA200 (exclusive)
        if price > sma200 and dist_sma200 > lv.dist_sma200_pct:
            add_bull("above_sma200", f"Above 200-SMA by {dist_sma200:.1f}%", "🏗️")
        elif price < sma200:
            add_bear("below_sma200", f"Below 200-SMA ({dist_sma200:.1f}%)", "🔻")

        net_score = bull - bear
        if net_score >= strategy.thresholds.buy:
            signal = "BUY"
        elif net_score <= strategy.thresholds.sell:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        clean_name = symbol.replace("NSE:", "").replace("-EQ", "")

        return {
            "symbol": symbol,
            "name": clean_name,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "score": bull,                 # backward-compatible bull score
            "bear_score": bear,            # NEW
            "net_score": net_score,        # NEW
            "signal": signal,              # NEW (BUY / NEUTRAL / SELL)
            "rsi": round(rsi, 1),
            "macd": round(macd_val, 2),
            "macd_signal": round(macd_signal_val, 2),
            "macd_hist": round(macd_hist, 2),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "stoch_k": round(stoch_k, 1),
            "stoch_d": round(stoch_d, 1),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "bb_width": round(bb_width, 1),
            "w52_high": round(w52_high, 2),
            "w52_low": round(w52_low, 2),
            "dist_52w": round(dist_52w_high, 1),
            "volume": int(curr_vol),
            "avg_volume": int(avg_vol),
            "vol_ratio": round(vol_ratio, 1),
            "atr": round(atr_val, 2),
            "atr_pct": round(atr_pct, 1),
            "adx": round(adx_val, 1),
            "plus_di": round(plus_di, 1),
            "minus_di": round(minus_di, 1),
            "cci": round(cci_val, 0),
            "williams_r": round(wr_val, 0),
            "mfi": round(mfi_val, 1),
            "vwap": round(vwap_val, 2),
            "roc": round(roc_val, 1),
            "cmf": round(cmf_val, 3),
            "ichimoku_tenkan": round(tenkan, 2),
            "ichimoku_kijun": round(kijun, 2),
            "pivot": round(pp, 2),
            "pivot_s1": round(s1, 2),
            "pivot_r1": round(r1, 2),
            "reasons": reasons,
        }
    except Exception as e:
        logger.exception(f"analyze_stock failed for {symbol}: {e}")
        return None

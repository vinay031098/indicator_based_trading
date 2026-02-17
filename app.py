"""
NSE Indicator-Based Trading Dashboard — Flask Backend
Provides: Authentication, date-based analysis, stock data via API endpoints
Supports: NIFTY 50, NIFTY 100, NIFTY 200, NIFTY 500, and ALL NSE stocks (~2100)
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import webbrowser
import threading
import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file (GITHUB_TOKEN, GEMINI_API_KEY, etc.)
from datetime import datetime, timedelta
from flyers_integration import FyersClient, NIFTY_50_FYERS, get_symbols_for_category, fetch_all_nse_equity_symbols
from fyers_apiv3 import fyersModel
from llm_analyzer import analyze_with_llm
from data_store import (
    get_available_dates, get_run_by_date, get_stored_analysis,
    create_run, save_stock_analysis, save_ai_recommendations, update_run_status
)
import requests

# ─── Config (env-based for local vs production) ─────────────────────────
PRODUCTION = os.environ.get("PRODUCTION", "0") == "1"
APP_ID = os.environ.get("FYERS_APP_ID", "HTEDSURO6P-100")
SECRET_ID = os.environ.get("FYERS_SECRET_ID", "6E0U40KRQT")
DOMAIN = os.environ.get("DOMAIN", "belezabrasileiro.com")

# Always use fyersapiapp.com — Fyers redirects there, user pastes URL back
REDIRECT_URI = "https://fyersapiapp.com"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "nifty50_indicators_trade_2026")

# Global client
fyers_client = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def auto_connect():
    """Auto-connect to Fyers: try saved token first, fallback to browser login."""
    global fyers_client
    fyers_client = FyersClient(APP_ID, SECRET_ID, redirect_uri=REDIRECT_URI)
    token_file = os.path.join(BASE_DIR, '.fyers_token')

    # Try saved token first
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            saved_token = f.read().strip()
        if saved_token:
            fyers_client.set_access_token(saved_token)
            try:
                profile = fyers_client.get_profile()
                if profile and profile.get('s') == 'ok':
                    print(f"\n✅ Auto-connected using saved token!")
                    if not PRODUCTION:
                        webbrowser.open("http://127.0.0.1:5000")
                    return
                else:
                    # Token may be expired for profile but could still work for data
                    # Keep it set — will fail gracefully on /api/analyze if truly expired
                    print(f"\n⚠️  Saved token may be expired — keeping it, will verify on use")
                    if not PRODUCTION:
                        webbrowser.open("http://127.0.0.1:5000")
                    return
            except:
                # Same — keep token, let it fail on actual API call
                print(f"\n⚠️  Saved token check failed — keeping it, will verify on use")
                if not PRODUCTION:
                    webbrowser.open("http://127.0.0.1:5000")
                return

    # Fallback: open browser for login (local only) or wait for /auth/login
    session_model = fyersModel.SessionModel(
        client_id=APP_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
        state="indicators_trade",
        scope="",
        nonce=""
    )
    auth_url = session_model.generate_authcode()

    if PRODUCTION:
        print(f"\n🔑 Token expired. Login at: https://{DOMAIN}/auth/login")
    else:
        print(f"\n📊 Opening dashboard...")
        webbrowser.open("http://127.0.0.1:5000")
        import time; time.sleep(0.5)
        print(f"🔑 Opening Fyers login...")
        webbrowser.open(auth_url)


# ─── Technical Indicator Calculations ────────────────────────────────────

def _ema(data, period):
    arr = np.asarray(data, dtype=float).flatten()
    if len(arr) < period:
        return arr
    ema = np.zeros(len(arr))
    ema[0] = np.mean(arr[:period])
    k = 2.0 / (period + 1)
    for i in range(1, len(arr)):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema


def _sma(data, period):
    arr = np.asarray(data, dtype=float).flatten()
    if len(arr) < period:
        return np.full(len(arr), np.mean(arr))
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.mean(arr[i - period + 1:i + 1])
    return result


def _atr(high, low, close, period=14):
    """Average True Range."""
    h = np.asarray(high, dtype=float).flatten()
    l = np.asarray(low, dtype=float).flatten()
    c = np.asarray(close, dtype=float).flatten()
    tr = np.zeros(len(c))
    tr[0] = h[0] - l[0]
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = np.zeros(len(c))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(c)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def _adx(high, low, close, period=14):
    """Average Directional Index."""
    h = np.asarray(high, dtype=float).flatten()
    l = np.asarray(low, dtype=float).flatten()
    c = np.asarray(close, dtype=float).flatten()
    n = len(c)
    if n < period * 2:
        return 0, 0, 0

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up = h[i] - h[i-1]
        down = l[i-1] - l[i]
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))

    atr = _ema(tr, period)
    plus_di = 100 * _ema(plus_dm, period) / np.where(atr > 0, atr, 1)
    minus_di = 100 * _ema(minus_dm, period) / np.where(atr > 0, atr, 1)

    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
    adx = _ema(dx, period)
    return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])


def _cci(high, low, close, period=20):
    """Commodity Channel Index."""
    tp = (np.asarray(high) + np.asarray(low) + np.asarray(close)) / 3
    tp = tp.flatten()
    sma = np.mean(tp[-period:])
    md = np.mean(np.abs(tp[-period:] - sma))
    return float((tp[-1] - sma) / (0.015 * md)) if md != 0 else 0


def _williams_r(high, low, close, period=14):
    """Williams %R."""
    h = np.asarray(high, dtype=float).flatten()
    l = np.asarray(low, dtype=float).flatten()
    c = np.asarray(close, dtype=float).flatten()
    hh = np.max(h[-period:])
    ll = np.min(l[-period:])
    return float(-100 * (hh - c[-1]) / (hh - ll)) if (hh - ll) != 0 else -50


def _mfi(high, low, close, volume, period=14):
    """Money Flow Index."""
    tp = (np.asarray(high) + np.asarray(low) + np.asarray(close)).flatten() / 3
    mf = tp * np.asarray(volume, dtype=float).flatten()
    pos_mf = 0
    neg_mf = 0
    for i in range(-period, 0):
        if tp[i] > tp[i-1]:
            pos_mf += mf[i]
        else:
            neg_mf += mf[i]
    mfr = pos_mf / neg_mf if neg_mf > 0 else 100
    return float(100 - (100 / (1 + mfr)))


def _obv(close, volume):
    """On-Balance Volume."""
    c = np.asarray(close, dtype=float).flatten()
    v = np.asarray(volume, dtype=float).flatten()
    obv = np.zeros(len(c))
    for i in range(1, len(c)):
        if c[i] > c[i-1]:
            obv[i] = obv[i-1] + v[i]
        elif c[i] < c[i-1]:
            obv[i] = obv[i-1] - v[i]
        else:
            obv[i] = obv[i-1]
    return obv


def _vwap(high, low, close, volume):
    """Volume Weighted Average Price (approx using available data)."""
    tp = (np.asarray(high) + np.asarray(low) + np.asarray(close)).flatten() / 3
    v = np.asarray(volume, dtype=float).flatten()
    cum_tpv = np.cumsum(tp * v)
    cum_v = np.cumsum(v)
    vwap = cum_tpv / np.where(cum_v > 0, cum_v, 1)
    return float(vwap[-1])


def _roc(data, period=12):
    """Rate of Change."""
    arr = np.asarray(data, dtype=float).flatten()
    if len(arr) <= period:
        return 0
    return float((arr[-1] - arr[-1 - period]) / arr[-1 - period] * 100) if arr[-1 - period] != 0 else 0


def _cmf(high, low, close, volume, period=20):
    """Chaikin Money Flow."""
    h = np.asarray(high, dtype=float).flatten()
    l = np.asarray(low, dtype=float).flatten()
    c = np.asarray(close, dtype=float).flatten()
    v = np.asarray(volume, dtype=float).flatten()
    mfm = np.where((h - l) != 0, ((c - l) - (h - c)) / (h - l), 0)
    mfv = mfm * v
    return float(np.sum(mfv[-period:])) / float(np.sum(v[-period:])) if np.sum(v[-period:]) > 0 else 0


def _ichimoku(high, low, close):
    """Ichimoku Cloud - Tenkan, Kijun, Senkou A & B."""
    h = np.asarray(high, dtype=float).flatten()
    l = np.asarray(low, dtype=float).flatten()
    c = np.asarray(close, dtype=float).flatten()
    tenkan = (np.max(h[-9:]) + np.min(l[-9:])) / 2
    kijun = (np.max(h[-26:]) + np.min(l[-26:])) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (np.max(h[-52:]) + np.min(l[-52:])) / 2 if len(h) >= 52 else (np.max(h) + np.min(l)) / 2
    return float(tenkan), float(kijun), float(senkou_a), float(senkou_b)


def _pivot_points(high, low, close):
    """Classic Pivot Points."""
    h = float(np.asarray(high, dtype=float).flatten()[-1])
    l = float(np.asarray(low, dtype=float).flatten()[-1])
    c = float(np.asarray(close, dtype=float).flatten()[-1])
    pp = (h + l + c) / 3
    s1 = 2 * pp - h
    r1 = 2 * pp - l
    s2 = pp - (h - l)
    r2 = pp + (h - l)
    return pp, s1, r1, s2, r2


def analyze_stock(symbol, data):
    """Analyze a single stock with 25+ indicators and return detailed data."""
    try:
        close = data['Close'].values.flatten()
        high = data['High'].values.flatten()
        low = data['Low'].values.flatten()
        volume = data['Volume'].values.flatten()

        price = float(close[-1])
        prev_close = float(close[-2]) if len(close) > 1 else price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0

        # ═══════════════════════════════════════════════════════════
        # 1. RSI (14)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = float(100 - (100 / (1 + rs)))

        # 2. SMAs (20, 50, 200)
        sma20 = float(np.mean(close[-20:]))
        sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else float(np.mean(close))
        sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else float(np.mean(close))

        # 3. EMAs (9, 21)
        ema9 = float(_ema(close, 9)[-1])
        ema21 = float(_ema(close, 21)[-1])

        # 4. MACD (12, 26, 9)
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        macd = float(macd_line[-1])
        macd_signal = float(signal_line[-1])
        macd_hist = macd - macd_signal
        prev_macd_hist = float(macd_line[-2] - signal_line[-2]) if len(macd_line) > 1 else 0

        # 5. Stochastic (14, 3, 3)
        min14 = float(np.min(low[-14:]))
        max14 = float(np.max(high[-14:]))
        stoch_k = ((price - min14) / (max14 - min14) * 100) if max14 != min14 else 50
        stoch_d = float(np.mean([((close[-i] - np.min(low[-14-i:-i or None])) /
                    (np.max(high[-14-i:-i or None]) - np.min(low[-14-i:-i or None])) * 100)
                    if (np.max(high[-14-i:-i or None]) - np.min(low[-14-i:-i or None])) != 0 else 50
                    for i in range(1, 4)]))

        # 6. Bollinger Bands (20, 2)
        bb_sma = float(np.mean(close[-20:]))
        bb_std = float(np.std(close[-20:]))
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_sma * 100 if bb_sma > 0 else 0

        # 7. 52-Week High/Low
        w52_high = float(np.max(high[-252:])) if len(high) >= 252 else float(np.max(high))
        w52_low = float(np.min(low[-252:])) if len(low) >= 252 else float(np.min(low))
        dist_52w_high = ((w52_high - price) / w52_high) * 100
        dist_52w_low = ((price - w52_low) / w52_low) * 100 if w52_low > 0 else 0

        # 8. Volume Analysis
        avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        curr_vol = float(volume[-1])
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1

        # 9. ATR (14) — Average True Range
        atr_arr = _atr(high, low, close, 14)
        atr_val = float(atr_arr[-1])
        atr_pct = (atr_val / price * 100) if price > 0 else 0

        # 10. ADX (14) — Trend Strength
        adx_val, plus_di, minus_di = _adx(high, low, close, 14)

        # 11. CCI (20) — Commodity Channel Index
        cci_val = _cci(high, low, close, 20)

        # 12. Williams %R (14)
        williams_r = _williams_r(high, low, close, 14)

        # 13. MFI (14) — Money Flow Index
        mfi_val = _mfi(high, low, close, volume, 14)

        # 14. OBV — On-Balance Volume
        obv_arr = _obv(close, volume)
        obv_slope = float(obv_arr[-1] - obv_arr[-5]) if len(obv_arr) >= 5 else 0

        # 15. VWAP
        vwap_val = _vwap(high, low, close, volume)

        # 16. Rate of Change (12)
        roc_val = _roc(close, 12)

        # 17. Chaikin Money Flow (20)
        cmf_val = _cmf(high, low, close, volume, 20)

        # 18. Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b = _ichimoku(high, low, close)

        # 19. Pivot Points
        pp, s1, r1, s2, r2 = _pivot_points(high, low, close)

        # 20. EMA Cross (9/21)
        prev_ema9 = float(_ema(close[:-1], 9)[-1]) if len(close) > 10 else ema9
        prev_ema21 = float(_ema(close[:-1], 21)[-1]) if len(close) > 22 else ema21
        ema_cross_bull = (ema9 > ema21) and (prev_ema9 <= prev_ema21)

        # 21. Price vs VWAP
        price_vs_vwap = ((price - vwap_val) / vwap_val * 100) if vwap_val > 0 else 0

        # 22. Consecutive Green/Red candles
        green_count = 0
        for i in range(-1, -min(10, len(close)), -1):
            if close[i] > close[i-1]:
                green_count += 1
            else:
                break

        # 23. Distance from SMA200 (trend)
        dist_sma200 = ((price - sma200) / sma200 * 100) if sma200 > 0 else 0

        # 24. Bollinger %B
        bb_pct_b = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

        # 25. MACD Histogram momentum (rising/falling)
        macd_hist_rising = macd_hist > prev_macd_hist

        # ═══════════════════════════════════════════════════════════
        # ─── SCORING (25 indicators, max ~30 points) ──────────────
        # ═══════════════════════════════════════════════════════════
        score = 0
        reasons = []

        # 1. RSI — Relative Strength Index
        if rsi < 30:
            score += 2
            reasons.append({"text": f"RSI Oversold ({rsi:.1f})", "type": "bullish", "icon": "📉"})
        elif rsi < 40:
            score += 1
            reasons.append({"text": f"RSI Low ({rsi:.1f})", "type": "bullish", "icon": "📊"})
        elif rsi > 70:
            reasons.append({"text": f"RSI Overbought ({rsi:.1f})", "type": "bearish", "icon": "⚠️"})

        # 2. MACD Crossover
        if macd > macd_signal:
            score += 1
            reasons.append({"text": "MACD Bullish Crossover", "type": "bullish", "icon": "✅"})
        else:
            reasons.append({"text": "MACD Bearish", "type": "bearish", "icon": "❌"})

        # 3. MACD Histogram Rising
        if macd_hist > 0 and macd_hist_rising:
            score += 1
            reasons.append({"text": "MACD Histogram Rising", "type": "bullish", "icon": "📈"})

        # 4. Below 20-SMA (Dip Buy)
        if price < sma20:
            score += 1
            reasons.append({"text": "Below 20-SMA (dip buy)", "type": "bullish", "icon": "⬇️"})

        # 5. Stochastic Oversold
        if stoch_k < 20:
            score += 1
            reasons.append({"text": f"Stochastic Oversold ({stoch_k:.1f})", "type": "bullish", "icon": "🔻"})
        elif stoch_k < 20 and stoch_k > stoch_d:
            score += 2
            reasons.append({"text": f"Stochastic Bullish Cross ({stoch_k:.1f})", "type": "bullish", "icon": "🔄"})

        # 6. Near 52W High (Momentum)
        if dist_52w_high < 5:
            score += 2
            reasons.append({"text": f"Near 52W High ({dist_52w_high:.1f}% away)", "type": "bullish", "icon": "🚀"})
        elif dist_52w_high < 10:
            score += 1
            reasons.append({"text": f"Close to 52W High ({dist_52w_high:.1f}% away)", "type": "info", "icon": "📍"})

        # 7. Golden Cross (SMA alignment)
        if price > sma20 > sma50 > sma200:
            score += 2
            reasons.append({"text": "Golden Cross (20>50>200 SMA)", "type": "bullish", "icon": "⭐"})

        # 8. Below Bollinger Lower Band
        if price < bb_lower:
            score += 1
            reasons.append({"text": "Below Bollinger Lower Band", "type": "bullish", "icon": "💎"})

        # 9. EMA 9/21 Bullish Cross
        if ema_cross_bull:
            score += 2
            reasons.append({"text": "EMA 9/21 Bullish Crossover", "type": "bullish", "icon": "🔀"})
        elif ema9 > ema21:
            score += 1
            reasons.append({"text": "EMA 9 above EMA 21", "type": "bullish", "icon": "📐"})

        # 10. ADX Trending + DI+
        if adx_val > 25 and plus_di > minus_di:
            score += 2
            reasons.append({"text": f"Strong Uptrend ADX={adx_val:.0f} DI+>DI-", "type": "bullish", "icon": "💪"})
        elif adx_val > 20 and plus_di > minus_di:
            score += 1
            reasons.append({"text": f"Moderate Uptrend ADX={adx_val:.0f}", "type": "bullish", "icon": "📊"})
        elif adx_val > 25 and minus_di > plus_di:
            reasons.append({"text": f"Strong Downtrend ADX={adx_val:.0f}", "type": "bearish", "icon": "⛔"})

        # 11. CCI Oversold
        if cci_val < -100:
            score += 1
            reasons.append({"text": f"CCI Oversold ({cci_val:.0f})", "type": "bullish", "icon": "🎯"})
        elif cci_val > 100:
            reasons.append({"text": f"CCI Overbought ({cci_val:.0f})", "type": "info", "icon": "🔔"})

        # 12. Williams %R Oversold
        if williams_r < -80:
            score += 1
            reasons.append({"text": f"Williams %R Oversold ({williams_r:.0f})", "type": "bullish", "icon": "🏷️"})

        # 13. MFI — Money Flow
        if mfi_val < 20:
            score += 2
            reasons.append({"text": f"MFI Oversold ({mfi_val:.0f}) — Money flowing in", "type": "bullish", "icon": "💰"})
        elif mfi_val < 40:
            score += 1
            reasons.append({"text": f"MFI Low ({mfi_val:.0f})", "type": "bullish", "icon": "💵"})
        elif mfi_val > 80:
            reasons.append({"text": f"MFI Overbought ({mfi_val:.0f})", "type": "bearish", "icon": "💸"})

        # 14. OBV Rising (Volume confirming price)
        if obv_slope > 0 and change > 0:
            score += 1
            reasons.append({"text": "OBV Rising — Volume confirms uptrend", "type": "bullish", "icon": "📊"})

        # 15. Price below VWAP (Buy zone)
        if price < vwap_val:
            score += 1
            reasons.append({"text": f"Below VWAP (₹{vwap_val:.0f}) — Undervalued", "type": "bullish", "icon": "🎪"})

        # 16. Rate of Change positive
        if roc_val > 5:
            score += 1
            reasons.append({"text": f"Strong Momentum ROC={roc_val:.1f}%", "type": "bullish", "icon": "⚡"})
        elif roc_val < -10:
            reasons.append({"text": f"Weak Momentum ROC={roc_val:.1f}%", "type": "bearish", "icon": "🐌"})

        # 17. Chaikin Money Flow positive
        if cmf_val > 0.1:
            score += 1
            reasons.append({"text": f"CMF Positive ({cmf_val:.2f}) — Buying pressure", "type": "bullish", "icon": "🏦"})
        elif cmf_val < -0.1:
            reasons.append({"text": f"CMF Negative ({cmf_val:.2f}) — Selling pressure", "type": "bearish", "icon": "🔴"})

        # 18. Ichimoku — Price above cloud
        if price > senkou_a and price > senkou_b:
            score += 1
            reasons.append({"text": "Above Ichimoku Cloud — Bullish", "type": "bullish", "icon": "☁️"})
        elif price < senkou_a and price < senkou_b:
            reasons.append({"text": "Below Ichimoku Cloud — Bearish", "type": "bearish", "icon": "🌧️"})

        # 19. Ichimoku — Tenkan > Kijun (bullish)
        if tenkan > kijun:
            score += 1
            reasons.append({"text": "Ichimoku TK Cross Bullish", "type": "bullish", "icon": "⛩️"})

        # 20. Pivot — Price above Pivot Point
        if price > r1:
            score += 1
            reasons.append({"text": f"Above R1 Pivot (₹{r1:.0f}) — Strong", "type": "bullish", "icon": "🏔️"})
        elif price < s1:
            reasons.append({"text": f"Below S1 Support (₹{s1:.0f})", "type": "bearish", "icon": "🕳️"})

        # 21. Bollinger Squeeze (low volatility = breakout soon)
        if bb_width < 5:
            score += 1
            reasons.append({"text": f"Bollinger Squeeze ({bb_width:.1f}%) — Breakout imminent", "type": "bullish", "icon": "🤏"})

        # 22. Near 52W Low (Value Buy)
        if dist_52w_low < 10:
            score += 1
            reasons.append({"text": f"Near 52W Low ({dist_52w_low:.1f}% above) — Value buy", "type": "bullish", "icon": "🛒"})

        # 23. Volume Spike + Price Up
        if vol_ratio > 1.5 and change > 0:
            score += 1
            reasons.append({"text": f"Volume Spike {vol_ratio:.1f}x + Price Up", "type": "bullish", "icon": "🔊"})
        elif vol_ratio > 1.5:
            reasons.append({"text": f"High Volume ({vol_ratio:.1f}x avg)", "type": "info", "icon": "🔊"})

        # 24. Consecutive Green Candles
        if green_count >= 3:
            score += 1
            reasons.append({"text": f"{green_count} Green Candles in a row", "type": "bullish", "icon": "🟢"})

        # 25. Above SMA200 (Long-term uptrend)
        if price > sma200 and dist_sma200 > 5:
            score += 1
            reasons.append({"text": f"Above 200-SMA by {dist_sma200:.1f}%", "type": "bullish", "icon": "🏗️"})
        elif price < sma200:
            reasons.append({"text": f"Below 200-SMA ({dist_sma200:.1f}%)", "type": "bearish", "icon": "🔻"})

        # Clean symbol name
        clean_name = symbol.replace("NSE:", "").replace("-EQ", "")

        return {
            "symbol": symbol,
            "name": clean_name,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "score": score,
            "rsi": round(rsi, 1),
            "macd": round(macd, 2),
            "macd_signal": round(macd_signal, 2),
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
            "williams_r": round(williams_r, 0),
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
        return None


# ─── Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard page."""
    authenticated = fyers_client is not None and fyers_client.access_token is not None
    return render_template("index.html", authenticated=authenticated, production=PRODUCTION)


@app.route("/auth/login")
def auth_login():
    """Return Fyers login URL. Frontend opens it in new tab."""
    global fyers_client
    if fyers_client is None:
        fyers_client = FyersClient(APP_ID, SECRET_ID, redirect_uri=REDIRECT_URI)

    session_model = fyersModel.SessionModel(
        client_id=APP_ID,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code",
        state="indicators_trade",
        scope="",
        nonce=""
    )
    auth_url = session_model.generate_authcode()
    return jsonify({"auth_url": auth_url})


@app.route("/auth/callback")
def auth_callback():
    """Fyers redirects here after login (production mode).
    Auto-exchanges auth_code for token and redirects to dashboard."""
    auth_code = request.args.get("auth_code", "")
    status = request.args.get("s", "")

    if status != "ok" or not auth_code:
        return render_template("index.html", authenticated=False,
                               error="Authentication failed. Please try again.", production=PRODUCTION)

    global fyers_client
    if fyers_client is None:
        fyers_client = FyersClient(APP_ID, SECRET_ID, redirect_uri=REDIRECT_URI)

    success = fyers_client.authenticate(auth_code=auth_code)

    if success:
        # Save token
        token_file = os.path.join(BASE_DIR, '.fyers_token')
        with open(token_file, 'w') as f:
            f.write(fyers_client.access_token)
        print(f"\n✅ Fyers connected via callback! Token saved.")
        return redirect(url_for("index"))
    else:
        return render_template("index.html", authenticated=False,
                               error="Token exchange failed. Try again.", production=PRODUCTION)


@app.route("/auth/connect", methods=["POST"])
def auth_connect():
    """Accept auth_code (or full redirect URL) from frontend and authenticate."""
    data = request.get_json()
    auth_input = data.get("auth_code", "").strip()

    # Extract auth_code from full URL if pasted
    if "auth_code=" in auth_input:
        import urllib.parse
        parsed = urllib.parse.urlparse(auth_input)
        params = urllib.parse.parse_qs(parsed.query)
        auth_input = params.get("auth_code", [""])[0]

    if not auth_input:
        return jsonify({"success": False, "message": "No auth_code found"})

    global fyers_client
    if fyers_client is None:
        fyers_client = FyersClient(APP_ID, SECRET_ID, redirect_uri=REDIRECT_URI)

    success = fyers_client.authenticate(auth_code=auth_input)

    if success:
        # Save token for next startup
        token_file = os.path.join(BASE_DIR, '.fyers_token')
        with open(token_file, 'w') as f:
            f.write(fyers_client.access_token)
        print(f"\n✅ Fyers connected! Token saved.")
    return jsonify({"success": success})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Run analysis for given date range and stock category. Returns JSON with all stock data."""
    global fyers_client
    if fyers_client is None or fyers_client.access_token is None:
        return jsonify({"error": "Not authenticated. Please login first."}), 401

    data = request.get_json()
    analysis_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    min_score = int(data.get("min_score", 2))
    category = data.get("category", "nifty50")

    # Calculate days of history needed (from 1 year before the date)
    try:
        target_date = datetime.strptime(analysis_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    days_from_now = (datetime.now() - target_date).days
    # We need at least 252 trading days (~1 year) of data before the target date
    total_days = days_from_now + 365

    # Get symbols for selected category
    try:
        symbols = get_symbols_for_category(category)
    except Exception as e:
        print(f"❌ Error getting symbols for {category}: {e}")
        return jsonify({"error": f"Failed to get stock list for {category}. Try again."}), 500

    category_label = {
        'nifty50': 'NIFTY 50',
        'nifty100': 'NIFTY 100',
        'nifty200': 'NIFTY 200',
        'nifty500': 'NIFTY 500',
        'all': f'All NSE ({len(symbols)})'
    }.get(category, 'NIFTY 50')

    print(f"\n🔍 Analyzing {category_label} stocks for {analysis_date}...")

    # Fetch stock data — may take minutes for large sets
    try:
        all_data = fyers_client.get_stock_data(symbols, resolution="D", days=total_days)
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        return jsonify({"error": f"Failed to fetch data from Fyers: {str(e)}"}), 500

    if not all_data:
        return jsonify({"error": "No stock data retrieved. Fyers API may be down or token expired. Try re-connecting."}), 500

    results = []
    skipped = []
    analysis_errors = 0
    for symbol, df in all_data.items():
        # Filter data up to the selected date
        try:
            filtered = df[df.index <= pd.Timestamp(analysis_date)]
            if len(filtered) < 50:
                skipped.append(symbol.replace("NSE:", "").replace("-EQ", ""))
                continue

            result = analyze_stock(symbol, filtered)
            if result:
                results.append(result)
        except Exception as e:
            analysis_errors += 1
            if analysis_errors <= 5:
                print(f"  ⚠️  Error analyzing {symbol}: {e}")

    if analysis_errors:
        print(f"  ⚠️  {analysis_errors} stocks had analysis errors")

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Split into qualified and unqualified
    qualified = [r for r in results if r["score"] >= min_score]
    unqualified = [r for r in results if r["score"] < min_score]

    print(f"✅ Analysis done: {len(results)} stocks analyzed, {len(qualified)} qualified")

    return jsonify({
        "date": analysis_date,
        "category": category,
        "category_label": category_label,
        "total_stocks": len(results),
        "qualified_count": len(qualified),
        "min_score": min_score,
        "qualified": qualified,
        "unqualified": unqualified,
        "skipped": skipped,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/llm-analyze", methods=["POST"])
def api_llm_analyze():
    """Send analyzed stock data to LLM for buy/hold/avoid recommendations.
    Also fetches 15-min candle data (last 3 days) for chart pattern analysis."""
    try:
        data = request.get_json()
        stocks = data.get("stocks", [])

        if not stocks:
            return jsonify({"error": "No stock data provided"}), 400

        # Pass fyers_client so LLM analyzer can fetch 15-min candle data
        result = analyze_with_llm(stocks, fyers_client=fyers_client)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)
    except Exception as e:
        print(f"!! LLM analyze error: {e}")
        return jsonify({"error": f"AI analysis failed: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chatbot endpoint — sends user question to GitHub Models for stock-related answers."""
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        stock_context = data.get("stock_context", [])

        if not message:
            return jsonify({"error": "Empty message"}), 400

        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            return jsonify({"error": "GITHUB_TOKEN not set. AI chat unavailable."}), 500

        # Build context from analyzed stocks if available
        context_text = ""
        if stock_context:
            context_text = "\n\nCurrently analyzed stocks (top results from dashboard):\n"
            for s in stock_context[:15]:
                context_text += f"  {s.get('name','?')}: ₹{s.get('price','?')} ({s.get('change_pct',0):+.1f}%) Score={s.get('score',0)}/30 RSI={s.get('rsi','?')} MACD_hist={s.get('macd_hist','?')} ADX={s.get('adx','?')}\n"

        system_prompt = (
            "You are a professional Indian stock market analyst assistant on an NSE trading dashboard.\n"
            "You have deep knowledge of NSE/BSE stocks, technical analysis, fundamental analysis, "
            "sectors, market trends, and Indian economy.\n\n"
            "Guidelines:\n"
            "- Give concise, actionable answers (2-4 sentences for simple questions, more for analysis)\n"
            "- Use ₹ symbol for prices\n"
            "- When analyzing a stock, mention key indicators like RSI, MACD, support/resistance\n"
            "- Be honest about uncertainty — say 'based on technical indicators' not 'will definitely'\n"
            "- Format important terms in <strong>bold</strong> HTML tags\n"
            "- Use <br> for line breaks in your response\n"
            "- If you mention Buy/Sell/Hold, make it clear this is not financial advice\n"
            + context_text
        )

        resp = requests.post(
            "https://models.inference.ai.azure.com/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.4,
                "max_tokens": 1024
            },
            timeout=30
        )

        if resp.status_code != 200:
            return jsonify({"error": f"AI error ({resp.status_code})"}), 500

        reply = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        if not reply:
            return jsonify({"error": "Empty AI response"}), 500

        return jsonify({"reply": reply})

    except requests.exceptions.Timeout:
        return jsonify({"error": "AI request timed out. Try again."}), 500
    except Exception as e:
        print(f"!! Chat error: {e}")
        return jsonify({"error": f"Chat failed: {str(e)}"}), 500


@app.route("/api/stored-dates")
def api_stored_dates():
    """Get list of all available pre-analyzed dates."""
    try:
        dates = get_available_dates()
        return jsonify({"dates": dates})
    except Exception as e:
        print(f"!! Error fetching stored dates: {e}")
        return jsonify({"dates": [], "error": str(e)})


@app.route("/api/stored-data", methods=["POST"])
def api_stored_data():
    """Get stored analysis data for a specific date."""
    try:
        data = request.get_json()
        run_date = data.get("date", "")
        category = data.get("category", "all")
        min_score = int(data.get("min_score", 0))

        if not run_date:
            return jsonify({"error": "No date provided"}), 400

        run = get_run_by_date(run_date, category)
        if not run:
            return jsonify({"error": f"No stored analysis found for {run_date} ({category})"}), 404

        result = get_stored_analysis(run['id'], min_score)
        return jsonify(result)

    except Exception as e:
        print(f"!! Error fetching stored data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/run-daily", methods=["POST"])
def api_run_daily():
    """Trigger daily analysis run from the UI (stores results in DB)."""
    global fyers_client
    if fyers_client is None or fyers_client.access_token is None:
        return jsonify({"error": "Not authenticated. Please login first."}), 401

    data = request.get_json()
    analysis_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    category = data.get("category", "all")
    min_score = int(data.get("min_score", 2))
    skip_ai = data.get("skip_ai", False)

    # Check if already exists
    existing = get_run_by_date(analysis_date, category)
    if existing and not data.get("force", False):
        return jsonify({
            "exists": True,
            "run": existing,
            "message": f"Analysis already exists for {analysis_date}. Use 'force' to overwrite."
        })

    try:
        # Create run record
        run_id = create_run(analysis_date, category, min_score)

        # Get symbols
        symbols = get_symbols_for_category(category)
        category_label = {
            'nifty50': 'NIFTY 50', 'nifty100': 'NIFTY 100',
            'nifty200': 'NIFTY 200', 'nifty500': 'NIFTY 500',
            'all': f'All NSE ({len(symbols)})'
        }.get(category, 'NIFTY 50')

        print(f"\n🔍 [Run {run_id}] Analyzing {category_label} for {analysis_date}...")

        # Fetch data
        target_date = datetime.strptime(analysis_date, "%Y-%m-%d")
        days_from_now = (datetime.now() - target_date).days
        total_days = days_from_now + 365

        all_data = fyers_client.get_stock_data(symbols, resolution="D", days=total_days)
        if not all_data:
            update_run_status(run_id, "failed")
            return jsonify({"error": "No stock data retrieved. Token may be expired."}), 500

        # Analyze stocks
        results = []
        for symbol, df in all_data.items():
            try:
                filtered = df[df.index <= pd.Timestamp(analysis_date)]
                if len(filtered) < 50:
                    continue
                result = analyze_stock(symbol, filtered)
                if result:
                    results.append(result)
            except:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        qualified = [r for r in results if r["score"] >= min_score]

        # Save to DB
        save_stock_analysis(run_id, results)

        # AI analysis
        ai_buy = ai_hold = ai_avoid = 0
        ai_completed = 0

        if not skip_ai and results:
            try:
                ai_result = analyze_with_llm(results, fyers_client=fyers_client)
                if "error" not in ai_result:
                    recs = ai_result.get("recommendations", {})
                    if recs:
                        save_ai_recommendations(run_id, recs)
                        ai_buy = sum(1 for r in recs.values() if r.get('action') == 'BUY')
                        ai_hold = sum(1 for r in recs.values() if r.get('action') == 'HOLD')
                        ai_avoid = sum(1 for r in recs.values() if r.get('action') == 'AVOID')
                        ai_completed = 1
            except Exception as e:
                print(f"  AI analysis error: {e}")

        # Mark completed
        update_run_status(run_id, "completed",
                          total_stocks=len(results),
                          qualified_count=len(qualified),
                          ai_completed=ai_completed,
                          ai_buy=ai_buy, ai_hold=ai_hold, ai_avoid=ai_avoid)

        print(f"✅ [Run {run_id}] Complete: {len(results)} stocks, AI={ai_completed}")

        # Return stored data
        stored = get_stored_analysis(run_id, min_score)
        return jsonify(stored)

    except Exception as e:
        print(f"!! Run daily error: {e}")
        try:
            update_run_status(run_id, "failed")
        except:
            pass
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/api/status")
def api_status():
    """Check auth status."""
    authenticated = fyers_client is not None and fyers_client.access_token is not None
    return jsonify({"authenticated": authenticated})


if __name__ == "__main__":
    mode = "PRODUCTION" if PRODUCTION else "LOCAL"
    url = f"https://{DOMAIN}" if PRODUCTION else "http://127.0.0.1:5000"

    print("\n" + "=" * 60)
    print(f"  NSE Trading Indicator Dashboard [{mode}]")
    print(f"  URL: {url}")
    print(f"  Redirect: {REDIRECT_URI}")
    print("=" * 60)

    # Auto-connect after server starts (1.5s delay)
    threading.Timer(1.5, auto_connect).start()

    if PRODUCTION:
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        app.run(debug=False, port=5000)

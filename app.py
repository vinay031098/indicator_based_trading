"""
NIFTY 50 Indicator-Based Trading Dashboard — Flask Backend
Provides: Authentication, date-based analysis, stock data via API endpoints
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import webbrowser
import threading
import os
from datetime import datetime, timedelta
from flyers_integration import FyersClient, NIFTY_50_FYERS
from fyers_apiv3 import fyersModel

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
                    print(f"\n⚠️  Saved token expired")
                    fyers_client.access_token = None
                    fyers_client.fyers = None
            except:
                print(f"\n⚠️  Saved token invalid")
                fyers_client.access_token = None
                fyers_client.fyers = None

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


def analyze_stock(symbol, data):
    """Analyze a single stock and return detailed indicator data."""
    try:
        close = data['Close'].values.flatten()
        high = data['High'].values.flatten()
        low = data['Low'].values.flatten()
        volume = data['Volume'].values.flatten()

        price = float(close[-1])
        prev_close = float(close[-2]) if len(close) > 1 else price
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0

        # RSI (14)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = float(100 - (100 / (1 + rs)))

        # SMAs
        sma20 = float(np.mean(close[-20:]))
        sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else float(np.mean(close))
        sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else float(np.mean(close))

        # MACD
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        macd = float(macd_line[-1])
        macd_signal = float(signal_line[-1])
        macd_hist = macd - macd_signal

        # Stochastic (14)
        min14 = float(np.min(low[-14:]))
        max14 = float(np.max(high[-14:]))
        stoch_k = ((price - min14) / (max14 - min14) * 100) if max14 != min14 else 50

        # Bollinger Bands (20, 2)
        bb_sma = float(np.mean(close[-20:]))
        bb_std = float(np.std(close[-20:]))
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std

        # 52-Week High/Low
        w52_high = float(np.max(high[-252:]))
        w52_low = float(np.min(low[-252:]))
        dist_52w_high = ((w52_high - price) / w52_high) * 100

        # Volume
        avg_vol = float(np.mean(volume[-20:])) if len(volume) >= 20 else float(np.mean(volume))
        curr_vol = float(volume[-1])
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1

        # ─── Scoring ──────────────────────────────────────────────
        score = 0
        reasons = []

        if rsi < 30:
            score += 2
            reasons.append({"text": f"RSI Oversold ({rsi:.1f})", "type": "bullish", "icon": "📉"})
        elif rsi < 40:
            score += 1
            reasons.append({"text": f"RSI Low ({rsi:.1f})", "type": "bullish", "icon": "📊"})
        elif rsi > 70:
            reasons.append({"text": f"RSI Overbought ({rsi:.1f})", "type": "info", "icon": "📈"})

        if macd > macd_signal:
            score += 1
            reasons.append({"text": "MACD Bullish Crossover", "type": "bullish", "icon": "✅"})
        else:
            reasons.append({"text": "MACD Bearish", "type": "bearish", "icon": "❌"})

        if price < sma20:
            score += 1
            reasons.append({"text": "Below 20-SMA (dip buy)", "type": "bullish", "icon": "⬇️"})

        if stoch_k < 20:
            score += 1
            reasons.append({"text": f"Stochastic Oversold ({stoch_k:.1f})", "type": "bullish", "icon": "🔻"})

        if dist_52w_high < 5:
            score += 2
            reasons.append({"text": f"Near 52W High ({dist_52w_high:.1f}% away)", "type": "bullish", "icon": "🚀"})
        elif dist_52w_high < 10:
            score += 1
            reasons.append({"text": f"Close to 52W High ({dist_52w_high:.1f}% away)", "type": "info", "icon": "📍"})

        if price > sma20 > sma50 > sma200:
            score += 2
            reasons.append({"text": "Golden Cross (20>50>200 SMA)", "type": "bullish", "icon": "⭐"})

        if price < bb_lower:
            score += 1
            reasons.append({"text": "Below Bollinger Lower Band", "type": "bullish", "icon": "💎"})

        if vol_ratio > 1.5:
            reasons.append({"text": f"High Volume ({vol_ratio:.1f}x avg)", "type": "info", "icon": "🔊"})

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
            "stoch_k": round(stoch_k, 1),
            "bb_upper": round(bb_upper, 2),
            "bb_lower": round(bb_lower, 2),
            "w52_high": round(w52_high, 2),
            "w52_low": round(w52_low, 2),
            "dist_52w": round(dist_52w_high, 1),
            "volume": int(curr_vol),
            "avg_volume": int(avg_vol),
            "vol_ratio": round(vol_ratio, 1),
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
    """Run analysis for given date range. Returns JSON with all stock data."""
    global fyers_client
    if fyers_client is None or fyers_client.access_token is None:
        return jsonify({"error": "Not authenticated. Please login first."}), 401

    data = request.get_json()
    analysis_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    min_score = int(data.get("min_score", 2))

    # Calculate days of history needed (from 1 year before the date)
    try:
        target_date = datetime.strptime(analysis_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    days_from_now = (datetime.now() - target_date).days
    # We need at least 252 trading days (~1 year) of data before the target date
    total_days = days_from_now + 365

    # Fetch all NIFTY 50 data
    all_data = fyers_client.get_all_nifty50_data(resolution="D", days=total_days)

    results = []
    skipped = []
    for symbol, df in all_data.items():
        # Filter data up to the selected date
        filtered = df[df.index <= pd.Timestamp(analysis_date)]
        if len(filtered) < 50:
            skipped.append(symbol.replace("NSE:", "").replace("-EQ", ""))
            continue

        result = analyze_stock(symbol, filtered)
        if result:
            results.append(result)

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    # Split into qualified and unqualified
    qualified = [r for r in results if r["score"] >= min_score]
    unqualified = [r for r in results if r["score"] < min_score]

    return jsonify({
        "date": analysis_date,
        "total_stocks": len(results),
        "qualified_count": len(qualified),
        "min_score": min_score,
        "qualified": qualified,
        "unqualified": unqualified,
        "skipped": skipped,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/status")
def api_status():
    """Check auth status."""
    authenticated = fyers_client is not None and fyers_client.access_token is not None
    return jsonify({"authenticated": authenticated})


if __name__ == "__main__":
    mode = "PRODUCTION" if PRODUCTION else "LOCAL"
    url = f"https://{DOMAIN}" if PRODUCTION else "http://127.0.0.1:5000"

    print("\n" + "=" * 60)
    print(f"  NIFTY 50 Trading Indicator Dashboard [{mode}]")
    print(f"  URL: {url}")
    print(f"  Redirect: {REDIRECT_URI}")
    print("=" * 60)

    # Auto-connect after server starts (1.5s delay)
    threading.Timer(1.5, auto_connect).start()

    if PRODUCTION:
        app.run(host="0.0.0.0", port=5000, debug=False)
    else:
        app.run(debug=False, port=5000)

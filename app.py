"""
NSE Indicator-Based Trading Dashboard — Flask Backend
Provides: Authentication, date-based analysis, stock data via API endpoints
Supports: NIFTY 50, NIFTY 100, NIFTY 200, NIFTY 500, and ALL NSE stocks (~2100)
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
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


# ─── Technical Indicator Calculations (imported from indicators.py) ─────
from indicators import analyze_stock

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

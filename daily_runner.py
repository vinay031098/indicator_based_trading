#!/usr/bin/env python3
"""
Daily Runner — Run full stock analysis + AI recommendations for all NSE stocks
and store results in SQLite database.

Usage:
  python3 daily_runner.py                    # Analyze ALL NSE stocks for today
  python3 daily_runner.py --date 2026-02-18  # Specific date
  python3 daily_runner.py --category nifty50 # Specific universe
  python3 daily_runner.py --skip-ai          # Skip AI analysis (indicators only)

Designed to be run once daily (e.g., after market hours via cron):
  30 16 * * 1-5 cd /path/to/project && python3 daily_runner.py >> logs/daily.log 2>&1
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from flyers_integration import FyersClient, get_symbols_for_category
from llm_analyzer import analyze_with_llm
from data_store import create_run, save_stock_analysis, save_ai_recommendations, update_run_status, get_run_by_date
from indicators import analyze_stock

# ─── Config ───────────────────────────────────────────────────────
APP_ID = os.environ.get("FYERS_APP_ID", "HTEDSURO6P-100")
SECRET_ID = os.environ.get("FYERS_SECRET_ID", "6E0U40KRQT")
REDIRECT_URI = "https://fyersapiapp.com"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_fyers_client():
    """Connect to Fyers using saved token."""
    token_file = os.path.join(BASE_DIR, '.fyers_token')
    if not os.path.exists(token_file):
        print("❌ No saved Fyers token found. Run the dashboard first and connect to Fyers.")
        print("   Then run this script — it uses the same saved token.")
        return None

    with open(token_file, 'r') as f:
        token = f.read().strip()

    if not token:
        print("❌ Fyers token file is empty.")
        return None

    client = FyersClient(APP_ID, SECRET_ID, redirect_uri=REDIRECT_URI)
    client.set_access_token(token)

    # Verify token
    try:
        profile = client.get_profile()
        if profile and profile.get('s') == 'ok':
            print(f"✅ Fyers connected: {profile.get('data', {}).get('name', 'Unknown')}")
            return client
        else:
            print(f"⚠️  Token may be expired but proceeding anyway...")
            return client
    except Exception as e:
        print(f"⚠️  Token verification failed ({e}), proceeding anyway...")
        return client


def run_daily_analysis(analysis_date: str, category: str = "all",
                        min_score: int = 2, skip_ai: bool = False):
    """
    Run full analysis pipeline:
    1. Connect to Fyers
    2. Fetch all stock data
    3. Analyze with 25 indicators
    4. Run AI recommendations
    5. Store everything in SQLite
    """
    start_time = time.time()
    print(f"\n{'='*60}")
    print(f"  📊 Daily Stock Analysis Runner")
    print(f"  Date: {analysis_date}")
    print(f"  Category: {category}")
    print(f"  Skip AI: {skip_ai}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Check if already run for this date
    existing = get_run_by_date(analysis_date, category)
    if existing:
        print(f"⚠️  Analysis already exists for {analysis_date} ({category}).")
        print(f"   Run ID: {existing['id']}, Stocks: {existing['total_stocks']}")
        response = input("   Overwrite? (y/N): ").strip().lower()
        if response != 'y':
            print("   Skipped.")
            return

    # Step 1: Connect to Fyers
    print("📡 Step 1: Connecting to Fyers...")
    client = get_fyers_client()
    if not client:
        print("❌ Cannot proceed without Fyers connection.")
        return

    # Step 2: Get symbols
    print(f"\n📋 Step 2: Getting {category} symbols...")
    try:
        symbols = get_symbols_for_category(category)
        print(f"   Found {len(symbols)} symbols")
    except Exception as e:
        print(f"❌ Error getting symbols: {e}")
        return

    # Step 3: Create run record
    run_id = create_run(analysis_date, category, min_score)
    print(f"\n💾 Created run ID: {run_id}")

    # Step 4: Fetch stock data
    print(f"\n📥 Step 3: Fetching historical data for {len(symbols)} stocks...")
    print(f"   This may take several minutes for large sets...")

    try:
        target_date = datetime.strptime(analysis_date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date format: {analysis_date}")
        return

    days_from_now = (datetime.now() - target_date).days
    total_days = days_from_now + 365  # 1 year of history

    try:
        all_data = client.get_stock_data(symbols, resolution="D", days=total_days)
    except Exception as e:
        print(f"❌ Error fetching stock data: {e}")
        update_run_status(run_id, "failed")
        return

    if not all_data:
        print("❌ No stock data retrieved. Token may be expired.")
        update_run_status(run_id, "failed")
        return

    print(f"   Received data for {len(all_data)} stocks")

    # Step 5: Analyze each stock
    print(f"\n🔍 Step 4: Running 25-indicator analysis...")
    results = []
    skipped = []
    errors = 0

    for i, (symbol, df) in enumerate(all_data.items()):
        try:
            filtered = df[df.index <= pd.Timestamp(analysis_date)]
            if len(filtered) < 50:
                skipped.append(symbol.replace("NSE:", "").replace("-EQ", ""))
                continue

            result = analyze_stock(symbol, filtered)
            if result:
                results.append(result)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"   ⚠️ Error analyzing {symbol}: {e}")

        if (i + 1) % 100 == 0:
            print(f"   [{i+1}/{len(all_data)}] analyzed...")

    results.sort(key=lambda x: x["score"], reverse=True)
    qualified = [r for r in results if r["score"] >= min_score]

    print(f"   ✅ {len(results)} stocks analyzed, {len(qualified)} qualified (score >= {min_score})")
    if skipped:
        print(f"   ⏭️ {len(skipped)} skipped (insufficient data)")
    if errors:
        print(f"   ⚠️ {errors} errors")

    # Step 6: Save stock analysis to DB
    print(f"\n💾 Step 5: Saving analysis to database...")
    save_stock_analysis(run_id, results)
    print(f"   Saved {len(results)} stock records")

    # Step 7: Run AI analysis
    ai_buy = 0
    ai_hold = 0
    ai_avoid = 0
    ai_completed = 0

    if not skip_ai:
        print(f"\n🤖 Step 6: Running AI analysis on {len(results)} stocks...")
        print(f"   This will take several minutes for large sets...")

        try:
            ai_result = analyze_with_llm(results, fyers_client=client)

            if "error" in ai_result:
                print(f"   ⚠️ AI analysis error: {ai_result['error']}")
            else:
                recs = ai_result.get("recommendations", {})
                if recs:
                    save_ai_recommendations(run_id, recs)
                    ai_buy = sum(1 for r in recs.values() if r.get('action') == 'BUY')
                    ai_hold = sum(1 for r in recs.values() if r.get('action') == 'HOLD')
                    ai_avoid = sum(1 for r in recs.values() if r.get('action') == 'AVOID')
                    ai_completed = 1
                    print(f"   ✅ AI done: {len(recs)} recommendations")
                    print(f"      🟢 BUY: {ai_buy}  🟡 HOLD: {ai_hold}  🔴 AVOID: {ai_avoid}")

                    if ai_result.get('warnings'):
                        print(f"      ⚠️ {len(ai_result['warnings'])} batch warnings")
        except Exception as e:
            print(f"   ❌ AI analysis failed: {e}")
    else:
        print(f"\n⏭️ Step 6: Skipping AI analysis (--skip-ai)")

    # Step 8: Update run status
    update_run_status(
        run_id, "completed",
        total_stocks=len(results),
        qualified_count=len(qualified),
        ai_completed=ai_completed,
        ai_buy=ai_buy,
        ai_hold=ai_hold,
        ai_avoid=ai_avoid
    )

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"  ✅ Daily Analysis Complete!")
    print(f"  Run ID: {run_id}")
    print(f"  Date: {analysis_date}")
    print(f"  Stocks: {len(results)} analyzed, {len(qualified)} qualified")
    print(f"  AI: {'✅ ' + str(ai_buy) + ' BUY / ' + str(ai_hold) + ' HOLD / ' + str(ai_avoid) + ' AVOID' if ai_completed else '⏭️ Skipped'}")
    print(f"  Time: {minutes}m {seconds}s")
    print(f"  DB: {os.path.basename(os.path.join(BASE_DIR, 'analysis_data.db'))}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Daily NSE Stock Analysis Runner")
    parser.add_argument("--date", "-d", type=str,
                        default=datetime.now().strftime("%Y-%m-%d"),
                        help="Analysis date (YYYY-MM-DD). Default: today")
    parser.add_argument("--category", "-c", type=str, default="all",
                        choices=["nifty50", "nifty100", "nifty200", "nifty500", "all"],
                        help="Stock universe. Default: all")
    parser.add_argument("--min-score", "-m", type=int, default=2,
                        help="Minimum score threshold. Default: 2")
    parser.add_argument("--skip-ai", action="store_true",
                        help="Skip AI/LLM analysis (indicators only)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Overwrite existing analysis without asking")

    args = parser.parse_args()

    # Validate date
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date format: {args.date}. Use YYYY-MM-DD")
        sys.exit(1)

    run_daily_analysis(
        analysis_date=args.date,
        category=args.category,
        min_score=args.min_score,
        skip_ai=args.skip_ai
    )


if __name__ == "__main__":
    main()

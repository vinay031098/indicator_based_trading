#!/usr/bin/env python3
"""
Auto-Trading Script - Analyzes NIFTY 50 via Fyers data and executes trades
Run: python auto_trade.py
"""

import sys
sys.path.insert(0, '/Users/vinay/Library/CloudStorage/GoogleDrive-vinay904412@gmail.com/My Drive/indicators_trade')

from main import NiftyAnalyzer
from flyers_integration import FyersClient

APP_ID = "HTEDSURO6P-100"
SECRET_ID = "6E0U40KRQT"


def main():
    print("=" * 80)
    print("NIFTY 50 AUTO-TRADER (FYERS)")
    print("=" * 80)

    # 1. Authenticate with Fyers
    print("\n1. CONNECTING TO FYERS...")
    client = FyersClient(APP_ID, SECRET_ID)
    if not client.authenticate():
        print("   ✗ Authentication failed. Exiting.")
        return

    print("   ✓ Connected!")

    # 2. Analyze stocks using Fyers data
    print("\n2. ANALYZING NIFTY 50 (Fyers data)...")
    analyzer = NiftyAnalyzer(client)
    results = analyzer.analyze_nifty50()

    # 3. Get top stocks
    top_stocks = [s for s in results if s['score'] >= 4][:10]

    if not top_stocks:
        print("No stocks with score >= 4 found.")
        return

    print(f"\n3. TOP QUALIFIED STOCKS ({len(top_stocks)}):")
    for stock in top_stocks[:5]:
        print(f"   • {stock['symbol']}: Score {stock['score']}, Price ₹{stock['price']:.2f}")

    # 4. Ask for confirmation & execute
    confirm = input("\n4. EXECUTE TRADES? (Enter number of stocks to trade, 0 to skip): ").strip()

    if confirm.isdigit() and int(confirm) > 0:
        num_trades = min(int(confirm), len(top_stocks))
        executed = client.execute_trades(top_stocks[:num_trades], qty_per_stock=1)
        successful = sum(1 for t in executed if t['success'])
        print(f"\n✓ {successful}/{num_trades} orders placed successfully")
    else:
        print("   Skipped trading.")


if __name__ == "__main__":
    main()

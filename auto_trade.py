#!/usr/bin/env python3
"""
Auto-Trading Script — analyze NIFTY 50 via Fyers data and (optionally) execute trades.
Run: python auto_trade.py
"""

from __future__ import annotations

import logging

from config import configure_logging
from fyers_integration import provider
from main import NiftyAnalyzer

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging()

    client = provider.get_client()
    if client is None:
        client = provider.build_auth_client()
        if not client.authenticate():
            logger.error("Authentication failed. Exiting.")
            return 1

    analyzer = NiftyAnalyzer(client)
    results = analyzer.analyze_nifty50(min_score=4)
    top_stocks = results[:10]

    if not top_stocks:
        logger.info("No stocks with score >= 4 found.")
        return 0

    print(f"\nTop qualified stocks ({len(top_stocks)}):")
    for stock in top_stocks[:5]:
        print(f"  - {stock['name']}: score {stock['score']}, price Rs{stock['price']:.2f}")

    confirm = input("\nExecute trades? (number of stocks to trade, 0 to skip): ").strip()
    if confirm.isdigit() and int(confirm) > 0:
        num = min(int(confirm), len(top_stocks))
        executed = client.execute_trades(top_stocks[:num], qty_per_stock=1)
        ok = sum(1 for t in executed if t["success"])
        print(f"\n{ok}/{num} orders placed successfully.")
    else:
        print("Skipped trading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

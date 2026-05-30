"""
NIFTY 50 Trading Indicators Analyzer — CLI (Fyers data source).

Thin CLI wrapper around the shared indicator engine (indicators.analyze_stock)
and the Fyers client provider. No duplicated indicator math.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List

from config import configure_logging
from fyers_integration import FyersClient, provider
from indicators import analyze_stock

logger = logging.getLogger(__name__)


class NiftyAnalyzer:
    """Kept for backward compatibility (auto_trade.py imports this)."""

    def __init__(self, fyers_client: FyersClient):
        self.client = fyers_client

    def fetch_all_data(self) -> Dict:
        return self.client.get_all_nifty50_data(resolution="D", days=365)

    def analyze_stock(self, symbol: str, data) -> Dict:
        return analyze_stock(symbol, data)

    def analyze_nifty50(self, min_score: int = 2) -> List[Dict]:
        all_data = self.fetch_all_data()
        results = [r for r in (analyze_stock(s, d) for s, d in all_data.items()) if r]
        results = [r for r in results if r["score"] >= min_score]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


def main() -> int:
    configure_logging()
    client = provider.get_client()
    if client is None:
        client = provider.build_auth_client()
        if not client.authenticate():
            logger.error("Cannot proceed without Fyers authentication.")
            return 1

    analyzer = NiftyAnalyzer(client)
    results = analyzer.analyze_nifty50()

    print("\n" + "=" * 80)
    print(f"NIFTY 50 ANALYSIS (FYERS DATA) — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)
    print(f"\nQualified stocks: {len(results)}\n")
    for s in results:
        print(f"{s['name']:18} | Rs {s['price']:10.2f} | Score {s['score']:>3} | "
              f"Signal {s.get('signal','?'):7} | RSI {s['rsi']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

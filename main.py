"""
NIFTY 50 Trading Indicators Analyzer — FYERS DATA SOURCE
Fetches all historical data from Fyers API, then analyzes with technical indicators.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from flyers_integration import FyersClient, NIFTY_50_FYERS


class NiftyAnalyzer:
    def __init__(self, fyers_client: FyersClient):
        self.client = fyers_client

    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch 1-year daily data for all NIFTY 50 stocks from Fyers."""
        return self.client.get_all_nifty50_data(resolution="D", days=365)

    def _ema(self, data, period):
        arr = np.asarray(data, dtype=float).flatten()
        if len(arr) < period:
            return arr
        ema = np.zeros(len(arr))
        ema[0] = np.mean(arr[:period])
        k = 2.0 / (period + 1)
        for i in range(1, len(arr)):
            ema[i] = arr[i] * k + ema[i-1] * (1 - k)
        return ema

    def analyze_stock(self, symbol: str, data: pd.DataFrame) -> Dict:
        close = data['Close'].values.flatten()
        high = data['High'].values.flatten()
        low = data['Low'].values.flatten()
        try:
            price = float(close[-1])
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            rs = np.mean(gain[-14:]) / np.mean(loss[-14:]) if np.mean(loss[-14:]) != 0 else 0
            rsi = float(100 - (100 / (1 + rs)))
            sma20 = float(np.mean(close[-20:]))
            sma50 = float(np.mean(close[-50:]))
            sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else float(np.mean(close))
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal_line = self._ema(macd_line, 9)
            macd = float(macd_line[-1])
            macd_sig = float(signal_line[-1])
            min14 = float(np.min(low[-14:]))
            max14 = float(np.max(high[-14:]))
            stoch_k = ((price - min14) / (max14 - min14) * 100) if max14 != min14 else 50
            w52_high = float(np.max(high[-252:]))
            w52_low = float(np.min(low[-252:]))
            dist52 = ((w52_high - price) / w52_high) * 100

            score = 0
            reasons = []
            if rsi < 30: score += 2; reasons.append(f"RSI Oversold: {rsi:.1f}")
            elif rsi < 40: score += 1; reasons.append(f"RSI Low: {rsi:.1f}")
            if macd > macd_sig: score += 1; reasons.append("MACD Bullish")
            if price < sma20: score += 1; reasons.append("Below 20-SMA")
            if stoch_k < 20: score += 1; reasons.append(f"Stoch Oversold: {stoch_k:.1f}")
            if dist52 < 5: score += 2; reasons.append(f"Near 52W High ({dist52:.1f}%)")
            elif dist52 < 10: score += 1; reasons.append(f"Close to 52W High ({dist52:.1f}%)")
            if price > sma20 > sma50 > sma200: score += 2; reasons.append("Golden Cross (20>50>200)")

            return {'symbol': symbol, 'price': price, 'score': score, 'rsi': rsi, '52w_dist': dist52, 'reasons': reasons}
        except:
            return None

    def analyze_nifty50(self):
        all_data = self.fetch_all_data()
        results = []
        for symbol, data in all_data.items():
            r = self.analyze_stock(symbol, data)
            if r and r['score'] >= 2:
                results.append(r)
        results.sort(key=lambda x: x['score'], reverse=True)

        print("\n" + "="*80)
        print(f"NIFTY 50 ANALYSIS (FYERS DATA) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"\nQualified Stocks: {len(results)}\n")
        for s in results:
            print(f"{s['symbol']:22} | Rs {s['price']:10.2f} | Score: {s['score']} | RSI: {s['rsi']:.1f} | 52W: {s['52w_dist']:.1f}%")
            for r in s['reasons']:
                print(f"  - {r}")
            print()
        return results


if __name__ == "__main__":
    APP_ID = "HTEDSURO6P-100"
    SECRET_ID = "6E0U40KRQT"

    print("=" * 80)
    print("NIFTY 50 INDICATOR-BASED ANALYZER — FYERS DATA SOURCE")
    print("=" * 80)

    client = FyersClient(APP_ID, SECRET_ID)

    if client.authenticate():
        analyzer = NiftyAnalyzer(client)
        results = analyzer.analyze_nifty50()
    else:
        print("\n✗ Cannot proceed without Fyers authentication.")

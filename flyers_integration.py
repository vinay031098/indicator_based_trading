"""
Fyers Trading API Integration (Official SDK v3)
Handles authentication, historical data, and order execution
Uses: fyers-apiv3 (official Fyers Python SDK)
Docs: https://myapi.fyers.in/docsv3
"""

import webbrowser
import time
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fyers_apiv3 import fyersModel


# ─── NIFTY 50 symbols in Fyers format (NSE:SYMBOL-EQ) ───────────────────
NIFTY_50_FYERS = [
    'NSE:RELIANCE-EQ', 'NSE:TCS-EQ', 'NSE:INFY-EQ', 'NSE:HINDUNILVR-EQ', 'NSE:ICICIBANK-EQ',
    'NSE:HDFCBANK-EQ', 'NSE:WIPRO-EQ', 'NSE:KOTAKBANK-EQ', 'NSE:SBIN-EQ', 'NSE:MARUTI-EQ',
    'NSE:BAJFINANCE-EQ', 'NSE:LT-EQ', 'NSE:AXISBANK-EQ', 'NSE:ASIANPAINT-EQ', 'NSE:SUNPHARMA-EQ',
    'NSE:NESTLEIND-EQ', 'NSE:POWERGRID-EQ', 'NSE:TECHM-EQ', 'NSE:BAJAJ-AUTO-EQ', 'NSE:JSWSTEEL-EQ',
    'NSE:NTPC-EQ', 'NSE:COALINDIA-EQ', 'NSE:APOLLOHOSP-EQ', 'NSE:ULTRACEMCO-EQ', 'NSE:HCLTECH-EQ',
    'NSE:INDUSINDBK-EQ', 'NSE:BHARTIARTL-EQ', 'NSE:EICHERMOT-EQ',
    'NSE:ADANIPORTS-EQ', 'NSE:GRASIM-EQ', 'NSE:SHRIRAMFIN-EQ', 'NSE:TATAMOTORS-EQ',
    'NSE:HEROMOTOCO-EQ', 'NSE:HAVELLS-EQ', 'NSE:DIVISLAB-EQ',
    'NSE:LUPIN-EQ', 'NSE:DRREDDY-EQ', 'NSE:VEDL-EQ', 'NSE:TITAN-EQ',
    'NSE:BEL-EQ', 'NSE:GODREJCP-EQ', 'NSE:TATACONSUM-EQ',
    'NSE:ONGC-EQ', 'NSE:M&M-EQ', 'NSE:CIPLA-EQ', 'NSE:BRITANNIA-EQ', 'NSE:HINDALCO-EQ', 'NSE:TATASTEEL-EQ'
]

# ─── NIFTY 100 additional symbols (beyond NIFTY 50) ─────────────────────
NIFTY_100_EXTRA = [
    'NSE:ADANIENT-EQ', 'NSE:ADANIGREEN-EQ', 'NSE:AMBUJACEM-EQ', 'NSE:ATGL-EQ',
    'NSE:BAJAJHLDNG-EQ', 'NSE:BANKBARODA-EQ', 'NSE:BERGEPAINT-EQ', 'NSE:BOSCHLTD-EQ',
    'NSE:CANBK-EQ', 'NSE:CHOLAFIN-EQ', 'NSE:COLPAL-EQ', 'NSE:DLF-EQ',
    'NSE:DABUR-EQ', 'NSE:DMART-EQ', 'NSE:GAIL-EQ', 'NSE:GODREJPROP-EQ',
    'NSE:HAL-EQ', 'NSE:HDFCLIFE-EQ', 'NSE:ICICIGI-EQ', 'NSE:ICICIPRULI-EQ',
    'NSE:INDIGO-EQ', 'NSE:IOC-EQ', 'NSE:IRCTC-EQ', 'NSE:JIOFIN-EQ',
    'NSE:JSWENERGY-EQ', 'NSE:LICI-EQ', 'NSE:LODHA-EQ', 'NSE:MARICO-EQ',
    'NSE:NHPC-EQ', 'NSE:PIDILITIND-EQ', 'NSE:PFC-EQ', 'NSE:PNB-EQ',
    'NSE:RECLTD-EQ', 'NSE:SBICARD-EQ', 'NSE:SBILIFE-EQ', 'NSE:SIEMENS-EQ',
    'NSE:SRF-EQ', 'NSE:TATAPOWER-EQ', 'NSE:TORNTPHARM-EQ', 'NSE:TRENT-EQ',
    'NSE:UNIONBANK-EQ', 'NSE:VBL-EQ', 'NSE:ZOMATO-EQ', 'NSE:ZYDUSLIFE-EQ',
    'NSE:ABB-EQ', 'NSE:MANKIND-EQ', 'NSE:MAXHEALTH-EQ', 'NSE:PAYTM-EQ',
    'NSE:TVSMOTOR-EQ', 'NSE:INDIANB-EQ'
]

# ─── Combined index lists ───────────────────────────────────────────────
NIFTY_100_FYERS = NIFTY_50_FYERS + NIFTY_100_EXTRA

# ─── Symbol master cache ────────────────────────────────────────────────
_all_nse_symbols_cache = None
_cache_timestamp = None
SYMBOL_MASTER_URL = "https://public.fyers.in/sym_details/NSE_CM.csv"
CACHE_DURATION = 86400  # 24 hours


def fetch_all_nse_equity_symbols() -> List[str]:
    """
    Download Fyers symbol master CSV and extract all NSE equity symbols.
    Only includes segment=0 (equities), suffix=-EQ (not ETFs, bonds, etc.)
    Caches results for 24 hours.
    Returns: list of symbols like ['NSE:RELIANCE-EQ', 'NSE:TCS-EQ', ...]
    """
    global _all_nse_symbols_cache, _cache_timestamp

    now = time.time()
    if _all_nse_symbols_cache and _cache_timestamp and (now - _cache_timestamp) < CACHE_DURATION:
        return _all_nse_symbols_cache

    try:
        print("📥 Downloading NSE symbol master from Fyers...")
        response = urllib.request.urlopen(SYMBOL_MASTER_URL, timeout=30)
        data = response.read().decode('utf-8')
        lines = data.strip().split('\n')

        eq_stocks = []
        for line in lines:
            parts = line.split(',')
            # Find NSE:XXX-EQ symbol in the row
            symbol = None
            for part in parts:
                part = part.strip()
                if part.startswith('NSE:') and part.endswith('-EQ'):
                    symbol = part
                    break
            if not symbol:
                continue

            # segment: 0 = equity stocks, 9 = ETFs
            try:
                segment = int(parts[2].strip())
            except (ValueError, IndexError):
                continue

            if segment == 0:
                eq_stocks.append(symbol)

        eq_stocks.sort()
        _all_nse_symbols_cache = eq_stocks
        _cache_timestamp = now
        print(f"✅ Found {len(eq_stocks)} NSE equity stocks")
        return eq_stocks

    except Exception as e:
        print(f"⚠️  Failed to fetch symbol master: {e}")
        if _all_nse_symbols_cache:
            print("   Using cached symbols")
            return _all_nse_symbols_cache
        # Fallback to NIFTY 50
        print("   Falling back to NIFTY 50 list")
        return NIFTY_50_FYERS


def get_symbols_for_category(category: str) -> List[str]:
    """
    Get stock symbols for a given category.
    Categories: 'nifty50', 'nifty100', 'nifty200', 'nifty500', 'all'
    """
    if category == 'nifty50':
        return NIFTY_50_FYERS
    elif category == 'nifty100':
        return NIFTY_100_FYERS
    elif category == 'nifty200':
        # Top 200 from all NSE — use first 200 from sorted list
        all_syms = fetch_all_nse_equity_symbols()
        # Prioritize NIFTY 100 + next most liquid
        return NIFTY_100_FYERS + [s for s in all_syms if s not in NIFTY_100_FYERS][:100]
    elif category == 'nifty500':
        all_syms = fetch_all_nse_equity_symbols()
        return NIFTY_100_FYERS + [s for s in all_syms if s not in NIFTY_100_FYERS][:400]
    elif category == 'all':
        return fetch_all_nse_equity_symbols()
    else:
        return NIFTY_50_FYERS


class FyersClient:
    """
    Fyers API Client using official fyers-apiv3 SDK.

    Auth flow (from docs):
      1. SessionModel.generate_authcode() → opens browser login URL
      2. User logs in → redirected to redirect_uri with ?auth_code=xxx
      3. SessionModel.set_token(auth_code) → SessionModel.generate_token() → access_token
      4. FyersModel(client_id, token=access_token) → ready for API calls

    Handles: Authentication, Historical Data, Quotes, Order Execution.
    """

    def __init__(self, app_id: str, secret_id: str, redirect_uri: str = "https://fyersapiapp.com"):
        """
        Args:
            app_id: Your Fyers App ID (e.g. "HTEDSURO6P-100")
            secret_id: Your Fyers App Secret (e.g. "6E0U40KRQT")
            redirect_uri: MUST match what you set in Fyers App Dashboard
                          (https://myapi.fyers.in/dashboard)
        """
        self.app_id = app_id
        self.secret_id = secret_id
        self.redirect_uri = redirect_uri
        self.access_token = None
        self.fyers = None  # FyersModel instance for API calls

    # ─── Authentication (using official SDK SessionModel) ────────────────

    def authenticate(self, auth_code: str = None) -> bool:
        """
        Complete authentication flow using official SDK:

        Step 1: Generate auth URL → open browser → user logs in
        Step 2: User pastes auth_code from redirect URL
        Step 3: SDK exchanges auth_code for access_token (handles appIdHash internally)
        Step 4: Initialize FyersModel for subsequent API calls
        """
        try:
            if not auth_code:
                # Step 1: Create session & generate auth URL
                session = fyersModel.SessionModel(
                    client_id=self.app_id,
                    redirect_uri=self.redirect_uri,
                    response_type="code",
                    grant_type="authorization_code",
                    state="indicators_trade",
                    scope="",
                    nonce=""
                )
                auth_url = session.generate_authcode()

                print("\n" + "=" * 80)
                print("FYERS AUTHENTICATION REQUIRED")
                print("=" * 80)
                print(f"\nOpening browser for login...\n")
                print(f"URL: {auth_url}\n")
                webbrowser.open(auth_url, new=1)
                print("After login, you will be redirected to your redirect URI.")
                print("Copy the 'auth_code' parameter from the URL bar.\n")
                print("Example: https://trade.fyers.in/...?s=ok&code=200&auth_code=eyJ0eX...")
                print("         Copy everything after 'auth_code=' until '&' or end\n")
                auth_code = input("Paste the auth_code here: ").strip()

            # Step 2: Create new session to exchange auth_code for access_token
            session = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.secret_id,
                grant_type="authorization_code"
            )
            session.set_token(auth_code)

            # Step 3: Generate access token (SDK handles appIdHash internally)
            response = session.generate_token()

            if response.get("s") == "ok" and response.get("access_token"):
                self.access_token = response["access_token"]
                self._init_fyers_model()
                print(f"\n✓ Authenticated successfully!")
                return True
            else:
                print(f"\n✗ Auth failed: {response.get('message', response)}")
                return False

        except Exception as e:
            print(f"\n✗ Auth error: {e}")
            return False

    def set_access_token(self, token: str):
        """Manually set access token (if you already have one saved)."""
        self.access_token = token
        self._init_fyers_model()

    def _init_fyers_model(self):
        """Initialize FyersModel with current access_token for API calls."""
        self.fyers = fyersModel.FyersModel(
            client_id=self.app_id,
            token=self.access_token,
            is_async=False,
            log_path=""
        )

    # ─── Historical Data ────────────────────────────────────────────────
    # API: GET https://api-t1.fyers.in/data/history
    # Response candles: [epoch, open, high, low, close, volume]

    def get_history(self, symbol: str, resolution: str = "D",
                    days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data for a symbol.

        Args:
            symbol: Fyers symbol e.g. "NSE:SBIN-EQ"
            resolution: "1"/"5"/"15"/"30"/"60"/"120"/"240"/"D" (D=daily)
            days: Number of days of history

        Returns:
            DataFrame with columns: [Open, High, Low, Close, Volume], index=Date
        """
        if not self.fyers:
            print("Not authenticated. Call authenticate() first.")
            return None

        range_to = datetime.now()
        range_from = range_to - timedelta(days=days)

        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",       # 1 = "yyyy-mm-dd" format
            "range_from": range_from.strftime("%Y-%m-%d"),
            "range_to": range_to.strftime("%Y-%m-%d"),
            "cont_flag": "1",         # continuous data
        }

        try:
            response = self.fyers.history(data=data)

            if response.get("s") != "ok":
                print(f"  ✗ History error for {symbol}: {response.get('message', 'Unknown')}")
                return None

            candles = response.get("candles", [])
            if not candles:
                print(f"  ✗ No candle data for {symbol}")
                return None

            df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], unit="s")
            df.set_index("Date", inplace=True)
            return df

        except Exception as e:
            print(f"  ✗ Error fetching {symbol}: {e}")
            return None

    def get_all_nifty50_data(self, resolution: str = "D", days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all NIFTY 50 stocks.
        Returns dict: {symbol: DataFrame}
        Rate limit: 10/sec, 200/min — we add small delays.
        """
        return self.get_stock_data(NIFTY_50_FYERS, resolution=resolution, days=days)

    def get_stock_data(self, symbols: List[str], resolution: str = "D", days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for a list of stock symbols.
        Returns dict: {symbol: DataFrame}
        Rate limit: 10/sec, 200/min — adaptive delays + retry with backoff.
        """
        stock_data = {}
        total = len(symbols)
        failed_symbols = []  # track for retry
        requests_this_minute = 0
        minute_start = time.time()

        print(f"\n📊 Downloading {total} stocks from Fyers...")

        def _fetch_one(symbol: str, attempt: int = 1) -> Optional[pd.DataFrame]:
            """Fetch a single symbol with retry on failure."""
            nonlocal requests_this_minute, minute_start

            # Per-minute rate limit: hard cap at 180/min (buffer under 200)
            requests_this_minute += 1
            elapsed = time.time() - minute_start
            if requests_this_minute >= 180 and elapsed < 60:
                wait = 60 - elapsed + 1
                print(f"  ⏸️  Rate limit pause ({wait:.0f}s) — {requests_this_minute} reqs in {elapsed:.0f}s")
                time.sleep(wait)
                requests_this_minute = 0
                minute_start = time.time()
            elif elapsed >= 60:
                # Reset counter each minute
                requests_this_minute = 1
                minute_start = time.time()

            df = self.get_history(symbol, resolution=resolution, days=days)
            return df

        # Pass 1: fetch all symbols
        for i, symbol in enumerate(symbols, 1):
            df = _fetch_one(symbol)
            if df is not None and len(df) > 50:
                stock_data[symbol] = df
                if i % 100 == 0 or i == total or total <= 100:
                    print(f"  [{i}/{total}] ✓ {symbol} ({len(df)} candles) — {len(stock_data)} ok")
            else:
                failed_symbols.append(symbol)
                if total <= 200:
                    print(f"  [{i}/{total}] ✗ {symbol} (will retry)")

            # Per-request delay: ~8 req/sec
            if i % 8 == 0:
                time.sleep(1.1)
            else:
                time.sleep(0.13)

        # Pass 2: retry failed symbols (rate limit / transient errors)
        if failed_symbols:
            print(f"\n🔄 Retrying {len(failed_symbols)} failed symbols...")
            time.sleep(2)  # cool-down before retry
            requests_this_minute = 0
            minute_start = time.time()
            still_failed = []

            for i, symbol in enumerate(failed_symbols, 1):
                df = _fetch_one(symbol, attempt=2)
                if df is not None and len(df) > 50:
                    stock_data[symbol] = df
                    print(f"  [retry {i}/{len(failed_symbols)}] ✓ {symbol} recovered")
                else:
                    still_failed.append(symbol)

                # Slower on retry — 5 req/sec
                if i % 5 == 0:
                    time.sleep(1.2)
                else:
                    time.sleep(0.2)

            if still_failed and len(still_failed) <= 50:
                print(f"  ⚠️  {len(still_failed)} stocks still failed: {', '.join(s.split(':')[1].split('-')[0] for s in still_failed[:20])}")
            elif still_failed:
                print(f"  ⚠️  {len(still_failed)} stocks still failed after retry")

        print(f"\n✅ Got data for {len(stock_data)}/{total} stocks")
        return stock_data

    # ─── Quotes (Real-time) ─────────────────────────────────────────────
    # API: GET https://api-t1.fyers.in/data/quotes?symbols=NSE:SBIN-EQ

    def get_quotes(self, symbols: List[str]) -> Optional[Dict]:
        """Get real-time quotes for list of symbols."""
        if not self.fyers:
            return None
        data = {"symbols": ",".join(symbols)}
        try:
            return self.fyers.quotes(data=data)
        except Exception as e:
            print(f"Quotes error: {e}")
            return None

    # ─── Market Depth ───────────────────────────────────────────────────
    # API: GET https://api-t1.fyers.in/data/depth?symbol=NSE:SBIN-EQ

    def get_depth(self, symbol: str) -> Optional[Dict]:
        """Get market depth (bid/ask) for a symbol."""
        if not self.fyers:
            return None
        data = {"symbol": symbol, "ohlcv_flag": "1"}
        try:
            return self.fyers.depth(data=data)
        except Exception as e:
            print(f"Depth error: {e}")
            return None

    # ─── User Info ──────────────────────────────────────────────────────

    def get_profile(self) -> Optional[Dict]:
        """Get user profile. API: GET https://api-t1.fyers.in/api/v3/profile"""
        if not self.fyers:
            return None
        try:
            return self.fyers.get_profile()
        except Exception as e:
            print(f"Profile error: {e}")
            return None

    def get_funds(self) -> Optional[Dict]:
        """Get account funds/balance. API: GET https://api-t1.fyers.in/api/v3/funds"""
        if not self.fyers:
            return None
        try:
            return self.fyers.funds()
        except Exception as e:
            print(f"Funds error: {e}")
            return None

    def get_holdings(self) -> Optional[Dict]:
        """Get current holdings. API: GET https://api-t1.fyers.in/api/v3/holdings"""
        if not self.fyers:
            return None
        try:
            return self.fyers.holdings()
        except Exception as e:
            print(f"Holdings error: {e}")
            return None

    def get_positions(self) -> Optional[Dict]:
        """Get current positions. API: GET https://api-t1.fyers.in/api/v3/positions"""
        if not self.fyers:
            return None
        try:
            return self.fyers.positions()
        except Exception as e:
            print(f"Positions error: {e}")
            return None

    def get_orderbook(self) -> Optional[Dict]:
        """Get order book. API: GET https://api-t1.fyers.in/api/v3/orders"""
        if not self.fyers:
            return None
        try:
            return self.fyers.orderbook()
        except Exception as e:
            print(f"Orderbook error: {e}")
            return None

    def get_tradebook(self) -> Optional[Dict]:
        """Get trade book. API: GET https://api-t1.fyers.in/api/v3/tradebook"""
        if not self.fyers:
            return None
        try:
            return self.fyers.tradebook()
        except Exception as e:
            print(f"Tradebook error: {e}")
            return None

    # ─── Market Status ──────────────────────────────────────────────────

    def get_market_status(self) -> Optional[Dict]:
        """Get market status for all exchanges."""
        if not self.fyers:
            return None
        try:
            return self.fyers.market_status()
        except Exception as e:
            print(f"Market status error: {e}")
            return None

    # ─── Order Execution ────────────────────────────────────────────────
    # API: POST https://api-t1.fyers.in/api/v3/orders/sync

    def place_order(self, symbol: str, qty: int, side: int = 1,
                    order_type: int = 2, product_type: str = "INTRADAY",
                    limit_price: float = 0, stop_price: float = 0) -> Dict:
        """
        Place an order via Fyers.

        Args:
            symbol: Fyers symbol e.g. "NSE:SBIN-EQ"
            qty: Quantity
            side: 1=Buy, -1=Sell
            order_type: 1=Limit, 2=Market, 3=Stop(SL-M), 4=StopLimit(SL-L)
            product_type: "INTRADAY" / "CNC" / "MARGIN" / "BO" / "CO" / "MTF"
            limit_price: Price for limit orders (0 for market)
            stop_price: Stop loss price (0 for none)

        Returns:
            dict with order result
        """
        if not self.fyers:
            return {"success": False, "message": "Not authenticated"}

        data = {
            "symbol": symbol,
            "qty": qty,
            "type": order_type,
            "side": side,
            "productType": product_type,
            "limitPrice": limit_price,
            "stopPrice": stop_price,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0,
            "orderTag": "indicators_trade"
        }

        try:
            response = self.fyers.place_order(data=data)
            if response.get("s") == "ok":
                return {
                    "success": True,
                    "order_id": response.get("id"),
                    "symbol": symbol,
                    "qty": qty,
                    "message": response.get("message", f"Order placed: {symbol}")
                }
            else:
                return {
                    "success": False,
                    "symbol": symbol,
                    "message": f"Order failed: {response.get('message', 'Unknown')}"
                }
        except Exception as e:
            return {"success": False, "symbol": symbol, "message": f"Error: {e}"}

    def modify_order(self, order_id: str, qty: int = None,
                     order_type: int = None, limit_price: float = None) -> Dict:
        """Modify an existing order."""
        if not self.fyers:
            return {"success": False, "message": "Not authenticated"}

        data = {"id": order_id}
        if qty is not None:
            data["qty"] = qty
        if order_type is not None:
            data["type"] = order_type
        if limit_price is not None:
            data["limitPrice"] = limit_price

        try:
            response = self.fyers.modify_order(data=data)
            return {"success": response.get("s") == "ok", "message": response.get("message", "")}
        except Exception as e:
            return {"success": False, "message": f"Error: {e}"}

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel a pending order."""
        if not self.fyers:
            return {"success": False, "message": "Not authenticated"}

        try:
            response = self.fyers.cancel_order(data={"id": order_id})
            return {"success": response.get("s") == "ok", "message": response.get("message", "")}
        except Exception as e:
            return {"success": False, "message": f"Error: {e}"}

    def exit_positions(self, position_id: str = None) -> Dict:
        """Exit a specific position or all positions."""
        if not self.fyers:
            return {"success": False, "message": "Not authenticated"}

        try:
            if position_id:
                response = self.fyers.exit_positions(data={"id": position_id})
            else:
                response = self.fyers.exit_positions(data={"exit_all": 1})
            return {"success": response.get("s") == "ok", "message": response.get("message", "")}
        except Exception as e:
            return {"success": False, "message": f"Error: {e}"}

    def execute_trades(self, qualified_stocks: List[Dict], qty_per_stock: int = 1) -> List[Dict]:
        """Execute market buy orders for top qualified stocks."""
        if not self.fyers:
            print("Not authenticated. Call authenticate() first.")
            return []

        executed = []
        print("\n" + "=" * 80)
        print("EXECUTING TRADES FOR QUALIFIED STOCKS")
        print("=" * 80 + "\n")

        for stock in qualified_stocks[:10]:
            symbol = stock["symbol"]
            price = stock["price"]
            score = stock["score"]

            print(f"Placing order: {symbol} | Price: Rs{price:.2f} | Score: {score}")
            result = self.place_order(symbol, qty_per_stock, side=1, order_type=2)
            executed.append(result)

            if result["success"]:
                print(f"  ✓ Order {result['order_id']}")
            else:
                print(f"  ✗ {result['message']}")

            time.sleep(0.3)  # respect rate limits

        ok = sum(1 for t in executed if t["success"])
        print(f"\n{'=' * 80}")
        print(f"TRADE SUMMARY: {ok}/{len(executed)} successful")
        print(f"{'=' * 80}\n")
        return executed


# ─── Convenience test ────────────────────────────────────────────────────

def test_connection():
    """Interactive test: authenticate and fetch sample data."""
    APP_ID = "HTEDSURO6P-100"
    SECRET_ID = "6E0U40KRQT"

    # IMPORTANT: redirect_uri MUST match your Fyers App Dashboard setting!
    # Check at: https://myapi.fyers.in/dashboard
    REDIRECT_URI = "https://fyersapiapp.com"

    client = FyersClient(APP_ID, SECRET_ID, redirect_uri=REDIRECT_URI)

    if client.authenticate():
        # Test profile
        profile = client.get_profile()
        if profile and profile.get("s") == "ok":
            print(f"\n✓ Connected! User: {profile.get('data', {}).get('name', 'N/A')}")

        # Test market status
        status = client.get_market_status()
        if status:
            print(f"✓ Market status retrieved")

        # Test fetching 1 stock's history
        print("\nFetching SBIN history (30 days)...")
        df = client.get_history("NSE:SBIN-EQ", resolution="D", days=30)
        if df is not None:
            print(df.tail())
            print(f"\n✓ Got {len(df)} candles for SBIN")
    else:
        print("✗ Authentication failed")


if __name__ == "__main__":
    test_connection()

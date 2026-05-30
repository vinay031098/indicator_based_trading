"""
Fyers Trading API Integration (official fyers-apiv3 SDK).

Improvements over the original module:
  - Task 11: FyersClientProvider — no global mutable client; builds clients on demand.
  - Task 28: parallel historical-data fetch with a shared token-bucket rate limiter.
  - Task 29: NSE symbol master cached to disk (24h) so restarts don't re-download.
  - Task 15: logging instead of print.
  - Task 8 : access token loaded/saved via security.py (encrypted at rest).

Docs: https://myapi.fyers.in/docsv3
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from config import settings
from security import load_fyers_token, save_fyers_token

logger = logging.getLogger(__name__)


def _apply_fyers_api_config():
    """Point the SDK trading client at the live production API host."""
    fyersModel = _fyers_model()
    fyersModel.Config.API = settings.fyers_api_base.rstrip("/")
    fyersModel.Config.DATA_API = settings.fyers_data_api_base.rstrip("/")


def _with_fyers_auth_api():
    """Temporarily point the SDK SessionModel at the OAuth host (api-t1).

    ``api.fyers.in`` is for live trading calls only; ``generate-authcode`` and
    ``validate-authcode`` must hit ``api-t1.fyers.in`` or Fyers returns HTTP 500
    "Invalid Request, please provide valid method".
    """
    fyersModel = _fyers_model()
    auth_base = settings.fyers_auth_api_base.rstrip("/")
    trading_base = settings.fyers_api_base.rstrip("/")
    previous = fyersModel.Config.API
    fyersModel.Config.API = auth_base
    return previous, trading_base, fyersModel


def _fyers_model():
    """Lazy import of the Fyers SDK.

    The official `fyers-apiv3` SDK declares the obsolete `asyncio` PyPI backport
    as a dependency, which is unimportable on modern Python. Importing it lazily
    keeps the dashboard's stored-data mode (and the test suite) working even when
    the SDK is unavailable; only live broker calls require it.
    """
    from fyers_apiv3 import fyersModel  # noqa: PLC0415

    return fyersModel


def _fyers_credentials() -> tuple[str, str]:
    """Return (app_id, secret), with deploy defaults when Render env is not synced."""
    app_id = (settings.fyers_app_id or "").strip()
    secret = (settings.fyers_secret_id or "").strip()
    if app_id and secret:
        return app_id, secret
    if not app_id and not secret:
        # Same values as deploy/deploy.sh — used when Render has not synced env vars.
        logger.warning(
            "FYERS_APP_ID / FYERS_SECRET_ID missing — using deploy defaults. "
            "Set them in Render → Environment."
        )
        return "JIHLRUYWGE-100", "DZQQB3O1GS"
    raise ValueError(
        "Incomplete Fyers config: set both FYERS_APP_ID and FYERS_SECRET_ID "
        "(Render → Environment, or .env locally)."
    )


def _require_fyers_credentials() -> None:
    app_id, secret = _fyers_credentials()
    if not app_id or not secret:
        raise ValueError(
            "FYERS_APP_ID and FYERS_SECRET_ID are missing. "
            "Set them in Render → Environment (production) or .env (local), "
            "then restart. Create an app at https://myapi.fyers.in/dashboard"
        )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYMBOL_CACHE_FILE = os.path.join(BASE_DIR, ".nse_symbols_cache.json")
SYMBOL_MASTER_URL = "https://public.fyers.in/sym_details/NSE_CM.csv"
CACHE_DURATION = 86400  # 24 hours

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

NIFTY_100_FYERS = NIFTY_50_FYERS + NIFTY_100_EXTRA

_symbol_lock = threading.Lock()


# ─── Rate limiter (Task 28) ─────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe limiter: at most `per_sec` calls/second and `per_min`/minute."""

    def __init__(self, per_sec: int = 8, per_min: int = 180):
        self._per_sec = per_sec
        self._per_min = per_min
        self._lock = threading.Lock()
        self._sec_window = time.time()
        self._sec_count = 0
        self._min_window = time.time()
        self._min_count = 0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                if now - self._min_window >= 60:
                    self._min_window = now
                    self._min_count = 0
                if now - self._sec_window >= 1:
                    self._sec_window = now
                    self._sec_count = 0

                if self._min_count >= self._per_min:
                    sleep_for = 60 - (now - self._min_window) + 0.05
                elif self._sec_count >= self._per_sec:
                    sleep_for = 1 - (now - self._sec_window) + 0.01
                else:
                    self._sec_count += 1
                    self._min_count += 1
                    return
            time.sleep(max(sleep_for, 0.01))


# ─── Symbol master (Task 29: persistent cache) ──────────────────────────

def _load_symbol_cache() -> Optional[List[str]]:
    if not os.path.exists(SYMBOL_CACHE_FILE):
        return None
    try:
        with open(SYMBOL_CACHE_FILE) as fh:
            blob = json.load(fh)
        if time.time() - blob.get("ts", 0) < CACHE_DURATION and blob.get("symbols"):
            return blob["symbols"]
    except Exception as exc:
        logger.warning("Symbol cache read failed: %s", exc)
    return None


def _write_symbol_cache(symbols: List[str]) -> None:
    try:
        with open(SYMBOL_CACHE_FILE, "w") as fh:
            json.dump({"ts": time.time(), "symbols": symbols}, fh)
    except OSError as exc:
        logger.warning("Symbol cache write failed: %s", exc)


def fetch_all_nse_equity_symbols() -> List[str]:
    """Return all NSE equity symbols, using a 24h disk cache."""
    with _symbol_lock:
        cached = _load_symbol_cache()
        if cached:
            return cached
        try:
            logger.info("Downloading NSE symbol master from Fyers...")
            response = urllib.request.urlopen(SYMBOL_MASTER_URL, timeout=30)
            lines = response.read().decode("utf-8").strip().split("\n")
            eq_stocks = []
            for line in lines:
                parts = line.split(",")
                symbol = next(
                    (p.strip() for p in parts if p.strip().startswith("NSE:") and p.strip().endswith("-EQ")),
                    None,
                )
                if not symbol:
                    continue
                try:
                    if int(parts[2].strip()) == 0:  # 0 = equity
                        eq_stocks.append(symbol)
                except (ValueError, IndexError):
                    continue
            eq_stocks.sort()
            _write_symbol_cache(eq_stocks)
            logger.info("Found %d NSE equity stocks", len(eq_stocks))
            return eq_stocks
        except Exception as exc:
            logger.warning("Failed to fetch symbol master (%s); falling back to NIFTY 50", exc)
            stale = _load_symbol_cache_ignore_age()
            return stale or NIFTY_50_FYERS


def _load_symbol_cache_ignore_age() -> Optional[List[str]]:
    if not os.path.exists(SYMBOL_CACHE_FILE):
        return None
    try:
        with open(SYMBOL_CACHE_FILE) as fh:
            return json.load(fh).get("symbols")
    except Exception:
        return None


def get_symbols_for_category(category: str) -> List[str]:
    if category == "nifty50":
        return NIFTY_50_FYERS
    if category == "nifty100":
        return NIFTY_100_FYERS
    if category == "nifty200":
        all_syms = fetch_all_nse_equity_symbols()
        return NIFTY_100_FYERS + [s for s in all_syms if s not in NIFTY_100_FYERS][:100]
    if category == "nifty500":
        all_syms = fetch_all_nse_equity_symbols()
        return NIFTY_100_FYERS + [s for s in all_syms if s not in NIFTY_100_FYERS][:400]
    if category == "all":
        return fetch_all_nse_equity_symbols()
    return NIFTY_50_FYERS


# ─── Client ─────────────────────────────────────────────────────────────

class FyersClient:
    """Fyers API client wrapping the official fyers-apiv3 SDK."""

    def __init__(self, app_id: str, secret_id: str, redirect_uri: str = "https://fyersapiapp.com"):
        self.app_id = app_id
        self.secret_id = secret_id
        self.redirect_uri = redirect_uri
        self.access_token: Optional[str] = None
        self.fyers = None
        self._rate = _RateLimiter()

    # ─── Auth ────────────────────────────────────────────────────────
    def generate_auth_url(self) -> str:
        """Build the Fyers browser login URL (OAuth host + redirect from settings)."""
        _require_fyers_credentials()
        previous, _, fyersModel = _with_fyers_auth_api()
        try:
            session = fyersModel.SessionModel(
                client_id=self.app_id.strip(),
                redirect_uri=self.redirect_uri.strip(),
                response_type="code",
                grant_type="authorization_code",
                state="indicators_trade",
            )
            return session.generate_authcode()
        finally:
            fyersModel.Config.API = previous

    def authenticate(self, auth_code: Optional[str] = None) -> bool:
        try:
            _require_fyers_credentials()
            if not auth_code:
                auth_url = self.generate_auth_url()
                logger.info("Fyers authentication required. Visit: %s", auth_url)
                auth_code = input("Paste the auth_code here: ").strip()

            previous, _, fyersModel = _with_fyers_auth_api()
            try:
                session = fyersModel.SessionModel(
                    client_id=self.app_id,
                    secret_key=self.secret_id,
                    grant_type="authorization_code",
                )
                session.set_token(auth_code.strip())
                response = session.generate_token()
            finally:
                fyersModel.Config.API = previous

            if response.get("s") == "ok" and response.get("access_token"):
                self.access_token = response["access_token"]
                self._init_model()
                logger.info("Fyers authenticated successfully.")
                return True
            message = response.get("message") or str(response)
            logger.warning("Fyers auth failed: %s", message)
            raise ValueError(message)
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Fyers auth error: %s", exc)
            raise ValueError(str(exc)) from exc

    def set_access_token(self, token: str) -> None:
        self.access_token = token
        self._init_model()

    def _init_model(self) -> None:
        _apply_fyers_api_config()
        fyersModel = _fyers_model()
        self.fyers = fyersModel.FyersModel(
            client_id=self.app_id, token=self.access_token, is_async=False, log_path=""
        )

    # ─── Historical data ─────────────────────────────────────────────
    def get_history(self, symbol: str, resolution: str = "D", days: int = 365) -> Optional[pd.DataFrame]:
        if not self.fyers:
            logger.warning("get_history called before authentication.")
            return None
        range_to = datetime.now()
        range_from = range_to - timedelta(days=days)
        data = {
            "symbol": symbol, "resolution": resolution, "date_format": "1",
            "range_from": range_from.strftime("%Y-%m-%d"),
            "range_to": range_to.strftime("%Y-%m-%d"), "cont_flag": "1",
        }
        try:
            self._rate.acquire()
            response = self.fyers.history(data=data)
            if response.get("s") != "ok":
                return None
            candles = response.get("candles", [])
            if not candles:
                return None
            df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], unit="s")
            df.set_index("Date", inplace=True)
            return df
        except Exception as exc:
            logger.debug("History error for %s: %s", symbol, exc)
            return None

    def get_stock_data(self, symbols: List[str], resolution: str = "D",
                       days: int = 365, max_workers: int = 6,
                       progress_cb=None) -> Dict[str, pd.DataFrame]:
        """Fetch history for many symbols in parallel within the rate limit (Task 28)."""
        stock_data: Dict[str, pd.DataFrame] = {}
        total = len(symbols)
        done = 0
        logger.info("Fetching %d symbols (parallel, max_workers=%d)...", total, max_workers)

        def _one(sym: str):
            return sym, self.get_history(sym, resolution=resolution, days=days)

        failed: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_one, s): s for s in symbols}
            for fut in as_completed(futures):
                sym, df = fut.result()
                if df is not None and len(df) > 50:
                    stock_data[sym] = df
                else:
                    failed.append(sym)
                done += 1
                if progress_cb and (done % 25 == 0 or done == total):
                    progress_cb(done, total, len(stock_data))

        if failed:
            logger.info("Retrying %d failed symbols...", len(failed))
            with ThreadPoolExecutor(max_workers=max(2, max_workers // 2)) as pool:
                futures = {pool.submit(_one, s): s for s in failed}
                for fut in as_completed(futures):
                    sym, df = fut.result()
                    if df is not None and len(df) > 50:
                        stock_data[sym] = df

        logger.info("Got data for %d/%d stocks", len(stock_data), total)
        return stock_data

    def get_all_nifty50_data(self, resolution: str = "D", days: int = 365) -> Dict[str, pd.DataFrame]:
        return self.get_stock_data(NIFTY_50_FYERS, resolution=resolution, days=days)

    # ─── Quotes / account / orders ───────────────────────────────────
    def _safe(self, fn, *args, **kwargs):
        if not self.fyers:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.debug("Fyers call failed: %s", exc)
            return None

    def get_quotes(self, symbols: List[str]) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.quotes(data={"symbols": ",".join(symbols)}))

    def get_depth(self, symbol: str) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.depth(data={"symbol": symbol, "ohlcv_flag": "1"}))

    def get_profile(self) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.get_profile())

    def get_funds(self) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.funds())

    def get_holdings(self) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.holdings())

    def get_positions(self) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.positions())

    def get_orderbook(self) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.orderbook())

    def get_tradebook(self) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.tradebook())

    def get_market_status(self) -> Optional[Dict]:
        return self._safe(lambda: self.fyers.market_status())

    def place_order(self, symbol: str, qty: int, side: int = 1, order_type: int = 2,
                    product_type: str = "INTRADAY", limit_price: float = 0,
                    stop_price: float = 0) -> Dict:
        if not self.fyers:
            return {"success": False, "message": "Not authenticated"}
        data = {
            "symbol": symbol, "qty": qty, "type": order_type, "side": side,
            "productType": product_type, "limitPrice": limit_price, "stopPrice": stop_price,
            "validity": "DAY", "disclosedQty": 0, "offlineOrder": False,
            "stopLoss": 0, "takeProfit": 0, "orderTag": "indicators_trade",
        }
        try:
            response = self.fyers.place_order(data=data)
            if response.get("s") == "ok":
                return {"success": True, "order_id": response.get("id"), "symbol": symbol,
                        "qty": qty, "message": response.get("message", f"Order placed: {symbol}")}
            return {"success": False, "symbol": symbol,
                    "message": f"Order failed: {response.get('message', 'Unknown')}"}
        except Exception as exc:
            return {"success": False, "symbol": symbol, "message": f"Error: {exc}"}

    def modify_order(self, order_id: str, qty: int = None, order_type: int = None,
                     limit_price: float = None) -> Dict:
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
        except Exception as exc:
            return {"success": False, "message": f"Error: {exc}"}

    def cancel_order(self, order_id: str) -> Dict:
        if not self.fyers:
            return {"success": False, "message": "Not authenticated"}
        try:
            response = self.fyers.cancel_order(data={"id": order_id})
            return {"success": response.get("s") == "ok", "message": response.get("message", "")}
        except Exception as exc:
            return {"success": False, "message": f"Error: {exc}"}

    def exit_positions(self, position_id: str = None) -> Dict:
        if not self.fyers:
            return {"success": False, "message": "Not authenticated"}
        try:
            payload = {"id": position_id} if position_id else {"exit_all": 1}
            response = self.fyers.exit_positions(data=payload)
            return {"success": response.get("s") == "ok", "message": response.get("message", "")}
        except Exception as exc:
            return {"success": False, "message": f"Error: {exc}"}

    def execute_trades(self, qualified_stocks: List[Dict], qty_per_stock: int = 1) -> List[Dict]:
        if not self.fyers:
            logger.warning("execute_trades called before authentication.")
            return []
        executed = []
        for stock in qualified_stocks[:10]:
            result = self.place_order(stock["symbol"], qty_per_stock, side=1, order_type=2)
            executed.append(result)
            logger.info("Order %s: %s", stock["symbol"], "ok" if result["success"] else result["message"])
            time.sleep(0.3)
        ok = sum(1 for t in executed if t["success"])
        logger.info("Trade summary: %d/%d successful", ok, len(executed))
        return executed


# ─── Client provider (Task 11) ─────────────────────────────────────────

class FyersClientProvider:
    """Builds a FyersClient from the persisted token on demand (no global state)."""

    def __init__(self):
        self._lock = threading.Lock()

    def build_auth_client(self) -> FyersClient:
        app_id, secret = _fyers_credentials()
        return FyersClient(
            app_id,
            secret,
            settings.effective_fyers_redirect_uri,
        )

    def get_client(self) -> Optional[FyersClient]:
        """Return a ready client if a saved token exists, else None."""
        token = load_fyers_token()
        if not token:
            return None
        client = self.build_auth_client()
        client.set_access_token(token)
        return client

    def connect_with_auth_code(self, auth_code: str) -> bool:
        client = self.build_auth_client()
        client.authenticate(auth_code=auth_code)
        save_fyers_token(client.access_token)
        return True

    def is_connected(self) -> bool:
        return load_fyers_token() is not None


provider = FyersClientProvider()

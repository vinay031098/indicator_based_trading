"""
LLM Stock Analyzer — Uses GitHub Models API (free with GitHub account)
to analyze stock indicators and recommend Buy / Hold / Avoid.

Primary: GitHub Models (gpt-4o-mini via models.inference.ai.azure.com)
Fallback: Google Gemini (gemini-2.5-flash)

No rigid rules — the LLM uses its own market expertise, all 25 indicator values,
and 15-minute candle data (last 3 days) to identify chart patterns and make decisions.

Sends stocks in batches to avoid output truncation.
Includes JSON repair for partial/malformed responses.
"""

import os
import json
import re
import time
import requests
import numpy as np
from datetime import datetime, timedelta

# ─── API Keys ─────────────────────────────────────────────────────
# GitHub PAT with models:read scope — get one at https://github.com/settings/tokens
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
# Gemini fallback
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBIl0FszcAAYd7YAR8MPsmhDGLu8S2mVU0")

# GitHub Models endpoint (OpenAI-compatible)
GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"
GITHUB_MODEL = "gpt-4o-mini"  # free tier, fast, good at JSON

# Gemini fallback
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]

BATCH_SIZE = 5  # smaller batches — prompts are large with 15-min candle data
MAX_RETRIES = 2


# ─── JSON repair helpers ──────────────────────────────────────────

def _repair_json(text):
    """Try to fix common JSON issues from LLM output."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try parsing as-is first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # If the array is truncated (missing closing ]), try to close it
    # Find the last complete object (ends with })
    if text.startswith("[") and not text.rstrip().endswith("]"):
        last_brace = text.rfind("}")
        if last_brace > 0:
            text = text[:last_brace + 1] + "]"

    # If a string value is unterminated, close it
    # Pattern: "key": "value without closing quote
    text = re.sub(r'"([^"]*?)$', r'"\1"', text)

    # Try again
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: extract individual JSON objects with regex
    objects = []
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            obj = json.loads(m.group())
            if "name" in obj and "action" in obj:
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    if objects:
        return objects

    return None


# ─── 15-min candle summarizer ─────────────────────────────────────

def _summarize_intraday(candles_15m):
    """
    Summarize 15-min candle data into a compact text for the LLM.
    Input: list of [epoch, O, H, L, C, V] from Fyers.
    Returns a multi-line string with key patterns.
    """
    if not candles_15m or len(candles_15m) < 5:
        return "No intraday data"

    lines = []
    # Group by date
    by_date = {}
    for c in candles_15m:
        dt = datetime.fromtimestamp(c[0])
        day = dt.strftime("%d-%b")
        if day not in by_date:
            by_date[day] = []
        by_date[day].append(c)

    for day, candles in list(by_date.items())[-3:]:  # last 3 days
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        vols = [c[5] for c in candles]

        day_open = opens[0]
        day_close = closes[-1]
        day_high = max(highs)
        day_low = min(lows)
        day_chg = ((day_close - day_open) / day_open * 100) if day_open > 0 else 0
        total_vol = sum(vols)

        # Detect intraday patterns
        patterns = []

        # Higher highs / lower lows trend
        hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        if hh_count > len(highs) * 0.6:
            patterns.append("higher-highs")
        if ll_count > len(lows) * 0.6:
            patterns.append("lower-lows")

        # Volume spike in last hour
        if len(vols) >= 4:
            avg_vol = np.mean(vols[:-4]) if len(vols) > 4 else np.mean(vols)
            last_hr_vol = np.mean(vols[-4:])
            if avg_vol > 0 and last_hr_vol > avg_vol * 1.5:
                patterns.append("late-vol-spike")

        # Gap up/down from previous day
        if len(lines) > 0 and candles_15m:
            pass  # handled at day level below

        # Doji / hammer detection (last 3 candles)
        for c in candles[-3:]:
            body = abs(c[4] - c[1])
            wick_upper = c[2] - max(c[1], c[4])
            wick_lower = min(c[1], c[4]) - c[3]
            rng = c[2] - c[3]
            if rng > 0 and body / rng < 0.15:
                patterns.append("doji")
                break
            if rng > 0 and wick_lower > body * 2 and wick_upper < body * 0.5:
                patterns.append("hammer")
                break

        # Close near high or low
        if day_high > day_low:
            pos = (day_close - day_low) / (day_high - day_low)
            if pos > 0.85:
                patterns.append("close-near-high")
            elif pos < 0.15:
                patterns.append("close-near-low")

        pat_str = " [" + ",".join(patterns) + "]" if patterns else ""
        lines.append(f"  {day}: O={day_open:.1f} H={day_high:.1f} L={day_low:.1f} C={day_close:.1f} chg={day_chg:+.1f}% vol={total_vol:.0f}{pat_str}")

    return "\n".join(lines)


# ─── Prompt builder (for a batch) ─────────────────────────────────

def _build_batch_prompt(stocks_batch):
    """Build prompt — no rigid rules, let LLM use its own expertise."""

    header = (
        "You are a world-class Indian stock market technical analyst with deep expertise in NSE/BSE.\n"
        "You have access to real-time market knowledge, sector trends, global cues, and institutional flows.\n\n"
        "For each stock below, you are given:\n"
        "1. ALL 25 technical indicator values (daily timeframe)\n"
        "2. 15-minute candle summary for last 3 trading days (with detected chart patterns)\n\n"
        "Use your full expertise to analyze:\n"
        "- Chart patterns (head & shoulders, double bottom/top, flags, wedges, cup & handle, etc.)\n"
        "- Candlestick patterns from 15-min data (doji, hammer, engulfing, morning/evening star)\n"
        "- Support/resistance levels from pivot points, Bollinger Bands, 52W high/low\n"
        "- Trend strength from ADX, moving average alignment, Ichimoku cloud\n"
        "- Momentum from RSI, MACD, Stochastic, ROC, CCI\n"
        "- Volume confirmation from MFI, OBV, CMF, volume ratio\n"
        "- Risk from ATR, distance from 52W extremes\n"
        "- Your knowledge of the company, sector, and current market conditions\n\n"
        "Make your OWN independent decision. Do NOT follow any fixed scoring rules.\n"
        "Think like a professional trader planning trades for the NEXT TRADING DAY.\n\n"
        "═══════════════════════════════════════════\n"
        "STOCKS DATA:\n"
        "═══════════════════════════════════════════\n\n"
    )

    for s in stocks_batch:
        header += (
            f"━━━ {s['name']} ━━━ ₹{s['price']} ({s['change_pct']:+.1f}%) Score={s['score']}/30\n"
            f"  RSI={s['rsi']} MACD={s['macd']}(sig={s['macd_signal']},hist={s['macd_hist']}) "
            f"StochK={s['stoch_k']} StochD={s.get('stoch_d','?')}\n"
            f"  SMA20={s['sma20']} SMA50={s['sma50']} SMA200={s['sma200']} "
            f"EMA9={s.get('ema9','?')} EMA21={s.get('ema21','?')}\n"
            f"  BB={s['bb_lower']}-{s['bb_upper']} BBwidth={s.get('bb_width','?')}%\n"
            f"  52WH={s['w52_high']} 52WL={s['w52_low']} Dist52WH={s.get('dist_52w','?')}%\n"
            f"  VolRatio={s['vol_ratio']}x AvgVol={s.get('avg_volume','?')}\n"
            f"  ATR={s.get('atr','?')} ATR%={s.get('atr_pct','?')} "
            f"ADX={s.get('adx','?')} DI+={s.get('plus_di','?')} DI-={s.get('minus_di','?')}\n"
            f"  CCI={s.get('cci','?')} WillR={s.get('williams_r','?')} MFI={s.get('mfi','?')}\n"
            f"  VWAP={s.get('vwap','?')} ROC={s.get('roc','?')}% CMF={s.get('cmf','?')}\n"
            f"  Ichimoku: Tenkan={s.get('ichimoku_tenkan','?')} Kijun={s.get('ichimoku_kijun','?')}\n"
            f"  Pivot={s.get('pivot','?')} S1={s.get('pivot_s1','?')} R1={s.get('pivot_r1','?')}\n"
        )
        # Add 15-min candle data if available
        intraday = s.get('intraday_summary', '')
        if intraday:
            header += f"  15-min candles (3 days):\n{intraday}\n"
        header += "\n"

    header += (
        "═══════════════════════════════════════════\n"
        "Return ONLY a JSON array. No markdown. No explanation.\n"
        "Each object: {\"name\":\"SYMBOL\",\"action\":\"BUY|HOLD|AVOID\","
        "\"confidence\":\"HIGH|MEDIUM|LOW\","
        "\"reason\":\"your analysis in 1-2 sentences including chart patterns identified\","
        "\"target\":number_or_0,\"stoploss\":number_or_0,\"risk_reward\":\"ratio_or_NA\"}\n"
    )
    return header


# ─── API call: GitHub Models (primary) ────────────────────────────

def _call_github_models(prompt):
    """Call GitHub Models API (OpenAI-compatible) and return parsed JSON list."""
    if not GITHUB_TOKEN:
        return None

    try:
        resp = requests.post(
            GITHUB_MODELS_URL,
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": GITHUB_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a world-class Indian stock market technical analyst. Use all provided indicator data, 15-minute chart patterns, and your market knowledge to make independent trading decisions. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 8192,
                "response_format": {"type": "json_object"}
            },
            timeout=90
        )

        if resp.status_code == 429:
            print(f"  ⚠ GitHub Models rate limited (429), will try Gemini fallback")
            return None
        if resp.status_code != 200:
            print(f"  ⚠ GitHub Models API {resp.status_code}: {resp.text[:200]}")
            return None

        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not text:
            return None

        parsed = _repair_json(text)
        # json_object mode may wrap in {"recommendations": [...]} — extract array
        if isinstance(parsed, dict):
            for key in ("recommendations", "stocks", "data", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            # If it's a single-item dict with a list value, use it
            for v in parsed.values():
                if isinstance(v, list):
                    return v

        if isinstance(parsed, list):
            return parsed

        print(f"  ⚠ GitHub Models: unexpected response shape")
        return None

    except Exception as e:
        print(f"  ⚠ GitHub Models error: {e}")
        return None


# ─── API call: Gemini (fallback) ──────────────────────────────────

def _call_gemini(prompt):
    """Call Gemini API with automatic model fallback. Returns parsed JSON list."""
    if not GEMINI_API_KEY:
        return None

    for model in GEMINI_MODELS:
        try:
            url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
            resp = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 8192,
                        "responseMimeType": "application/json"
                    }
                },
                timeout=90
            )

            if resp.status_code == 429:
                print(f"  ⚠ Gemini {model} rate limited, trying next...")
                continue
            if resp.status_code != 200:
                print(f"  ⚠ Gemini {model} error {resp.status_code}: {resp.text[:150]}")
                continue

            data = resp.json()
            candidate = (data.get("candidates") or [{}])[0]
            text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            if not text:
                continue

            parsed = _repair_json(text)
            if parsed and isinstance(parsed, list) and len(parsed) > 0:
                print(f"  ✓ Using Gemini model: {model}")
                return parsed

        except Exception as e:
            print(f"  ⚠ Gemini {model} exception: {e}")
            continue

    return None


# ─── Unified API call (tries GitHub first, then Gemini) ───────────

def _call_llm(prompt):
    """Try GitHub Models first, fall back to Gemini."""
    # Try GitHub Models
    result = _call_github_models(prompt)
    if result and isinstance(result, list) and len(result) > 0:
        return result

    # Fall back to Gemini
    result = _call_gemini(prompt)
    if result and isinstance(result, list) and len(result) > 0:
        return result

    return None


# ─── Main entry point ────────────────────────────────────────────

def analyze_with_llm(stocks_data, fyers_client=None):
    """
    Send stock data to LLM in batches, merge results.
    If fyers_client is provided, also fetches 15-min candle data for last 3 days.
    """

    if not GITHUB_TOKEN and not GEMINI_API_KEY:
        return {"error": "No API key set. Add GITHUB_TOKEN or GEMINI_API_KEY in environment."}

    using = "GitHub Models (gpt-4o-mini)" if GITHUB_TOKEN else "Gemini (fallback)"
    print(f"🤖 LLM provider: {using}")

    valid = [s for s in stocks_data if s is not None]
    if not valid:
        return {"error": "No stock data to analyze"}

    # Fetch 15-min candle data for each stock (last 3 trading days)
    if fyers_client and fyers_client.access_token:
        print(f"📊 Fetching 15-min candle data for {len(valid)} stocks...")
        for i, stock in enumerate(valid):
            symbol = stock.get('symbol', f"NSE:{stock['name']}-EQ")
            try:
                df = fyers_client.get_history(symbol, resolution="15", days=5)
                if df is not None and len(df) > 0:
                    # Convert to list of [epoch, O, H, L, C, V]
                    candles = []
                    for idx, row in df.iterrows():
                        candles.append([
                            int(idx.timestamp()),
                            float(row['Open']), float(row['High']),
                            float(row['Low']), float(row['Close']),
                            float(row['Volume'])
                        ])
                    stock['intraday_summary'] = _summarize_intraday(candles)
                else:
                    stock['intraday_summary'] = "No intraday data"
            except Exception as e:
                stock['intraday_summary'] = "No intraday data"

            # Rate limit: 8/sec
            if (i + 1) % 8 == 0:
                time.sleep(1.1)
            else:
                time.sleep(0.13)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(valid)}] 15-min data fetched...")

        print(f"✅ 15-min data ready for {len(valid)} stocks")
    else:
        print("⚠️  No Fyers client — skipping 15-min candle data")
        for stock in valid:
            stock['intraday_summary'] = ""

    # Split into batches
    batches = [valid[i:i + BATCH_SIZE] for i in range(0, len(valid), BATCH_SIZE)]
    rec_map = {}
    errors = []

    print(f"🤖 LLM: analyzing {len(valid)} stocks in {len(batches)} batches...")

    for idx, batch in enumerate(batches):
        names = [s['name'] for s in batch]
        prompt = _build_batch_prompt(batch)
        prompt_len = len(prompt)
        print(f"  📦 Batch {idx+1}/{len(batches)}: {', '.join(names)} ({prompt_len} chars)")

        result = None

        for attempt in range(MAX_RETRIES + 1):
            result = _call_llm(prompt)
            if result and isinstance(result, list) and len(result) > 0:
                break
            if attempt < MAX_RETRIES:
                print(f"  🔄 Retry {attempt+1} for batch {idx+1}...")
                time.sleep(1)

        if result and isinstance(result, list):
            for rec in result:
                name = rec.get("name", "").upper().strip()
                if name:
                    rec_map[name] = {
                        "action": rec.get("action", "HOLD"),
                        "confidence": rec.get("confidence", "LOW"),
                        "reason": rec.get("reason", ""),
                        "target": rec.get("target", 0),
                        "stoploss": rec.get("stoploss", 0),
                        "risk_reward": rec.get("risk_reward", "N/A")
                    }
            print(f"  ✅ Got {len(result)} recommendations")
        else:
            errors.append(f"Batch {idx+1} ({', '.join(names)}) failed")
            print(f"  ❌ Batch {idx+1} failed after retries")

        # Small delay between batches to respect rate limits
        if idx < len(batches) - 1:
            time.sleep(0.5)

    print(f"🤖 LLM done: {len(rec_map)} recommendations, {len(errors)} batch errors")

    if not rec_map:
        return {"error": "All batches failed. " + "; ".join(errors)}

    result = {"success": True, "recommendations": rec_map}
    if errors:
        result["warnings"] = errors
    return result

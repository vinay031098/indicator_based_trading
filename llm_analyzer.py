"""
LLM Stock Analyzer — Uses GitHub Models API (free with GitHub account)
to analyze stock indicators and recommend Buy / Hold / Avoid.

Primary: GitHub Models (gpt-4o-mini via models.inference.ai.azure.com)
Fallback: Google Gemini (gemini-2.5-flash)

Sends stocks in batches to avoid output truncation.
Includes JSON repair for partial/malformed responses.
"""

import os
import json
import re
import time
import requests

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

BATCH_SIZE = 10
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


# ─── Prompt builder (for a batch) ─────────────────────────────────

def _build_batch_prompt(stocks_batch):
    """Build a concise prompt for a small batch of stocks."""

    header = (
        "You are an expert Indian stock market technical analyst.\n"
        "Analyze these stocks for NEXT TRADING DAY.\n\n"
        "Rules:\n"
        "- BUY only if 4+ bullish signals align (RSI<45, MACD bullish, above 200-SMA, good volume)\n"
        "- AVOID if bearish (RSI>70, MACD bearish, below 200-SMA)\n"
        "- HOLD if mixed signals\n"
        "- Keep reason to 1 SHORT sentence\n\n"
        "Stocks:\n\n"
    )

    for s in stocks_batch:
        header += (
            f"[{s['name']}] ₹{s['price']} chg={s['change_pct']:.1f}% score={s['score']}/30 "
            f"RSI={s['rsi']} MACD={s['macd']}(sig={s['macd_signal']},hist={s['macd_hist']}) "
            f"StochK={s['stoch_k']} "
            f"SMA20={s['sma20']} SMA50={s['sma50']} SMA200={s['sma200']} "
            f"BB={s['bb_lower']}-{s['bb_upper']} "
            f"52WH={s['w52_high']} 52WL={s['w52_low']} "
            f"VolR={s['vol_ratio']}x "
            f"ATR={s.get('atr','?')} ADX={s.get('adx','?')} "
            f"CCI={s.get('cci','?')} WillR={s.get('williams_r','?')} "
            f"MFI={s.get('mfi','?')} ROC={s.get('roc','?')}%\n"
        )

    header += (
        "\nReturn ONLY a JSON array. No markdown. No explanation.\n"
        "Each object: {\"name\":\"SYMBOL\",\"action\":\"BUY|HOLD|AVOID\","
        "\"confidence\":\"HIGH|MEDIUM|LOW\",\"reason\":\"short reason\","
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
                    {"role": "system", "content": "You are a stock market technical analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"}
            },
            timeout=60
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
                        "maxOutputTokens": 4096,
                        "responseMimeType": "application/json"
                    }
                },
                timeout=60
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

def analyze_with_llm(stocks_data):
    """Send stock data to LLM in batches, merge results."""

    if not GITHUB_TOKEN and not GEMINI_API_KEY:
        return {"error": "No API key set. Add GITHUB_TOKEN or GEMINI_API_KEY in environment."}

    using = "GitHub Models (gpt-4o-mini)" if GITHUB_TOKEN else "Gemini (fallback)"
    print(f"🤖 LLM provider: {using}")

    valid = [s for s in stocks_data if s is not None]
    if not valid:
        return {"error": "No stock data to analyze"}

    # Split into batches
    batches = [valid[i:i + BATCH_SIZE] for i in range(0, len(valid), BATCH_SIZE)]
    rec_map = {}
    errors = []

    print(f"🤖 LLM: analyzing {len(valid)} stocks in {len(batches)} batches...")

    for idx, batch in enumerate(batches):
        names = [s['name'] for s in batch]
        print(f"  📦 Batch {idx+1}/{len(batches)}: {', '.join(names)}")

        prompt = _build_batch_prompt(batch)
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

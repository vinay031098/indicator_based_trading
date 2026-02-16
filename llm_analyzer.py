"""
LLM Stock Analyzer — Uses Google Gemini (free) to analyze stock indicators
and recommend Buy / Hold / Avoid for next trading day.
"""

import os
import json
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBIl0FszcAAYd7YAR8MPsmhDGLu8S2mVU0")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"


def build_prompt(stocks_data):
    """Build a prompt with all stock indicator data for LLM analysis."""
    
    prompt = """You are an expert Indian stock market technical analyst. Analyze the following NIFTY 50 stocks based on their technical indicators for NEXT TRADING DAY action.

For each stock, give:
1. **Action**: BUY, HOLD, or AVOID
2. **Confidence**: HIGH, MEDIUM, or LOW
3. **Reason**: 1-2 sentence explanation using the indicator data
4. **Target**: Expected price target if BUY
5. **Stoploss**: Suggested stoploss if BUY

IMPORTANT RULES:
- BUY only if multiple strong bullish signals align (score >= 5, RSI < 45, MACD bullish, good volume)
- AVOID if bearish signals dominate (RSI > 70, MACD bearish, below 200-SMA, high ADX downtrend)
- HOLD if mixed signals
- Be conservative. Only recommend BUY with HIGH confidence when at least 4-5 indicators agree
- Consider risk/reward ratio using ATR for stoploss calculation

Here are the stocks with their technical indicators:

"""
    for stock in stocks_data:
        if stock is None:
            continue
        prompt += f"""
═══ {stock['name']} (₹{stock['price']}) ═══
Change: {stock['change_pct']:.2f}% | Score: {stock['score']}/30
RSI(14): {stock['rsi']} | MACD: {stock['macd']} (Signal: {stock['macd_signal']}, Hist: {stock['macd_hist']})
Stochastic %K: {stock['stoch_k']} %D: {stock.get('stoch_d', 'N/A')}
SMA20: {stock['sma20']} | SMA50: {stock['sma50']} | SMA200: {stock['sma200']}
EMA9: {stock.get('ema9', 'N/A')} | EMA21: {stock.get('ema21', 'N/A')}
Bollinger: Lower={stock['bb_lower']} Upper={stock['bb_upper']} Width={stock.get('bb_width', 'N/A')}%
52W High: {stock['w52_high']} | 52W Low: {stock['w52_low']} | Dist from High: {stock['dist_52w']}%
Volume: {stock['volume']:,} | Avg Vol: {stock['avg_volume']:,} | Vol Ratio: {stock['vol_ratio']}x
ATR: {stock.get('atr', 'N/A')} ({stock.get('atr_pct', 'N/A')}%) | ADX: {stock.get('adx', 'N/A')} (+DI: {stock.get('plus_di', 'N/A')} -DI: {stock.get('minus_di', 'N/A')})
CCI: {stock.get('cci', 'N/A')} | Williams %R: {stock.get('williams_r', 'N/A')} | MFI: {stock.get('mfi', 'N/A')}
VWAP: {stock.get('vwap', 'N/A')} | ROC: {stock.get('roc', 'N/A')}% | CMF: {stock.get('cmf', 'N/A')}
Ichimoku: Tenkan={stock.get('ichimoku_tenkan', 'N/A')} Kijun={stock.get('ichimoku_kijun', 'N/A')}
Pivot: PP={stock.get('pivot', 'N/A')} S1={stock.get('pivot_s1', 'N/A')} R1={stock.get('pivot_r1', 'N/A')}
Signals: {', '.join([r['icon'] + ' ' + r['text'] for r in stock.get('reasons', [])])}
"""

    prompt += """

Respond in STRICT JSON format only. No markdown, no code blocks, no explanation outside JSON.
Return a JSON array of objects, one per stock:

[
  {
    "name": "RELIANCE",
    "action": "BUY",
    "confidence": "HIGH",
    "reason": "RSI oversold at 28 with MACD bullish crossover and volume spike. Strong support at 200-SMA.",
    "target": 2850,
    "stoploss": 2720,
    "risk_reward": "1:2.5"
  },
  ...
]

Include ALL stocks in the response. Return ONLY the JSON array, nothing else.
"""
    return prompt


def analyze_with_llm(stocks_data):
    """Send stock data to Gemini and get buy/hold/avoid recommendations."""
    
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set. Add it in Render environment variables."}
    
    # Filter out None stocks
    valid_stocks = [s for s in stocks_data if s is not None]
    
    if not valid_stocks:
        return {"error": "No stock data to analyze"}
    
    prompt = build_prompt(valid_stocks)
    
    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 8192,
                    "responseMimeType": "application/json"
                }
            },
            timeout=60
        )
        
        if response.status_code != 200:
            return {"error": f"Gemini API error: {response.status_code} - {response.text[:200]}"}
        
        data = response.json()
        
        # Extract text from Gemini response
        text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not text:
            return {"error": "Empty response from Gemini"}
        
        # Parse JSON response
        # Clean up response — remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        recommendations = json.loads(text)
        
        # Build a lookup map
        rec_map = {}
        for rec in recommendations:
            name = rec.get("name", "").upper().strip()
            rec_map[name] = {
                "action": rec.get("action", "HOLD"),
                "confidence": rec.get("confidence", "LOW"),
                "reason": rec.get("reason", ""),
                "target": rec.get("target", 0),
                "stoploss": rec.get("stoploss", 0),
                "risk_reward": rec.get("risk_reward", "N/A")
            }
        
        return {"success": True, "recommendations": rec_map}
        
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse LLM response: {str(e)}", "raw": text[:500] if 'text' in dir() else ""}
    except requests.exceptions.Timeout:
        return {"error": "Gemini API timeout. Try again."}
    except Exception as e:
        return {"error": f"LLM analysis failed: {str(e)}"}

import os
import time
import json
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

# ----------------------
# Setup
# ----------------------
app = Flask(__name__)
CORS(app)  # Allow all origins; you can restrict with origins=["https://craftyalpha.com"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY


# ----------------------
# Helpers
# ----------------------
def fetch_bars(ticker, interval, period="7d"):
    """Fetch OHLCV data using yfinance, ensure proper columns."""
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def compute_indicators(df):
    """Compute technical indicators safely."""
    df = df.copy().dropna()
    if df.empty:
        return df

    try:
        df["sma20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["rsi14"] = ta.momentum.rsi(df["close"], window=14)
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["atr14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
    except Exception as e:
        logger.error(f"compute_indicators failed: {e}")
        return pd.DataFrame()
    return df


def summarize_structure(df_daily, df_1h, df_15m, df_1m):
    """Summarize key indicators for multiple timeframes."""
    s = {}
    try:
        s["daily_trend"] = "bullish" if not df_daily.empty and df_daily["close"].iloc[-1] > df_daily["sma20"].iloc[-1] else "bearish"
        s["1h_rsi"] = float(df_1h["rsi14"].iloc[-1]) if not df_1h.empty and "rsi14" in df_1h.columns else None
        s["15m_macd_hist"] = float(df_15m["macd"].iloc[-1] - df_15m["macd_signal"].iloc[-1]) if not df_15m.empty else None
        s["1m_atr"] = float(df_1m["atr14"].iloc[-1]) if not df_1m.empty else None
    except Exception as e:
        logger.error(f"summarize_structure failed: {e}")
    return s


def heuristic_analysis(ticker, capital, summary, df_1m):
    """Simple heuristic fallback if no OpenAI key."""
    entries = []
    last = float(df_1m["close"].iloc[-1]) if not df_1m.empty else None
    atr = summary.get("1m_atr") or (float(df_1m["close"].pct_change().std()) * last if last else None)

    if last is None or atr is None:
        return {"entries": [], "explanation": "Insufficient data for heuristic analysis."}

    daily_trend = summary.get("daily_trend", "n/a")
    macd_hist = summary.get("15m_macd_hist") or 0

    # Flexible heuristic
    if daily_trend in ["bullish", "n/a"] and macd_hist >= -0.1:
        stop = round(last - (2 * atr), 4)
        tgt1 = round(last + (2 * atr), 4)
        tgt2 = round(last + (4 * atr), 4)
        risk_per_trade = 0.01 * capital
        qty = int(max(1, risk_per_trade / (last - stop))) if last - stop > 0 else 0

        entries.append({
            "type": "long",
            "entry_price": last,
            "stop_loss": stop,
            "target_1": tgt1,
            "target_2": tgt2,
            "confidence": "medium",
            "rationale": f"Daily trend {daily_trend} + 15m MACD histogram {macd_hist:.4f}; ATR-based stops."
        })
        return {"entries": entries, "position_size": qty, "explanation": "Flexible heuristic suggestion (no LLM key provided)."}
    else:
        return {"entries": [], "explanation": "No clear heuristic setup detected."}


def build_prompt(ticker, capital, summary, recent_samples):
    """Build OpenAI prompt."""
    return f"""
You are a professional quantitative day-trading assistant.
Ticker: {ticker}
Capital: {capital}
Multi-timeframe summary: {json.dumps(summary)}
Recent samples: {json.dumps(recent_samples)}
Constraints:
- Return up to 3 trade setups with type, entry price, stop-loss, targets, position size, rationale.
- Use ATR-based stops when appropriate.
- Return strictly JSON with keys: entries, position_size, explanation, trade_id.
"""


def call_llm(prompt):
    resp = openai.ChatCompletion.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "system", "content": "You are a trading assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )
    return resp["choices"][0]["message"]["content"]


# ----------------------
# Routes
# ----------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    ticker = data.get('ticker')
    capital = float(data.get('capital', 1000))

    if not ticker:
        return jsonify({'error':'ticker required'}), 400

    # Fetch data
    df_daily = compute_indicators(fetch_bars(ticker, "1d", "1y"))
    df_1h = compute_indicators(fetch_bars(ticker, "60m", "60d"))
    df_15m = compute_indicators(fetch_bars(ticker, "15m", "60d"))
    df_1m = compute_indicators(fetch_bars(ticker, "1m", "7d"))

    summary = summarize_structure(df_daily, df_1h, df_15m, df_1m)

    recent_samples = {
        'daily_last_close': float(df_daily['close'].iloc[-1]) if not df_daily.empty else None,
        '1h_last_close': float(df_1h['close'].iloc[-1]) if not df_1h.empty else None,
        '15m_last_close': float(df_15m['close'].iloc[-1]) if not df_15m.empty else None,
    }

    # Try OpenAI LLM if API key exists
    if OPENAI_KEY:
        try:
            prompt = build_prompt(ticker, capital, summary, recent_samples)
            llm_out = call_llm(prompt)
            try:
                parsed = json.loads(llm_out)
                parsed['trade_id'] = parsed.get('trade_id') or f"{ticker}-{int(time.time())}"
                parsed['source'] = 'LLM'
                return jsonify(parsed)
            except Exception:
                # Failed to parse LLM output; fall back
                heuristic_result = heuristic_analysis(ticker, capital, summary, df_1m)
                heuristic_result['trade_id'] = f"{ticker}-{int(time.time())}"
                heuristic_result['source'] = 'Heuristic (LLM parse failed)'
                return jsonify(heuristic_result)
        except Exception as e:
            # LLM call failed; fall back
            heuristic_result = heuristic_analysis(ticker, capital, summary, df_1m)
            heuristic_result['trade_id'] = f"{ticker}-{int(time.time())}"
            heuristic_result['source'] = f"Heuristic (LLM call failed: {str(e)})"
            return jsonify(heuristic_result)
    else:
        # No API key, use heuristic
        heuristic_result = heuristic_analysis(ticker, capital, summary, df_1m)
        heuristic_result['trade_id'] = f"{ticker}-{int(time.time())}"
        heuristic_result['source'] = 'Heuristic (no LLM key)'
        return jsonify(heuristic_result)


@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.json
    feedback_file = os.environ.get("FEEDBACK_STORE", "llm_trader_feedback.jsonl")
    with open(feedback_file, "a") as f:
        f.write(json.dumps({"ts": time.time(), "payload": payload}) + "\n")
    return jsonify({"ok": True})


# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

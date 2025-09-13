import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import openai

from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
# Setup
# -----------------------------
app = Flask(__name__)

CORS(
    app,
    origins=["https://craftyalpha.com"],  # only allow your WP frontend
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)

logging.basicConfig(level=logging.INFO)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

FEEDBACK_STORE = os.environ.get(
    "FEEDBACK_STORE",
    str(Path.cwd() / "llm_trader_feedback.jsonl")
)

# -----------------------------
# Helpers
# -----------------------------
def fetch_bars(ticker, interval, period="7d"):
    """Download OHLCV data from Yahoo Finance."""
    try:
        df = yf.download(
            ticker, interval=interval, period=period, progress=False
        )
        if df is None or df.empty:
            logging.warning(f"No data for {ticker} {interval}")
            return pd.DataFrame()
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })
        return df
    except Exception as e:
        logging.error(f"fetch_bars failed: {e}")
        return pd.DataFrame()


def compute_indicators(df):
    """Compute technical indicators for a given DataFrame."""
    if df.empty:
        return df
    try:
        df = df.copy().dropna()
        df["sma20"] = ta.trend.sma_indicator(df["close"], window=20)
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["rsi14"] = ta.momentum.rsi(df["close"], window=14)
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["atr14"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=14
        )
    except Exception as e:
        logging.error(f"compute_indicators failed: {e}")
    return df


def summarize_structure(df_daily, df_1h, df_15m, df_1m):
    """Summarize market structure across timeframes."""
    s = {}
    try:
        s["daily_trend"] = (
            "up" if df_daily["close"].iloc[-1] > df_daily["sma20"].iloc[-1] else "down"
        )
    except Exception:
        s["daily_trend"] = "n/a"

    s["1h_rsi"] = float(df_1h["rsi14"].iloc[-1]) if not df_1h.empty else None
    s["15m_macd_hist"] = (
        float(df_15m["macd"].iloc[-1] - df_15m["macd_signal"].iloc[-1])
        if not df_15m.empty else None
    )
    s["1m_atr"] = (
        float(df_1m["atr14"].iloc[-1]) if not df_1m.empty else None
    )

    return s


def heuristic_analysis(ticker, capital, summary, df_1m):
    """Fallback heuristic analysis (no LLM)."""
    entries = []
    if df_1m.empty:
        return {"entries": [], "explanation": "Insufficient 1m data."}

    last = float(df_1m["close"].iloc[-1])
    atr = summary.get("1m_atr") or (df_1m["close"].pct_change().std() * last)

    if atr is None or atr == 0:
        return {"entries": [], "explanation": "No ATR available."}

    daily_trend = summary.get("daily_trend", "n/a")
    macd_hist = summary.get("15m_macd_hist") or 0

    # More permissive heuristic
    if daily_trend in ["up", "n/a"] and macd_hist >= -0.1:
        stop = round(last - (2 * atr), 4)
        tgt1 = round(last + (2 * atr), 4)
        tgt2 = round(last + (4 * atr), 4)
        risk_per_trade = 0.01 * capital
        qty = int(max(1, risk_per_trade / (last - stop))) if last > stop else 0

        entries.append({
            "type": "long",
            "entry_price": last,
            "stop_loss": stop,
            "target_1": tgt1,
            "target_2": tgt2,
            "confidence": "medium",
            "rationale": f"Trend {daily_trend}, MACD {macd_hist:.4f}, ATR-based stops."
        })

        return {
            "entries": entries,
            "position_size": qty,
            "explanation": "Heuristic trade suggestion."
        }
    else:
        return {"entries": [], "explanation": "No clear heuristic setup."}


def build_prompt(ticker, capital, summary, recent_samples):
    return f"""
You are a professional quantitative day-trading assistant.
Ticker: {ticker}
Capital: {capital}
Summary: {json.dumps(summary)}
Recent samples: {json.dumps(recent_samples)}

Rules:
- Up to 3 setups with type, entry, stop, targets, position size, rationale.
- ATR-based stops preferred.
- If no trade: return empty list and reason.

Return strictly JSON: entries (list), position_size (int), explanation (str), trade_id (str).
"""


def call_llm(prompt):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    resp = openai.ChatCompletion.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        messages=[
            {"role": "system", "content": "You are a trading assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )
    return resp["choices"][0]["message"]["content"]

# -----------------------------
# Routes
# -----------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    ticker = data.get("ticker")
    capital = float(data.get("capital", 1000))

    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    # Fetch data
    df_daily = compute_indicators(fetch_bars(ticker, "1d", "1y"))
    df_1h = compute_indicators(fetch_bars(ticker, "60m", "60d"))
    df_15m = compute_indicators(fetch_bars(ticker, "15m", "60d"))
    df_1m = compute_indicators(fetch_bars(ticker, "1m", "7d"))

    summary = summarize_structure(df_daily, df_1h, df_15m, df_1m)

    recent_samples = {
        "daily_last_close": float(df_daily["close"].iloc[-1]) if not df_daily.empty else None,
        "1h_last_close": float(df_1h["close"].iloc[-1]) if not df_1h.empty else None,
        "15m_last_close": float(df_15m["close"].iloc[-1]) if not df_15m.empty else None,
    }

    if OPENAI_KEY:
        try:
            prompt = build_prompt(ticker, capital, summary, recent_samples)
            llm_out = call_llm(prompt)
            parsed = json.loads(llm_out)
            parsed["trade_id"] = parsed.get("trade_id") or f"{ticker}-{int(time.time())}"
            return jsonify(parsed)
        except Exception as e:
            logging.error(f"LLM error: {e}")
            return jsonify({"error": "LLM call failed", "details": str(e)}), 500
    else:
        result = heuristic_analysis(ticker, capital, summary, df_1m)
        result["trade_id"] = f"{ticker}-{int(time.time())}"
        return jsonify(result)


@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.json
    try:
        with open(FEEDBACK_STORE, "a") as f:
            f.write(json.dumps({"ts": time.time(), "payload": payload}) + "\n")
    except Exception as e:
        logging.error(f"feedback write failed: {e}")
    return jsonify({"ok": True})

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

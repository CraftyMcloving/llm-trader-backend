import logging
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_bars(ticker, interval, period="7d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Ensure expected columns
    rename_map = {"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    df = df.rename(columns=rename_map)

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df.columns:
            df[c] = np.nan

    return df


def compute_indicators(df):
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
    s = {}
    try:
        if not df_daily.empty:
            s["daily_trend"] = "bullish" if df_daily["close"].iloc[-1] > df_daily["sma20"].iloc[-1] else "bearish"
        if not df_1h.empty and "rsi14" in df_1h.columns:
            s["1h_rsi"] = float(df_1h["rsi14"].iloc[-1])
        if not df_15m.empty and "rsi14" in df_15m.columns:
            s["15m_rsi"] = float(df_15m["rsi14"].iloc[-1])
        if not df_1m.empty and "rsi14" in df_1m.columns:
            s["1m_rsi"] = float(df_1m["rsi14"].iloc[-1])
    except Exception as e:
        logger.error(f"summarize_structure failed: {e}")
    return s


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    ticker = data.get("ticker", "AAPL")

    # Fetch and compute for different timeframes
    df_daily = compute_indicators(fetch_bars(ticker, "1d", "6mo"))
    df_1h = compute_indicators(fetch_bars(ticker, "1h", "60d"))
    df_15m = compute_indicators(fetch_bars(ticker, "15m", "5d"))
    df_1m = compute_indicators(fetch_bars(ticker, "1m", "1d"))

    summary = summarize_structure(df_daily, df_1h, df_15m, df_1m)

    return jsonify({"ticker": ticker, "summary": summary})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

if OPENAI_KEY:
    prompt = build_prompt(ticker, capital, summary, recent_samples)
    try:
        llm_out = call_llm(prompt)
        try:
            parsed = json.loads(llm_out)
        except Exception:
            return jsonify({'entries': [], 'explanation': llm_out, 'raw': llm_out, 'source':'LLM'})
        parsed['trade_id'] = parsed.get('trade_id') or f"{ticker}-{int(time.time())}"
        parsed['source'] = 'LLM'
        return jsonify(parsed)
    except Exception as e:
        return jsonify({'error':'LLM call failed', 'details': str(e), 'source':'LLM'}), 500
else:
    result = heuristic_analysis(ticker, capital, summary, df_1m)
    result['trade_id'] = f"{ticker}-{int(time.time())}"
    result['source'] = 'Heuristic'
    return jsonify(result)

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import openai
import requests

# ===== Config =====
app = Flask(__name__)
CORS(app, origins=["https://craftyalpha.com"])
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

MISTRAL_API = os.environ.get("MISTRAL_API")  # Optional fallback API

# ===== Data fetching =====
def fetch_bars(ticker, interval, period="7d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ===== Indicators =====
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

# ===== Multi-timeframe summary =====
def summarize_structure(df_daily, df_1h, df_15m, df_1m):
    s = {}
    try:
        s['daily_trend'] = 'up' if not df_daily.empty and df_daily['close'].iloc[-1] > df_daily['sma20'].iloc[-1] else 'down'
        s['1h_rsi'] = float(df_1h['rsi14'].iloc[-1]) if not df_1h.empty else None
        s['15m_macd_hist'] = float(df_15m['macd'].iloc[-1] - df_15m['macd_signal'].iloc[-1]) if not df_15m.empty else None
        s['1m_atr'] = float(df_1m['atr14'].iloc[-1]) if not df_1m.empty else None
    except Exception as e:
        logger.error(f"summarize_structure failed: {e}")
    return s

# ===== Heuristic analysis =====
def heuristic_analysis(ticker, capital, risk_percent, summary, df_1m):
    entries = []
    last = float(df_1m['close'].iloc[-1]) if not df_1m.empty else None
    atr = summary.get('1m_atr') or (float(df_1m['close'].pct_change().std()) * last if last else None)
    if last is None or atr is None:
        return {'entries': [], 'explanation': 'Insufficient data for heuristic analysis.'}

    daily_trend = summary.get('daily_trend', 'n/a')
    macd_hist = summary.get('15m_macd_hist') or 0

    if daily_trend in ['up','n/a'] and macd_hist >= -0.1:
        stop = round(last - (2 * atr), 4)
        tgt1 = round(last + (2 * atr), 4)
        tgt2 = round(last + (4 * atr), 4)
        risk_per_trade_usd = (risk_percent / 100) * capital
        qty = int(risk_per_trade_usd / (last - stop)) if (last - stop) > 0 else 0

        entries.append({
            'type':'long',
            'entry_price': last,
            'stop_loss': stop,
            'target_1': tgt1,
            'target_2': tgt2,
            'confidence': 'medium',
            'rationale': f'Daily trend {daily_trend} + 15m MACD histogram {macd_hist:.4f}; ATR-based stops.',
            'position_size': qty,
            'capital_at_risk_usd': round(risk_per_trade_usd, 2)
        })
    return {'entries': entries, 'position_size': qty if entries else 0, 'explanation': f'Heuristic suggestion with {risk_percent}% risk per trade.'}

# ===== Prompt builder for LLM =====
def build_prompt(ticker, capital, summary, recent_samples, top_n=5):
    return f"""
You are a professional quantitative trading assistant.
Ticker: {ticker}
Capital: {capital}
Multi-timeframe summary: {json.dumps(summary)}
Recent samples: {json.dumps(recent_samples)}
Constraints:
- Provide up to {top_n} candidate trades sorted by risk (least to most).
- Include type, entry price, stop-loss, targets, position size, rationale.
- Use ATR-based stops and explain confidence.
- Return strictly JSON with keys: entries (list), explanation (string).
"""

# ===== Call LLM with fallback =====
def call_llm_with_fallback(prompt):
    trade_json = None
    used_model = None
    openai_models = ["gpt-5", "gpt-4.1", "gpt-4", "gpt-3.5-turbo"]

    for model in openai_models:
        if not OPENAI_KEY:
            break
        try:
            resp = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a trading assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            trade_json = resp.choices[0].message.content
            used_model = model
            logger.info(f"LLM succeeded with model {model}")
            break
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")

    if not trade_json and MISTRAL_API:
        try:
            r = requests.post(MISTRAL_API, json={"prompt": prompt})
            if r.status_code == 200:
                trade_json = r.text
                used_model = "Mistral-7B"
                logger.info("LLM succeeded with Mistral fallback")
        except Exception as ex:
            logger.error(f"Mistral fallback failed: {ex}")

    return trade_json, used_model


# ===== Flask endpoint =====
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    ticker = data.get("ticker")
    capital = float(data.get("capital", 1000))
    risk_percent = float(data.get("risk_percent", 1))

    df_daily = compute_indicators(fetch_bars(ticker, "1d", "1y"))
    df_1h = compute_indicators(fetch_bars(ticker, "60m", "60d"))
    df_15m = compute_indicators(fetch_bars(ticker, "15m", "60d"))
    df_1m = compute_indicators(fetch_bars(ticker, "1m", "7d"))

    summary = summarize_structure(df_daily, df_1h, df_15m, df_1m)
    recent_samples = {
        'daily_last_close': float(df_daily['close'].iloc[-1]) if not df_daily.empty else None,
        '1h_last_close': float(df_1h['close'].iloc[-1]) if not df_1h.empty else None,
        '15m_last_close': float(df_15m['close'].iloc[-1]) if not df_15m.empty else None
    }

    trade_id = f"{ticker}-{int(time.time())}"

   if OPENAI_KEY or MISTRAL_API:
    prompt = build_prompt(ticker, capital, summary, recent_samples, top_n=5)
    llm_out, used_model = call_llm_with_fallback(prompt)

    if llm_out:
        try:
            parsed = json.loads(llm_out)
        except Exception:
            parsed = {'entries': [], 'explanation': llm_out}

        # Ensure risk info is included
        for e in parsed.get('entries', []):
            if 'position_size' not in e and 'stop_loss' in e and 'entry_price' in e:
                risk_usd = (risk_percent / 100) * capital
                qty = int(risk_usd / (e['entry_price'] - e['stop_loss'])) if (e['entry_price'] - e['stop_loss']) > 0 else 0
                e['position_size'] = qty
                e['capital_at_risk_usd'] = round(risk_usd, 2)

        parsed['trade_id'] = trade_id
        parsed['source'] = f"LLM ({used_model})"
        return jsonify(parsed)

    else:
        result = heuristic_analysis(ticker, capital, risk_percent, summary, df_1m)
        result['trade_id'] = trade_id
        result['source'] = 'Heuristic (LLM + fallback failed)'
        return jsonify(result)

else:
    result = heuristic_analysis(ticker, capital, risk_percent, summary, df_1m)
    result['trade_id'] = trade_id
    result['source'] = 'Heuristic'
    return jsonify(result)
        
@app.route("/top5", methods=["GET"])
def top5():
    tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "BTC-USD", "ETH-USD"]
    capital = float(request.args.get("capital", 1000))
    risk_percent = float(request.args.get("risk_percent", 1))

    results = []
    for ticker in tickers:
        try:
            df_daily = compute_indicators(fetch_bars(ticker, "1d", "1y"))
            df_1h = compute_indicators(fetch_bars(ticker, "60m", "60d"))
            df_15m = compute_indicators(fetch_bars(ticker, "15m", "60d"))
            df_1m = compute_indicators(fetch_bars(ticker, "1m", "7d"))

            summary = summarize_structure(df_daily, df_1h, df_15m, df_1m)
            trade = heuristic_analysis(ticker, capital, risk_percent, summary, df_1m)

            if trade["entries"]:
                e = trade["entries"][0]
                risk_score = abs((e["entry_price"] - e["stop_loss"]) / e["entry_price"]) if e["stop_loss"] else 1
                e["ticker"] = ticker
                e["risk_score"] = round(risk_score, 4)
                results.append(e)
        except Exception as ex:
            logger.error(f"Top5 error for {ticker}: {ex}")

    if not results:
        return jsonify({"trades": []})

    # Prepare condensed trade list for LLM
    trade_summaries = [
        f"{t['ticker']} | {t['type'].upper()} | Entry: {t['entry_price']} | SL: {t['stop_loss']} | "
        f"T1: {t['target_1']} | Risk Score: {t['risk_score']}"
        for t in results
    ]
    prompt = (
        "You are a professional trader assistant. Given these candidate trades:\n\n"
        + "\n".join(trade_summaries)
        + "\n\nRank the best 5 trades from safest (lowest risk) to most speculative. "
          "Add a one-sentence rationale for each. Respond in JSON with this format:\n\n"
          "{ \"trades\": [ { \"ticker\": ..., \"rank\": 1, \"risk_level\": ..., \"reason\": ... }, ... ] }"
    )

    # ===== Call LLM with fallback =====
    ranked_json = None
    used_model = None
    openai_models = ["gpt-5", "gpt-4.1", "gpt-4", "gpt-3.5-turbo"]

    for model in openai_models:
        if OPENAI_KEY:
            try:
                resp = openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                ranked_json = json.loads(resp.choices[0].message.content)
                used_model = model
                logger.info(f"Top5 ranking succeeded with model {model}")
                break
            except Exception as e:
                logger.warning(f"Model {model} failed for Top5: {e}")

    if not ranked_json and MISTRAL_API:
        try:
            r = requests.post(MISTRAL_API, json={"prompt": prompt})
            if r.status_code == 200:
                ranked_json = json.loads(r.text)
                used_model = "Mistral-7B"
                logger.info("Top5 ranking succeeded with Mistral fallback")
        except Exception as ex:
            logger.error(f"Mistral Top5 fallback failed: {ex}")

    # Fallback: sort by risk score numerically
    if not ranked_json:
        fallback_sorted = sorted(results, key=lambda x: x["risk_score"])[:5]
        for t in fallback_sorted:
            if t["risk_score"] < 0.01:
                t["risk_level"] = "low"
            elif t["risk_score"] < 0.03:
                t["risk_level"] = "medium"
            else:
                t["risk_level"] = "high"
            t["llm_reason"] = "Ranked heuristically by risk score (LLM unavailable)."
        return jsonify({"trades": fallback_sorted})

    # Merge GPT/Mistral output with trade details
    ranked_trades = []
    for item in ranked_json.get("trades", []):
        match = next((t for t in results if t["ticker"] == item["ticker"]), None)
        if match:
            match["rank"] = item.get("rank")
            match["risk_level"] = item.get("risk_level", "unknown")
            match["llm_reason"] = item.get("reason", "")
            ranked_trades.append(match)

    return jsonify({"trades": sorted(ranked_trades, key=lambda x: x["rank"])[:5], "source": used_model})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

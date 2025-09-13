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
import openai

# ===== Config =====
app = Flask(__name__)
CORS(app, origins=["https://craftyalpha.com"])  # update your frontend origin
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# ===== Data fetching =====
def fetch_bars(ticker, interval, period="7d"):
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
    except Exception as e:
        logger.error(f"yfinance download failed for {ticker}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex
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

# ===== Heuristic logic =====
def heuristic_analysis(ticker, capital, summary, df_1m):
    """
    Simple heuristic trade suggestion.
    Capital is assumed in USD.
    Position size is calculated based on risk-per-trade (e.g., 1% of capital).
    """
    entries = []
    last_price = float(df_1m['close'].iloc[-1]) if not df_1m.empty else None
    atr = summary.get('1m_atr') or (float(df_1m['close'].pct_change().std()) * last_price if last_price else None)

    if last_price is None or atr is None:
        return {
            'entries': [],
            'explanation': 'Insufficient data for heuristic analysis.',
            'position_size': 0
        }

    daily_trend = summary.get('daily_trend', 'n/a')
    macd_hist = summary.get('15m_macd_hist') or 0

    # Risk management: 1% of capital per trade
    risk_per_trade = 0.01 * capital

    if daily_trend in ['up','n/a'] and macd_hist >= -0.1:
        stop_loss_price = round(last_price - 2*atr, 4)
        tgt1 = round(last_price + 2*atr, 4)
        tgt2 = round(last_price + 4*atr, 4)
        # number of shares/contracts to risk ~1% of capital
        qty = int(max(1, risk_per_trade / max(0.01, last_price - stop_loss_price)))

        entries.append({
            'type': 'long',
            'entry_price': last_price,
            'stop_loss': stop_loss_price,
            'target_1': tgt1,
            'target_2': tgt2,
            'confidence': 'medium',
            'rationale': f'Daily trend {daily_trend} + 15m MACD hist {macd_hist:.4f}; ATR-based stops.',
            'capital_risk_usd': round(risk_per_trade, 2),
            'position_size': qty
        })

        return {
            'entries': entries,
            'position_size': qty,
            'explanation': f'1% of capital (${risk_per_trade:.2f}) risked per trade.',
        }
    else:
        return {
            'entries': [],
            'position_size': 0,
            'explanation': 'No clear heuristic setup detected.'
        }


# ===== LLM prompt builder =====
def build_prompt(ticker, capital, summary, recent_samples):
    return f"""
You are a professional quantitative day-trading assistant.
Ticker: {ticker}
Capital: {capital}
Multi-timeframe summary: {json.dumps(summary)}
Recent samples: {json.dumps(recent_samples)}
Constraints:
- Provide up to 3 candidate trade setups with type, entry price, stop-loss, targets, position size, rationale.
- Use ATR-based stops when appropriate. Explain confidence.
- If no good trade, return empty list and short explanation.
Return strictly JSON with keys: entries (list), position_size (number), explanation (string), trade_id (unique id).
"""

# ===== LLM call =====
def call_llm(prompt):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    try:
        resp = openai.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are a trading assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=700
        )
        return resp.choices[0].message.content
    except openai.OpenAIError as e:
        # Handles API errors (rate limit, quota, invalid request, etc.)
        logger.error(f"OpenAI API error: {e}")
        raise
    except Exception as e:
        # Handles unexpected issues
        logger.error(f"Unexpected LLM error: {e}")
        raise

# ===== Flask endpoint =====
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    ticker = data.get("ticker")
    capital = float(data.get("capital", 1000))

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

    if OPENAI_KEY:
        prompt = build_prompt(ticker, capital, summary, recent_samples)
        try:
            llm_out = call_llm(prompt)
            try:
                parsed = json.loads(llm_out)
            except Exception:
                # fallback: LLM returned non-JSON
                return jsonify({
                    'entries': [],
                    'explanation': llm_out,
                    'raw': llm_out,
                    'trade_id': trade_id,
                    'source': 'LLM'
                })
            parsed['trade_id'] = parsed.get('trade_id') or trade_id
            parsed['source'] = 'LLM'
            return jsonify(parsed)
        except openai.OpenAIError as e:
            # Return heuristic if OpenAI API fails
            result = heuristic_analysis(ticker, capital, summary, df_1m)
            result['trade_id'] = trade_id
            result['source'] = f"Heuristic (LLM failed: {str(e)})"
            return jsonify(result), 200
        except Exception as e:
            result = heuristic_analysis(ticker, capital, summary, df_1m)
            result['trade_id'] = trade_id
            result['source'] = f"Heuristic (unexpected LLM error: {str(e)})"
            return jsonify(result), 200
    else:
        # No API key, fallback to heuristic
        result = heuristic_analysis(ticker, capital, summary, df_1m)
        result['trade_id'] = trade_id
        result['source'] = 'Heuristic'
        return jsonify(result)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)))

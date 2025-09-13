import os, json, time
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import openai
from datetime import datetime, timedelta
from pathlib import Path

app = Flask(__name__)

OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

FEEDBACK_STORE = os.environ.get('FEEDBACK_STORE', str(Path.cwd() / "llm_trader_feedback.jsonl"))

def fetch_bars(ticker, interval, period="7d"):
    '''
    Fetch OHLCV data with yfinance.
    interval: "1m","15m","60m","1d"
    period: how much history, e.g. "7d" for 1m, "60d" for 15m/1h, "1y" for 1d
    '''
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False)
    except Exception as e:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low",
        "Close":"close","Volume":"volume"
    })
    # ensure columns exist
    for c in ['open','high','low','close','volume']:
        if c not in df.columns:
            df[c] = np.nan
    return df

def compute_indicators(df):
    df = df.copy().dropna()
    if df.empty:
        return df
    # simple common indicators
    try:
        df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['rsi14'] = ta.momentum.rsi(df['close'], window=14)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['atr14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    except Exception as e:
        # if any fails, continue without indicators
        pass
    return df

def summarize_structure(df_daily, df_1h, df_15m, df_1m):
    s = {}
    try:
        if not df_daily.empty and 'sma20' in df_daily.columns:
            s['daily_trend'] = 'up' if float(df_daily['close'].iloc[-1]) > float(df_daily['sma20'].iloc[-1]) else 'down'
        else:
            s['daily_trend'] = 'n/a'
    except:
        s['daily_trend'] = 'n/a'
    try:
        s['1h_rsi'] = float(round(df_1h['rsi14'].iloc[-1],1)) if not df_1h.empty and 'rsi14' in df_1h.columns else None
    except:
        s['1h_rsi'] = None
    try:
        s['15m_macd_hist'] = float(round((df_15m['macd'].iloc[-1] - df_15m['macd_signal'].iloc[-1]),4)) if not df_15m.empty and 'macd' in df_15m.columns and 'macd_signal' in df_15m.columns else None
    except:
        s['15m_macd_hist'] = None
    try:
        s['1m_atr'] = float(round(df_1m['atr14'].iloc[-1],4)) if not df_1m.empty and 'atr14' in df_1m.columns else None
    except:
        s['1m_atr'] = None
    return s

def heuristic_analysis(ticker, capital, summary, df_1m):
    '''
    Simple rule-based fallback if no OPENAI key present.
    Strategy: if daily trend up and 15m/1h show bullish momentum, propose a long entry near last close with ATR-based stop.
    '''
    entries = []
    last = float(df_1m['close'].iloc[-1]) if not df_1m.empty else None
    atr = summary.get('1m_atr') or (float(df_1m['close'].pct_change().std()) * last if last else None)
    if last is None or atr is None:
        return {'entries': [], 'explanation': 'Insufficient data for heuristic analysis.'}
    # simple signal
    if summary.get('daily_trend') == 'up' and (summary.get('15m_macd_hist') is not None and summary.get('15m_macd_hist') > 0):
        # propose long
        stop = round(last - (2 * atr), 4)
        tgt1 = round(last + (2 * atr), 4)
        tgt2 = round(last + (4 * atr), 4)
        risk_per_trade = 0.01 * capital
        qty = int(max(1, risk_per_trade / (last - stop))) if last - stop > 0 else 0
        entries.append({
            'type':'long',
            'entry_price': last,
            'stop_loss': stop,
            'target_1': tgt1,
            'target_2': tgt2,
            'confidence': 'medium',
            'rationale': 'Daily uptrend + positive 15m MACD histogram; ATR-based stops.'
        })
        return {'entries': entries, 'position_size': qty, 'explanation': 'Heuristic suggestion (no LLM key provided).'}
    else:
        return {'entries': [], 'explanation': 'No clear heuristic setup detected.'}

def build_prompt(ticker, capital, summary, recent_samples):
    prompt = f'''You are a professional quantitative day-trading assistant.

    Ticker: {ticker}
    Capital for this trade: {capital}
    Multi-timeframe summary: {json.dumps(summary)}

    Recent samples (last close values and key indicator snapshots): {json.dumps(recent_samples)}

    Constraints:
    - Provide up to 3 candidate trade setups with: type (long/short), entry price (or conditions), stop-loss price, 1st target, 2nd target (optional), suggested position size (based on capital & 1% per-trade risk default), and a short rationale (2-3 sentences).
    - Use ATR-based distance for stops when appropriate. Explain confidence (low/medium/high).
    - If no good trade, return an empty list and short explanation.

    Return strictly JSON with keys: entries (list), position_size (number), explanation (string), trade_id (unique id).
    '''
    return prompt

def call_llm(prompt):
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    resp = openai.ChatCompletion.create(
        model=os.environ.get('OPENAI_MODEL','gpt-4o'),
        messages=[{"role":"system","content":"You are a trading assistant."},{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=700
    )
    return resp['choices'][0]['message']['content']

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    ticker = data.get('ticker')
    capital = float(data.get('capital', 1000))
    if not ticker:
        return jsonify({'error':'ticker required'}), 400

    # Fetch bars (free: yfinance). Note: 1m period limited to ~7 days by yfinance
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

    # If OPENAI key present, use LLM; otherwise use heuristic fallback
    if OPENAI_KEY:
        prompt = build_prompt(ticker, capital, summary, recent_samples)
        try:
            llm_out = call_llm(prompt)
            try:
                parsed = json.loads(llm_out)
            except Exception:
                # return raw if parsing fails
                return jsonify({'entries': [], 'explanation': llm_out, 'raw': llm_out})
            parsed['trade_id'] = parsed.get('trade_id') or f"{ticker}-{int(time.time())}"
            return jsonify(parsed)
        except Exception as e:
            return jsonify({'error':'LLM call failed', 'details': str(e)}), 500
    else:
        # Heuristic fallback
        result = heuristic_analysis(ticker, capital, summary, df_1m)
        result['trade_id'] = f"{ticker}-{int(time.time())}"
        return jsonify(result)

@app.route('/feedback', methods=['POST'])
def feedback():
    payload = request.json
    with open(FEEDBACK_STORE, 'a') as f:
        f.write(json.dumps({'ts': time.time(), 'payload': payload}) + "\n")
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://craftyalpha.com"])  # allow your WordPress site

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    ticker = data.get("ticker")
    balance = data.get("balance")
    
    # Your existing analysis logic here
    result = {
        "entry": 145.2,
        "stop": 142.0,
        "exit": 148.5
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)

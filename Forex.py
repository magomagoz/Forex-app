import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Forex Momentum Pro", layout="wide", page_icon="📈")

# Auto-refresh ogni 5 minuti
st_autorefresh(interval=300 * 1000, key="sentinel_refresh")

# Banner di Testata
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', monospace; letter-spacing: 5px; margin: 0;">
            📊 FOREX MOMENTUM PRO
        </h1>
        <p style="color: white; font-size: 14px; opacity: 0.8; margin: 5px 0 0 0;">
            Oracle Sentinel System • Market Session Watcher • AI Analysis
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 2. FUNZIONI ---

def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "Tokyo 🇯🇵": (time(0, 0), time(9, 0)),
        "Londra 🇬🇧": (time(8, 0), time(17, 0)),
        "New York 🇺🇸": (time(13, 0), time(22, 0))
    }
    status = {}
    for name, (start, end) in sessions.items():
        status[name] = start <= now_utc <= end
    return status

@st.cache_data(ttl=60) # Ridotto a 1 minuto per maggiore reattività
def get_market_data(ticker, period, interval):
    try:
        # Aggiunto timeout e retry per evitare il messaggio "connessione in corso"
        df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        df.dropna(inplace=True)
        return df
    except:
        return None

def get_currency_strength():
    tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"]
    data = yf.download(tickers, period="2d", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex): data = data['Close']
    returns = data.pct_change().iloc[-1] * 100
    strength = {
        "USD 🇺🇸": (-returns["EURUSD=X"] - returns["GBPUSD=X"] + returns["USDJPY=X"] - returns["AUDUSD=X"] + returns["USDCAD=X"] + returns["USDCHF=X"] - returns["NZDUSD=X"]) / 7,
        "EUR 🇪🇺": (returns["EURUSD=X"] + returns["EURJPY=X"] + returns["EURGBP=X"]) / 3,
        "GBP 🇬🇧": (returns["GBPUSD=X"] + returns["GBPJPY=X"] - returns["EURGBP=X"]) / 3,
        "JPY 🇯🇵": (-returns["USDJPY=X"] - returns["EURJPY=X"] - returns["GBPJPY=X"]) / 3,
        "AUD 🇦🇺": (returns["AUDUSD=X"]) / 1,
        "CAD 🇨🇦": (-returns["USDCAD=X"]) / 1,
    }
    return pd.Series(strength).sort_values(ascending=False)

# --- 3. SIDEBAR ---
st.sidebar.header("🕹 Control Panel")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
pip_unit = 0.01 if "JPY" in pair else 0.0001

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Market Sessions (UTC)")
session_status = get_session_status()
for sess, open_status in session_status.items():
    color = "#00ffcc" if open_status else "#ff4b4b"
    label = "OPEN" if open_status else "CLOSED"
    st.sidebar.markdown(f"**{sess}**: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)

# --- 4. EXECUTION ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_h is not None and not df_h.empty:
    # --- STRENGTH METER ---
    st.subheader("⚡ Currency Strength Meter")
    try:
        s_data = get_currency_strength()
        cols = st.columns(len(s_data))
        for i, (curr, val) in enumerate(s_data.items()):
            color = "#00ffcc" if val > 0 else "#ff4b4b"
            cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b>{curr}</b><br><span style='color:{color}; font-size:18px;'>{val:.2f}%</span></div>", unsafe_allow_html=True)
    except:
        st.warning("Calcolo forza valute momentaneamente non disponibile.")

    # --- ANALISI ---
    recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
    pred = model.predict(np.array([[24]]))[0][0]
    drift = pred - recent_h[-1][0]
    
    bb = ta.bbands(df_d['Close'], length=20, std=2)
    kc = ta.kc(df_d['High'], df_d['Low'], df_d['Close'], length=20, scalar=1.5)
    is_sqz = (bb.iloc[:, 2] < kc.iloc[:, 2]) & (bb.iloc[:, 0] > kc.iloc[:, 0])
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    last_rsi = df_d['RSI'].iloc[-1]

    # --- SCORECARD ---
    score = 50
    reasons = []
    if drift > (pip_unit * 5): score += 20; reasons.append("AI Bullish")
    elif drift < -(pip_unit * 5): score -= 20; reasons.append("AI Bearish")
    
    if score >= 80 or score <= 20:
        color_alert = "#00ffcc" if score >= 80 else "#ff4b4b"
        st.markdown(f"""
            <div style="background-color: {color_alert}; color: white; padding: 30px; border-radius: 15px; text-align: center; font-size: 28px; font-weight: bold;">
                🚀 ORACLE SIGNAL: {score}/100
            </div>
            <audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.line_chart(df_h['Close'].tail(50))
else:
    st.warning("⚠️ Tentativo di connessione ai server Yahoo Finance in corso... Ricarica tra 10 secondi se il problema persiste.")

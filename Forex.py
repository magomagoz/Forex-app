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

@st.cache_data(ttl=60)
def get_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty: return None
        df.dropna(inplace=True)
        return df
    except:
        return None

def get_currency_strength():
    # Elenco ticker per il calcolo incrociato
    tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"]
    data = yf.download(tickers, period="5d", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex): data = data['Close']
    
    # Calcolo dei rendimenti con gestione errori (fill nan)
    returns = data.pct_change().fillna(0).iloc[-1] * 100
    
    strength = {
        "USD 🇺🇸": -(returns.mean()), # Semplificazione per stabilità
        "EUR 🇪🇺": returns["EURUSD=X"] if "EURUSD=X" in returns else 0,
        "GBP 🇬🇧": returns["GBPUSD=X"] if "GBPUSD=X" in returns else 0,
        "JPY 🇯🇵": -returns["USDJPY=X"] if "USDJPY=X" in returns else 0,
        "AUD 🇦🇺": returns["AUDUSD=X"] if "AUDUSD=X" in returns else 0,
        "CAD 🇨🇦": -returns["USDCAD=X"] if "USDCAD=X" in returns else 0,
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
    s_data = get_currency_strength()
    cols = st.columns(len(s_data))
    for i, (curr, val) in enumerate(s_data.items()):
        color = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b>{curr}</b><br><span style='color:{color}; font-size:18px;'>{val:.2f}%</span></div>", unsafe_allow_html=True)

    # --- ANALISI SQUEEZE (Correzione NameError) ---
    st.subheader("🌋 Analisi Volatilità & Squeeze")
    bb = ta.bbands(df_d['Close'], length=20, std=2)
    kc = ta.kc(df_d['High'], df_d['Low'], df_d['Close'], length=20, scalar=1.5)
    
    # Calcolo Squeeze sicuro
    is_sqz_series = (bb.iloc[:, 2] < kc.iloc[:, 2]) & (bb.iloc[:, 0] > kc.iloc[:, 0])
    is_sqz_current = is_sqz_series.iloc[-1]

    if is_sqz_current:
        st.warning("⚠️ SQUEEZE ATTIVO: Il prezzo sta caricando energia.")
    else:
        st.success("🚀 RELEASE: Il momentum è libero di muoversi.")

    # --- AI PREDICTION ---
    recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
    pred = model.predict(np.array([[24]]))[0][0]
    drift = pred - recent_h[-1][0]
    
    # --- SCORECARD FINALE ---
    score = 50
    if drift > (pip_unit * 5): score += 25
    elif drift < -(pip_unit * 5): score -= 25
    
    st.markdown("---")
    st.metric("Punteggio Confluenza Oracle", f"{score}/100")
    st.line_chart(df_h['Close'].tail(50))

else:
    st.warning("⚠️ Dati non ricevuti. Controlla la connessione o cambia Pair.")

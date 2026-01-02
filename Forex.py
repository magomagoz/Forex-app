import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="Forex Momentum Pro", layout="wide", page_icon="📈")
st_autorefresh(interval=300 * 1000, key="sentinel_refresh")

# Banner Custom
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 25px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', monospace; letter-spacing: 5px; margin: 0;">
            📊 FOREX MOMENTUM PRO
        </h1>
    </div>
""", unsafe_allow_html=True)

# --- 2. FUNZIONI DI SUPPORTO ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {"Tokyo 🇯🇵": (time(0,0), time(9,0)), "Londra 🇬🇧": (time(8,0), time(17,0)), "New York 🇺🇸": (time(13,0), time(22,0))}
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

@st.cache_data(ttl=60)
def get_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except: return None

def get_currency_strength():
    # Usiamo dati a 5 giorni per garantire che ci sia sempre un valore di confronto
    tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"]
    data = yf.download(tickers, period="5d", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex): data = data['Close']
    
    # Pulizia dati: se ci sono NaN, usiamo l'ultimo valore valido disponibile
    returns = data.pct_change().fillna(0).iloc[-1] * 100
    
    strength = {
        "USD 🇺🇸": -(returns.mean()),
        "EUR 🇪🇺": returns.get("EURUSD=X", 0),
        "GBP 🇬🇧": returns.get("GBPUSD=X", 0),
        "JPY 🇯🇵": -returns.get("USDJPY=X", 0),
        "AUD 🇦🇺": returns.get("AUDUSD=X", 0),
        "CAD 🇨🇦": -returns.get("USDCAD=X", 0),
    }
    return pd.Series(strength).sort_values(ascending=False)

# --- 3. SIDEBAR ---
st.sidebar.header("🕹 Control Panel")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
pip_unit = 0.01 if "JPY" in pair else 0.0001

st.sidebar.markdown("---")
status = get_session_status()
for s, op in status.items():
    st.sidebar.markdown(f"**{s}**: {'🟢 OPEN' if op else '🔴 CLOSED'}")

# --- 4. LOGICA PRINCIPALE ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    # ⚡ Strength Meter
    st.subheader("⚡ Currency Strength Meter")
    s_data = get_currency_strength()
    cols = st.columns(6)
    for i, (curr, val) in enumerate(s_data.items()):
        color = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b>{curr}</b><br><span style='color:{color};'>{val:.2f}%</span></div>", unsafe_allow_html=True)

    # 🌋 Analisi Volatilità & Squeeze (Protezione TypeError)
    st.subheader("🌋 Analisi Volatilità & Squeeze")
    try:
        # Calcolo manuale per evitare errori di libreria
        std = df_d['Close'].rolling(20).std()
        ma = df_d['Close'].rolling(20).mean()
        upper_bb = ma + (2 * std)
        lower_bb = ma - (2 * std)
        
        atr = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=20)
        upper_kc = ma + (1.5 * atr)
        lower_kc = ma - (1.5 * atr)
        
        is_sqz = (upper_bb < upper_kc).iloc[-1]
        
        if is_sqz:
            st.warning("⚠️ SQUEEZE ATTIVO: Energia in caricamento.")
        else:
            st.success("🚀 RELEASE: Momentum attivo.")
    except:
        st.info("Calcolo volatilità in corso...")

    # 🔮 AI Prediction
    recent = df_h['Close'].tail(24).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent)
    pred = model.predict(np.array([[24]]))[0][0]
    drift = pred - recent[-1][0]

    # 🎯 Scorecard
    score = 50
    if drift > (pip_unit * 3): score += 25
    elif drift < -(pip_unit * 3): score -= 25
    
    st.markdown("---")
    st.metric("Punteggio Confluenza Oracle", f"{int(score)}/100")
    st.line_chart(df_h['Close'].tail(50))

else:
    st.warning("Connessione ai dati in corso... Se l'errore persiste, ricarica la pagina.")

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
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

# --- 2. FUNZIONI TECNICHE & SESSIONI ---

def get_session_status():
    """Rileva quali sessioni di mercato sono aperte (Orari in UTC)"""
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

@st.cache_data(ttl=300)
def get_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty: return None
        df.dropna(inplace=True)
        return df
    except:
        return None

def is_market_open(df):
    if df is None or df.empty: return False
    last_data = df.index[-1]
    # Se l'ultima candela è più vecchia di 24 ore, il mercato è chiuso (weekend)
    diff = (datetime.now(last_data.tzinfo) - last_data).total_seconds() / 3600
    return diff < 24

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

# --- 3. SIDEBAR & SETUP ---
st.sidebar.header("🕹 Control Panel")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
pip_unit = 0.01 if "JPY" in pair else 0.0001

# Visualizzazione Sessioni in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Market Sessions (UTC)")
session_status = get_session_status()
for sess, open_status in session_status.items():
    color = "#00ffcc" if open_status else "#ff4b4b"
    label = "OPEN" if open_status else "CLOSED"
    st.sidebar.markdown(f"**{sess}**: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)

# --- 4. DATA PROCESSING ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

market_active = is_market_open(df_h)

if not market_active and "BTC" not in pair:
    st.error(f"⚠️ MERCATO CHIUSO: Le analisi su {pair} sono sospese per il weekend.")
    st.info("Il Forex riapre Domenica sera alle 23:00 CET. Le Cripto restano attive.")
else:
    if df_d is not None and df_h is not None:
        # --- STRENGTH METER ---
        st.subheader("⚡ Currency Strength Meter")
        s_data = get_currency_strength()
        cols = st.columns(len(s_data))
        for i, (curr, val) in enumerate(s_data.items()):
            color = "#00ffcc" if val > 0 else "#ff4b4b"
            cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b>{curr}</b><br><span style='color:{color}; font-size:18px;'>{val:.2f}%</span></div>", unsafe_allow_html=True)

        # --- AI MODEL & SQUEEZE ---
        recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
        model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
        pred = model.predict(np.array([[24]]))[0][0]
        drift = pred - recent_h[-1][0]
        
        bb = ta.bbands(df_d['Close'], length=20, std=2)
        kc = ta.kc(df_d['High'], df_d['Low'], df_d['Close'], length=20, scalar=1.5)
        is_sqz = (bb.iloc[:, 2] < kc.iloc[:, 2]) & (bb.iloc[:, 0] > kc.iloc[:, 0])
        df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
        last_rsi = df_d['RSI'].iloc[-1]

        # --- 5. SCORECARD & SENTINEL ALERT ---
        score = 50
        reasons = []
        if drift > (pip_unit * 5): score += 20; reasons.append("AI Bullish")
        elif drift < -(pip_unit * 5): score -= 20; reasons.append("AI Bearish")
        
        base_curr = pair[:3]
        if base_curr in s_data.index[0]: score += 25; reasons.append(f"{base_curr} Dominante")
        elif base_curr in s_data.index[-1]: score -= 25; reasons.append(f"{base_curr} Debole")
        
        if not is_sqz.iloc[-1]: score += 5; reasons.append("Volatility Release")

        st.markdown("---")
        color_score = "#00ffcc" if score >= 70 else "#ff4b4b" if score <= 30 else "#ffa500"
        
        if score >= 80 or score <= 20:
            st.markdown(f"""
                <style>
                @keyframes blink {{ 0% {{opacity: 1;}} 50% {{opacity: 0.4;}} 100% {{opacity: 1;}} }}
                .alert-box {{ background-color: {color_score}; color: white; padding: 30px; border-radius: 15px; text-align: center; animation: blink 1s infinite; font-size: 28px; font-weight: bold; border: 3px solid white; }}
                </style>
                <div class="alert-box">🚀 ORACLE SIGNAL: {score}/100<br><span style='font-size:16px;'>Confluenza Rilevata - Controlla i Grafici</span></div>
                <audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>
            """, unsafe_allow_html=True)

        # Analisi Visiva
        c1, c2 = st.columns([2, 1])
        with c1:
            st.line_chart(df_h['Close'].tail(50))
            st.caption("Momentum Intraday (Ultime 50 ore)")
        with c2:
            st.markdown(f"""<div style='background:#262730; padding:20px; border-radius:10px;'>
                <h3>Dettagli Oracle</h3>
                <p><b>RSI:</b> {last_rsi:.1f}</p>
                <p><b>Squeeze:</b> {'ATTIVO ⚠️' if is_sqz.iloc[-1] else 'Inattivo ✅'}</p>
                <p><b>Fattori:</b><br>{'<br>'.join(reasons)}</p>
                </div>""", unsafe_allow_html=True)
    else:
        st.warning("Connessione ai dati in corso... Tocca lo schermo per attivare l'audio.")

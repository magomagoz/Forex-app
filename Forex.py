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
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")
st_autorefresh(interval=300 * 1000, key="sentinel_refresh")

# --- BANNER DI TESTATA ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', Courier, monospace; letter-spacing: 5px; margin: 0;">
            📊 FOREX MOMENTUM PRO
        </h1>
        <p style="color: white; font-size: 14px; opacity: 0.8; margin: 5px 0 0 0;">
            Sentinel System • AI Confluence • Notification v2.0
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 2. FUNZIONI TECNICHE ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "Tokyo 🇯🇵": (time(0,0), time(9,0)), 
        "Londra 🇬🇧": (time(8,0), time(17,0)), 
        "New York 🇺🇸": (time(13,0), time(22,0))
    }
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

@st.cache_data(ttl=300)
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
    tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"]
    data = yf.download(tickers, period="5d", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
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

def get_pip_value(pair):
    if "JPY" in pair: return 0.01, "{:.2f}"
    return 0.0001, "{:.4f}"

# --- 3. SIDEBAR ---
st.sidebar.header("🕹 Trading Desk")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
pip_unit, price_fmt = get_pip_value(pair)

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 4. LOGICA SENTINEL ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    # --- CALCOLO AI & DRIFT ---
    recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
    pred_1h = model.predict(np.array([[24]]))[0][0]
    drift = pred_1h - recent_h[-1][0]
    
    # --- CALCOLO SQUEEZE ---
    std = df_d['Close'].rolling(20).std()
    ma = df_d['Close'].rolling(20).mean()
    upper_bb = ma + (2 * std)
    lower_bb = ma - (2 * std)
    atr_val = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=20)
    upper_kc = ma + (1.5 * atr_val)
    is_sqz = (upper_bb.iloc[-1] < upper_kc.iloc[-1])

    # --- CALCOLO FORZA VALUTE ---
    strength_data = get_currency_strength()

    # --- 5. ORACLE SCORE & NOTIFICHE TESTUALI ---
    final_score = 50
    reasons = []
    
    if drift > (pip_unit * 2): final_score += 20; reasons.append("AI Bullish Inerzia")
    elif drift < -(pip_unit * 2): final_score -= 20; reasons.append("AI Bearish Inerzia")
    
    if strength_data.index[0] in pair[:3]: final_score += 20; reasons.append(f"Forza: {pair[:3]} Dominante")
    elif strength_data.index[-1] in pair[:3]: final_score -= 20; reasons.append(f"Forza: {pair[:3]} Debole")

    # --- NOTIFICA PERSISTENTE IN ALTO ---
    if final_score >= 80:
        st.toast(f"🚀 SEGNALE LONG RILEVATO: {pair}", icon="🔥")
        st.error(f"⚠️ SENTINEL ALERT: Confluenza Rialzista ({final_score}/100) - {', '.join(reasons)}")
    elif final_score <= 20:
        st.toast(f"📉 SEGNALE SHORT RILEVATO: {pair}", icon="⚠️")
        st.error(f"⚠️ SENTINEL ALERT: Confluenza Ribassista ({final_score}/100) - {', '.join(reasons)}")

    # --- 6. VISUALIZZAZIONE ---
    st.subheader("⚡ Currency Strength Meter")
    cols = st.columns(6)
    for i, curr in enumerate(strength_data.index[:6]):
        val = strength_data[curr]
        col_c = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"<div style='text-align:center; border:1px solid #333; border-radius:10px; padding:10px; background:#111;'><b>{curr}</b><br><span style='color:{col_c};'>{val:.2f}%</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Prezzo", price_fmt.format(recent_h[-1][0]))
    m2.metric("AI Prediction (+1h)", price_fmt.format(pred_1h), f"{drift:.5f}")
    m3.metric("Oracle Score", f"{final_score}/100")

    st.line_chart(df_h['Close'].tail(50))
    
    if is_sqz:
        st.warning("🌋 SQUEEZE ATTIVO: Possibile movimento esplosivo imminente.")
    
    # Audio Alert
    if final_score >= 80 or final_score <= 20:
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)

else:
    st.info("In attesa di dati freschi dal mercato...")

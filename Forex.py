import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
import pytz
import time as time_lib
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIGURAZIONE & REFRESH ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

# Refresh globale ogni 120 secondi
st_autorefresh(interval=120 * 1000, key="sentinel_full_refresh")

if 'signal_history' not in st.session_state:
    st.session_state['signal_history'] = pd.DataFrame(columns=['Orario', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP'])

# --- 2. FUNZIONI TECNICHE (Tutte le tue logiche originali) ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "Tokyo 🇯🇵": (time(0,0), time(9,0)), 
        "Londra 🇬🇧": (time(8,0), time(17,0)), 
        "New York 🇺🇸": (time(13,0), time(22,0))
    }
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

def is_low_liquidity():
    now_utc = datetime.now(pytz.utc).time()
    return time(23, 0) <= now_utc or now_utc <= time(1, 0)

@st.cache_data(ttl=110)
def get_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except:
        return None

def get_currency_strength():
    try:
        tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"]
        data = yf.download(tickers, period="2d", interval="1d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data['Close']
        returns = data.pct_change().iloc[-1] * 100
        strength = {
            "USD 🇺🇸": (-returns["EURUSD=X"] - returns["GBPUSD=X"] + returns["USDJPY=X"] - returns["AUDUSD=X"] + returns["USDCAD=X"] + returns["USDCHF=X"] - returns["NZDUSD=X"]) / 7,
            "EUR 🇪🇺": (returns["EURUSD=X"] + returns["EURJPY=X"] + returns["EURGBP=X"]) / 3,
            "GBP 🇬🇧": (returns["GBPUSD=X"] + returns["GBPJPY=X"] - returns["EURGBP=X"]) / 3,
            "JPY 🇯🇵": (-returns["USDJPY=X"] - returns["EURJPY=X"] - returns["GBPJPY=X"]) / 3,
            "AUD 🇦🇺": returns.get("AUDUSD=X", 0),
            "CAD 🇨🇦": -returns.get("USDCAD=X", 0),
        }
        return pd.Series(strength).sort_values(ascending=False)
    except:
        return pd.Series()

def get_pip_info(pair):
    if "JPY" in pair: return 0.01, "{:.2f}", 1000 
    return 0.0001, "{:.4f}", 10

# --- 3. SIDEBAR & TIMER SCORREVOLE ---
st.sidebar.header("🕹 Trading Desk")

if "timer_start" not in st.session_state:
    st.session_state.timer_start = time_lib.time()

elapsed = time_lib.time() - st.session_state.timer_start
remaining = max(0, int(120 - elapsed))

if remaining <= 0:
    st.session_state.timer_start = time_lib.time()
    remaining = 120

st.sidebar.metric("⏳ Prossimo Update Dati", f"{remaining}s")

pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)

if st.sidebar.button("🔄 AGGIORNA ORA"):
    st.cache_data.clear()
    st.session_state.timer_start = time_lib.time()
    st.rerun()

st.sidebar.markdown("---")
for s, op in get_session_status().items():
    st.sidebar.markdown(f"**{s}**: {'🟢 OPEN' if op else '🔴 CLOSED'}")

# --- 4. BANNER ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', Courier, monospace; letter-spacing: 5px; margin: 0;">📊 FOREX MOMENTUM PRO</h1>
        <p style="color: white; font-size: 14px; opacity: 0.8; margin: 5px 0 0 0;">Sentinel System • AI Drift Engine • Visual Analytics v6.0</p>
    </div>
""", unsafe_allow_html=True)

# --- 5. CORE ENGINE ---
pip_unit, price_fmt, pip_mult = get_pip_info(pair)
df_h = get_market_data(pair, "1d", "5m") 
df_d = get_market_data(pair, "1y", "1d")

if df_h is not None and not df_h.empty and df_d is not None:
    # GRAFICO LINEARE (Fix per iPad)
    st.subheader(f"📈 Analisi Intraday: {pair}")
    st.line_chart(df_h['Close'])
    
    prezzo_attuale = df_h['Close'].iloc[-1]
    st.metric("Prezzo Attuale", price_fmt.format(prezzo_attuale), f"{prezzo_attuale - df_h['Close'].iloc[-2]:.5f}")

    # STRENGTH METER
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    s_data = get_currency_strength()
    if not s_data.empty:
        cols = st.columns(6)
        for i, (curr, val) in enumerate(s_data.items()[:6]):
            col_c = "#00ffcc" if val > 0 else "#ff4b4b"
            cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b style='color:white;'>{curr}</b><br><span style='color:{col_c}; font-weight:bold;'>{val:.2f}%</span></div>", unsafe_allow_html=True)

    # AI ANALYSIS
    st.markdown("---")
    lookback = 24
    recent_prices = df_h['Close'].tail(lookback).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(lookback).reshape(-1, 1), recent_prices)
    pred_price = model.predict(np.array([[lookback]]))[0][0]
    drift = pred_price - prezzo_attuale

    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    last_rsi, last_atr = df_d['RSI'].iloc[-1], df_d['ATR'].iloc[-1]

    score = 50
    if drift > (pip_unit * 2): score += 25
    elif drift < -(pip_unit * 2): score -= 25
    if not s_data.empty:
        if s_data.index[0] in pair[:3]: score += 25
        elif s_data.index[-1] in pair[:3]: score -= 25

    m1, m2, m3 = st.columns(3)
    m1.metric("RSI (Daily)", f"{last_rsi:.1f}")
    m2.metric("Inerzia AI (1h)", price_fmt.format(pred_price), f"{drift:.5f}")
    m3.metric("Sentinel Score", f"{score}/100")

    # SEGNALE
    st.markdown("---")
    if is_low_liquidity():
        st.warning("⚠️ FILTRO LIQUIDITÀ ATTIVO (Pausa Rollover)")
        action = None
    else:
        action = "LONG" if (score >= 75 and last_rsi < 65) else "SHORT" if (score <= 25 and last_rsi > 35) else None

    if action:
        sl = prezzo_attuale - (1.5 * last_atr) if action == "LONG" else prezzo_attuale + (1.5 * last_atr)
        tp = prezzo_attuale + (3 * last_atr) if action == "LONG" else prezzo_attuale - (3 * last_atr)
        risk_cash = balance * (risk_pc / 100)
        dist_pips = abs(prezzo_attuale - sl) / pip_unit
        lotti = risk_cash / (dist_pips * pip_mult) if dist_pips > 0 else 0
        
        color = "#00ffcc" if action == "LONG" else "#ff4b4b"
        st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 20px; border-radius: 15px; background: #0e1117;">
                <h2 style="color: {color}; margin-top:0;">🚀 SEGNALE: {action}</h2>
                <table style="width:100%; color: white; font-size: 18px;">
                    <tr><td><b>Entry:</b> {price_fmt.format(prezzo_attuale)}</td><td><b>SL:</b> {price_fmt.format(sl)}</td><td><b>TP:</b> {price_fmt.format(tp)}</td></tr>
                    <tr style="color: #ffcc00;"><td><b>Rischio:</b> ${risk_cash:.2f}</td><td><b>Pips:</b> {dist_pips:.1f}</td><td><b>SIZE: {lotti:.2f} Lotti</b></td></tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)
        
        new_row = pd.DataFrame([{'Orario': datetime.now().strftime("%H:%M:%S"), 'Asset': pair, 'Direzione': action, 'Prezzo': prezzo_attuale, 'SL': sl, 'TP': tp}])
        st.session_state['signal_history'] = pd.concat([st.session_state['signal_history'], new_row], ignore_index=True)

    if not st.session_state['signal_history'].empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Registro")
        st.sidebar.dataframe(st.session_state['signal_history'].tail(5))

else:
    st.error("Caricamento dati in corso...")

# LOOP TIMER (Cruciale per lo scorrimento visivo)
if remaining > 0:
    time_lib.sleep(1)
    st.rerun()

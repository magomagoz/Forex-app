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

# Refresh globale ogni 60 secondi
st_autorefresh(interval=60 * 1000, key="sentinel_refresh")

if 'signal_history' not in st.session_state:
    st.session_state['signal_history'] = pd.DataFrame(columns=['Orario', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP'])

# --- 2. FUNZIONI TECNICHE ---
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

@st.cache_data(ttl=50)
def get_realtime_data(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except:
        return None

def get_currency_strength():
    try:
        tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"]
        data = yf.download(tickers, period="2d", interval="1d", progress=False, timeout=15)
        if data is None or data.empty: return pd.Series(dtype=float)
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
        return pd.Series(dtype=float)

def get_pip_info(pair):
    if "JPY" in pair: return 0.01, "{:.2f}", 1000 
    return 0.0001, "{:.4f}", 10

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi = df['Close'], df['RSI']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DIV. BEARISH"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 DIV. BULLISH"
    return "Neutrale"

# --- 3. SIDEBAR & TIMER ---
st.sidebar.header("🕹 Trading Desk")

if "last_update" not in st.session_state:
    st.session_state.last_update = time_lib.time()

elapsed = time_lib.time() - st.session_state.last_update
remaining = max(0, int(60 - elapsed))

if remaining <= 0:
    st.session_state.last_update = time_lib.time()
    remaining = 60

st.sidebar.metric("⏳ Prossimo Scan", f"{remaining}s")

pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)

if st.sidebar.button("🔄 FORZA AGGIORNAMENTO"):
    st.cache_data.clear()
    st.session_state.last_update = time_lib.time()
    st.rerun()

st.sidebar.markdown("---")
for s, op in get_session_status().items():
    st.sidebar.markdown(f"**{s}**: {'🟢 OPEN' if op else '🔴 CLOSED'}")

# --- 4. BANNER ---
st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 20px; border-radius: 15px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 MOMENTUM PRO V9</h1><p style="color: white; opacity: 0.8;">Real-Time Data Stream (1m) • Sentinel AI</p></div>', unsafe_allow_html=True)

# --- 5. DATA ENGINE ---
pip_unit, price_fmt, pip_mult = get_pip_info(pair)
df_rt = get_realtime_data(pair) 
df_d = yf.download(pair, period="1y", interval="1d", progress=False) 

if df_rt is not None and not df_rt.empty:
    # GRAFICO REAL-TIME
    st.subheader(f"📈 Grafico Real-Time (Intervallo 1m): {pair}")
    st.line_chart(df_rt['Close'].tail(60), use_container_width=True)
    
    curr_price = float(df_rt['Close'].iloc[-1])
    diff_prezzo = curr_price - float(df_rt['Close'].iloc[-2])
    st.metric("Prezzo Live (Yahoo)", price_fmt.format(curr_price), f"{diff_prezzo:.5f}")

    # STRENGTH METER
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    s_data = get_currency_strength()
    
    if not s_data.empty:
        cols = st.columns(6)
        for i, (curr, val) in enumerate(s_data.items()):
            if i < 6:
                col_c = "#00ffcc" if val > 0 else "#ff4b4b"
                cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b style='color:white;'>{curr}</b><br><span style='color:{col_c}; font-weight:bold;'>{val:.2f}%</span></div>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Dati Forza Valute momentaneamente non disponibili.")

    # ANALISI AI & SENTINEL
    if df_d is not None and not df_d.empty:
        if isinstance(df_d.columns, pd.MultiIndex): 
            df_d.columns = df_d.columns.get_level_values(0)
        
        st.markdown("---")
        df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
        df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
        df_d['ADX'] = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])['ADX_14']
        
        last_rsi = float(df_d['RSI'].iloc[-1])
        last_atr = float(df_d['ATR'].iloc[-1])
        last_adx = float(df_d['ADX'].iloc[-1])
        div_sig = detect_divergence(df_d)

        # AI Drift (15 min)
        lookback = 15
        y = df_rt['Close'].tail(lookback).values
        x = np.arange(lookback).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        drift = model.predict([[lookback]])[0] - curr_price

        score = 50
        if drift > (pip_unit * 2): score += 25
        elif drift < -(pip_unit * 2): score -= 25
        if not s_data.empty:
            if s_data.index[0] in pair[:3]: score += 25
            elif s_data.index[-1] in pair[:3]: score -= 25

        c1, c2, c3 = st.columns(3)
        c1.metric("RSI Daily", f"{last_rsi:.1f}", div_sig)
        c2.metric("Inerzia AI (15m)", price_fmt.format(curr_price + drift), f"{drift:.5f}")
        c3.metric("Sentinel Score", f"{score}/100")
        
        # LOGICA SEGNALE
        if not is_low_liquidity():
            action = "LONG" if (score >= 75 and last_rsi < 65) else "SHORT" if (score <= 25 and last_rsi > 35) else None
            if action:
                sl = curr_price - (1.5 * last_atr) if action == "LONG" else curr_price + (1.5 * last_atr)
                tp = curr_price + (3 * last_atr) if action == "LONG" else curr_price - (3 * last_atr)
                
                risk_cash = balance * (risk_pc / 100)
                dist_pips = abs(curr_price - sl) / pip_unit
                lotti = risk_cash / (dist_pips * pip_mult) if dist_pips > 0 else 0
                
                color = "#00ffcc" if action == "LONG" else "#ff4b4b"
                st.markdown(f"""
                    <div style="border: 2px solid {color}; padding: 20px; border-radius: 15px; background: #0e1117;">
                        <h2 style="color: {color}; margin-top:0;">🚀 SEGNALE SENTINEL: {action}</h2>
                        <p style="font-size:18px;">Entry: {price_fmt.format(curr_price)} | SL: {price_fmt.format(sl)} | TP: {price_fmt.format(tp)}</p>
                        <p style="color:#ffcc00; font-weight:bold;">SIZE CONSIGLIATA: {lotti:.2f} LOTTI</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)
                
                # Update History
                new_sig = pd.DataFrame([{'Orario': datetime.now().strftime("%H:%M:%S"), 'Asset': pair, 'Direzione': action, 'Prezzo': curr_price, 'SL': sl, 'TP': tp}])
                st.session_state['signal_history'] = pd.concat([st.session_state['signal_history'], new_sig], ignore_index=True)

        # INFO EXTRA
        with st.expander("📊 Dettagli Analisi"):
            st.write(f"ADX (Forza Trend): {last_adx:.1f}")
            st.write(f"ATR (Volatilità): {last_atr:.5f}")

else:
    st.error("In attesa di connessione a Yahoo Finance o dati non disponibili...")

# REGISTRO NELLA SIDEBAR
if not st.session_state['signal_history'].empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Storico Segnali")
    st.sidebar.dataframe(st.session_state['signal_history'].tail(5))

# Loop Timer per refresh fluido
time_lib.sleep(1)
st.rerun()

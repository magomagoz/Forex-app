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
import plotly.graph_objects as go

# --- 1. CONFIGURAZIONE & REFRESH ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")
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
    # Filtro weekend e ore morte (23:00 - 01:00 UTC)
    is_weekend = datetime.now(pytz.utc).weekday() >= 5
    is_dead_hours = time(23, 0) <= now_utc or now_utc <= time(1, 0)
    return is_weekend or is_dead_hours

@st.cache_data(ttl=50)
def get_realtime_data(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1m", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def get_currency_strength():
    try:
        tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"]
        data = yf.download(tickers, period="2d", interval="1d", progress=False, timeout=15)
        if data is None or data.empty: return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex): data = data['Close']
        returns = data.pct_change().iloc[-1] * 100
        strength = {
            "USD": -returns.mean(),
            "EUR": returns.get("EURUSD=X", 0),
            "GBP": returns.get("GBPUSD=X", 0),
            "JPY": -returns.get("USDJPY=X", 0),
            "AUD": returns.get("AUDUSD=X", 0),
            "CAD": -returns.get("USDCAD=X", 0)
        }
        return pd.Series(strength).sort_values(ascending=False)
    except: return pd.Series(dtype=float)

def get_asset_params(pair):
    if "-" in pair: return 1.0, "{:.2f}", 1, "CRYPTO"
    if "JPY" in pair: return 0.01, "{:.2f}", 100, "FOREX"
    return 0.0001, "{:.5f}", 10, "FOREX"

def detect_divergence(df):
    if len(df) < 25: return "Analisi..."
    price, rsi = df['close'], df['rsi']
    if price.iloc[-1] > price.iloc[-20:].max() * 0.99 and rsi.iloc[-1] < rsi.iloc[-20:].max():
        return "📉 DIV. BEARISH"
    if price.iloc[-1] < price.iloc[-20:].min() * 1.01 and rsi.iloc[-1] > rsi.iloc[-20:].min():
        return "📈 DIV. BULLISH"
    return "Neutrale"

# --- 3. SIDEBAR & TIMER (LE 20 RIGHE MANCANTI) ---
st.sidebar.header("🕹 Trading Desk")
if "start_time" not in st.session_state: st.session_state.start_time = time_lib.time()
countdown = 60 - int(time_lib.time() - st.session_state.start_time) % 60
st.sidebar.metric("⏳ Next AI Scan", f"{countdown}s")

pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance ($)", value=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Market Sessions")
for s, op in get_session_status().items():
    st.sidebar.write(f"{'🟢' if op else '🔴'} {s}")

# --- 4. BANNER & GRAFICO ---
st.markdown('<div style="background: #0e1117; padding: 20px; border-radius: 15px; border: 1px solid #00ffcc; text-align: center;"><h1 style="color: #00ffcc; margin:0;">📊 SENTINEL AI V15</h1></div>', unsafe_allow_html=True)

pip_unit, price_fmt, pip_mult, asset_type = get_asset_params(pair)
df_rt = get_realtime_data(pair)
df_d = yf.download(pair, period="1y", interval="1d", progress=False)

if df_rt is not None and not df_rt.empty:
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    
    c_up = [c for c in df_rt.columns if c.startswith('BBU')][0]
    c_lo = [c for c in df_rt.columns if c.startswith('BBL')][0]

    # Grafico Candlestick
    fig = go.Figure(data=[go.Candlestick(x=df_rt.index[-60:], open=df_rt['open'][-60:], high=df_rt['high'][-60:], low=df_rt['low'][-60:], close=df_rt['close'][-60:], name='Price')])
    fig.add_trace(go.Scatter(x=df_rt.index[-60:], y=df_rt[c_up][-60:], line=dict(color='rgba(0, 255, 204, 0.2)'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=df_rt.index[-60:], y=df_rt[c_lo][-60:], line=dict(color='rgba(0, 255, 204, 0.2)'), fill='tonexty', name='Lower BB'))
    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    curr_price = float(df_rt['close'].iloc[-1])
    st.metric(f"Live {pair}", price_fmt.format(curr_price))

    # --- AI ANALYSIS ---
    if df_d is not None and not df_d.empty:
        df_d.columns = [c.lower() for c in df_d.columns]
        df_d['rsi'] = ta.rsi(df_d['close'], length=14)
        df_d['atr'] = ta.atr(df_d['high'], df_d['low'], df_d['close'], length=14)
        
        div = detect_divergence(df_d)
        model = LinearRegression().fit(np.arange(15).reshape(-1,1), df_rt['close'].tail(15).values)
        drift = model.predict([[15]])[0] - curr_price
        
        score = 50
        if curr_price < df_rt[c_lo].iloc[-1]: score += 20
        if curr_price > df_rt[c_up].iloc[-1]: score -= 20

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("RSI (D)", f"{df_d['rsi'].iloc[-1]:.1f}", div)
        col2.metric("AI Drift (15m)", f"{drift:.5f}")
        col3.metric("Sentinel Score", f"{score}/100")

        # Generazione Segnale con Money Management
        if not is_low_liquidity():
            action = "LONG" if (score >= 65 and df_d['rsi'].iloc[-1] < 65) else "SHORT" if (score <= 35 and df_d['rsi'].iloc[-1] > 35) else None
            if action:
                atr_v = df_d['atr'].iloc[-1]
                sl = curr_price - (1.5 * atr_v) if action == "LONG" else curr_price + (1.5 * atr_v)
                tp = curr_price + (3.0 * atr_v) if action == "LONG" else curr_price - (3.0 * atr_v)
                
                risk_val = balance * (risk_pc / 100)
                pips_risk = abs(curr_price - sl) / pip_unit
                size = risk_val / (pips_risk * pip_mult) if pips_risk > 0 else 0
                
                st.success(f"🚀 {action} SIGNAL | Entry: {price_fmt.format(curr_price)} | SL: {price_fmt.format(sl)} | TP: {price_fmt.format(tp)} | SIZE: {size:.2f}")
                st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3"></audio>', unsafe_allow_html=True)

# Storico
if not st.session_state['signal_history'].empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Last Signals")
    st.sidebar.dataframe(st.session_state['signal_history'].tail(3))

time_lib.sleep(1)
st.rerun()

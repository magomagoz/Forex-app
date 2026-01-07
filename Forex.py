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
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Stato'])
if 'last_alert' not in st.session_state:
    st.session_state['last_alert'] = None

# --- 2. ASSET E MAPPING ---
asset_map = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X", "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"
}

# --- 3. FUNZIONI TECNICHE ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "**Tokyo** 🇯🇵": (time(0,0), time(9,0)), 
        "**Londra** 🇬🇧": (time(8,0), time(17,0)), 
        "**New York** 🇺🇸": (time(13,0), time(22,0))
    }
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

def is_low_liquidity():
    now_utc = datetime.now(pytz.utc).time()
    return time(23, 0) <= now_utc or now_utc <= time(1, 0)

@st.cache_data(ttl=50)
def get_realtime_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def get_currency_strength():
    try:
        forex = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURCHF=X","EURJPY=X", "GBPJPY=X", "GBPCHF=X","EURGBP=X"]
        crypto = ["BTC-USD", "ETH-USD"]
        data = yf.download(forex + crypto, period="2d", interval="1d", progress=False, timeout=15)
        if data is None or data.empty: return pd.Series(dtype=float)
        close_data = data['Close'] if 'Close' in data else data
        returns = close_data.pct_change().iloc[-1] * 100
        strength = {
            "USD 🇺🇸": (-returns.get("EURUSD=X",0) - returns.get("GBPUSD=X",0) + returns.get("USDJPY=X",0) - returns.get("AUDUSD=X",0) + returns.get("USDCAD=X",0) + returns.get("USDCHF=X",0) - returns.get("NZDUSD=X",0)) / 7,
            "EUR 🇪🇺": (returns.get("EURUSD=X",0) + returns.get("EURJPY=X",0) + returns.get("EURGBP=X",0)) / 3,
            "GBP 🇬🇧": (returns.get("GBPUSD=X",0) + returns.get("GBPJPY=X",0) - returns.get("EURGBP=X",0)) / 3,
            "JPY 🇯🇵": (-returns.get("USDJPY=X",0) - returns.get("EURJPY=X",0) - returns.get("GBPJPY=X",0)) / 3,
            "CHF 🇨🇭": (-returns.get("USDCHF=X",0) - returns.get("EURCHF=X",0) - returns.get("GBPCHF=X",0)) / 3,
            "AUD 🇦🇺": returns.get("AUDUSD=X", 0),
            "BTC ₿": returns.get("BTC-USD", 0),
            "ETH 💎": returns.get("ETH-USD", 0)
        }
        return pd.Series(strength).sort_values(ascending=False)
    except: return pd.Series(dtype=float)

def get_asset_params(pair):
    if "-" in pair: return 1.0, "{:.2f}", 1
    if "JPY" in pair: return 0.01, "{:.2f}", 1000
    return 0.0001, "{:.5f}", 10

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi_col = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi_col.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi_col.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DECRESCITA"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 CRESCITA"
    return "Neutrale"

# --- 4. MOTORI SENTINELLA ---
def update_signal_outcomes():
    if st.session_state['signal_history'].empty: return
    df = st.session_state['signal_history']
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            data = yf.download(asset_map[row['Asset']], period="1d", interval="5m", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                high, low = data['High'].max(), data['Low'].min()
                if row['Direzione'] == 'COMPRA':
                    if high >= row['TP']: df.at[idx, 'Stato'] = '✅ TARGET'
                    elif low <= row['SL']: df.at[idx, 'Stato'] = '❌ STOP LOSS'
                else:
                    if low <= row['TP']: df.at[idx, 'Stato'] = '✅ TARGET'
                    elif high >= row['SL']: df.at[idx, 'Stato'] = '❌ STOP LOSS'
        except: continue

def run_sentinel():
    for label, ticker in asset_map.items():
        try:
            df_rt = get_realtime_data(ticker)
            df_d = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df_rt is None or df_rt.empty or df_d.empty: continue
            
            if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
            df_d.columns = [c.lower() for c in df_d.columns]
            
            # Calcolo RSI e ATR con nomi variabili richiesti
            df_d['rsi'] = ta.rsi(df_d['close'], length=14)
            df_d['atr'] = ta.atr(df_d['high'], df_d['low'], df_d['close'], length=14)
            last_rsi = float(df_d['rsi'].iloc[-1])
            last_atr = float(df_d['atr'].iloc[-1])
            
            bb = ta.bbands(df_rt['close'], length=20, std=2)
            curr_price = float(df_rt['close'].iloc[-1])
            lower_bb, upper_bb = bb.iloc[-1, 0], bb.iloc[-1, 2]
            
            action = None
            if curr_price < lower_bb and last_rsi < 40: action = "COMPRA"
            elif curr_price > upper_bb and last_rsi > 60: action = "VENDI"
            
            if action:
                history = st.session_state['signal_history']
                if history.empty or not ((history['Asset'] == label) & (history['Direzione'] == action)).tail(1).any():
                    pip_unit, price_fmt, _ = get_asset_params(ticker)
                    sl = curr_price - (1.5 * last_atr) if action == "COMPRA" else curr_price + (1.5 * last_atr)
                    tp = curr_price + (3 * last_atr) if action == "COMPRA" else curr_price - (3 * last_atr)
                    
                    new_sig = {
                        'DataOra': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'Asset': label, 'Direzione': action, 'Prezzo': curr_price,
                        'SL': sl, 'TP': tp, 'Stato': 'In Corso'
                    }
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), history], ignore_index=True)
                    st.session_state['last_alert'] = new_sig
                    st.rerun()
        except: continue

# --- 5. POPUP A TUTTO SCHERMO ---
if st.session_state['last_alert']:
    alert = st.session_state['last_alert']
    st.markdown(f"""
        <div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-color: rgba(0,0,0,0.95); z-index: 9999; display: flex; flex-direction: column; justify-content: center; align-items: center; color: white; text-align: center;">
            <h1 style="font-size: 5em; color: #00ffcc;">🚀 NUOVO SEGNALE</h1>
            <h2 style="font-size: 4em;">{alert['Asset']} - {alert['Direzione']}</h2>
            <p style="font-size: 2em;">Prezzo: {alert['Prezzo']:.5f}</p>
            <p style="font-size: 1.5em; color: gray;">{alert['DataOra']}</p>
            <br>
        </div>
    """, unsafe_allow_html=True)
    if st.button("✅ CHIUDI AVVISO E TORNA AL MONITOR", use_container_width=True):
        st.session_state['last_alert'] = None
        st.rerun()
    st.stop()

# --- 6. SIDEBAR & HEADER ---
st.sidebar.header("🛠 Trading Desk (5m)")
selected_label = st.sidebar.selectbox("**Asset Visualizzato**", list(asset_map.keys()))
pair = asset_map[selected_label]
balance = st.sidebar.number_input("**Balance (€)**", value=1000)
risk_pc = st.sidebar.slider("**Rischio %**", 0.5, 5.0, 1.0)

if st.sidebar.button("🗑️ Reset Cronologia"):
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Stato'])
    st.rerun()

st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1></div>', unsafe_allow_html=True)

# --- 7. ESECUZIONE LOGICA ---
update_signal_outcomes()
run_sentinel()

# --- 8. GRAFICO E ANALISI ASSET SELEZIONATO ---
df_rt = get_realtime_data(pair)
df_d = yf.download(pair, period="1y", interval="1d", progress=False)

if df_rt is not None and not df_rt.empty:
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df_rt.index[-60:], open=df_rt['open'][-60:], high=df_rt['high'][-60:], low=df_rt['low'][-60:], close=df_rt['close'][-60:], name='Price'))
    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 9. STRENGTH METER & CRONOLOGIA ---
st.subheader("⚡ Market Strength Meter")
s_data = get_currency_strength()
if not s_data.empty:
    cols = st.columns(len(s_data))
    for i, (curr, val) in enumerate(s_data.items()):
        cols[i].metric(curr, f"{val:.2f}%")

st.markdown("---")
st.subheader("📜 Cronologia Segnali")
if not st.session_state['signal_history'].empty:
    st.dataframe(st.session_state['signal_history'], use_container_width=True)
else:
    st.info("Nessun segnale rilevato.")

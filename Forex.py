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
st.set_page_config(page_title="Forex Momentum Pro AI (M5)", layout="wide", page_icon="📈")

# Refresh globale ogni 60 secondi
st_autorefresh(interval=60 * 1000, key="sentinel_refresh")

if 'signal_history' not in st.session_state:
    st.session_state['signal_history'] = pd.DataFrame(columns=['Orario', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP'])

# --- 2. FUNZIONI TECNICHE COMPLETE ---
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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except:
        return None

def get_currency_strength():
    try:
        tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURCHF=X","EURJPY=X", "GBPJPY=X", "GBPCHF=X","EURGBP=X"]
        data = yf.download(tickers, period="2d", interval="1d", progress=False, timeout=15)
        if data is None or data.empty: return pd.Series(dtype=float)
        
        if isinstance(data.columns, pd.MultiIndex):
            close_data = data['Close']
        else:
            close_data = data

        returns = close_data.pct_change().iloc[-1] * 100
        strength = {
            "USD 🇺🇸": (-returns.get("EURUSD=X",0) - returns.get("GBPUSD=X",0) + returns.get("USDJPY=X",0) - returns.get("AUDUSD=X",0) + returns.get("USDCAD=X",0) + returns.get("USDCHF=X",0) - returns.get("NZDUSD=X",0)) / 7,
            "EUR 🇪🇺": (returns.get("EURUSD=X",0) + returns.get("EURJPY=X",0) + returns.get("EURGBP=X",0)) / 3,
            "GBP 🇬🇧": (returns.get("GBPUSD=X",0) + returns.get("GBPJPY=X",0) - returns.get("EURGBP=X",0)) / 3,
            "JPY 🇯🇵": (-returns.get("USDJPY=X",0) - returns.get("EURJPY=X",0) - returns.get("GBPJPY=X",0)) / 3,
            "CHF 🇨🇭": (-returns.get("USDCHF=X",0) - returns.get("EURCHF=X",0) - returns.get("GBPCHF=X",0)) / 3,
            "AUD 🇦🇺": returns.get("AUDUSD=X", 0),
            "CAD 🇨🇦": -returns.get("USDCAD=X", 0),
        }
        return pd.Series(strength).sort_values(ascending=False)
    except:
        return pd.Series(dtype=float)

def get_asset_params(pair):
    if "-" in pair: return 1.0, "{:.2f}", 1, "CRYPTO"
    if "JPY" in pair: return 0.01, "{:.2f}", 1000, "FOREX" 
    return 0.0001, "{:.5f}", 10, "FOREX"

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DIV. BEARISH"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 DIV. BULLISH"
    return "Neutrale"

# --- 3. SIDEBAR & TIMER ---
st.sidebar.header("🛠 Trading Desk (M5)")
if "start_time" not in st.session_state: st.session_state.start_time = time_lib.time()
countdown = 60 - int(time_lib.time() - st.session_state.start_time) % 60
st.sidebar.metric("⏳ **Prossimo Scan**" f"{countdown}s")

pair = st.sidebar.selectbox("**Asset**", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "BTC-USD", "ETH-USD"])
balance = st.sidebar.number_input("**Balance Conto (€)**", value=1000)
risk_pc = st.sidebar.slider("**Rischio %**", 0.5, 5.0, 1.0)

if st.sidebar.button("🔄 **AGGIORNAMENTO**"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("🌍 **Sessioni di Mercato**")
for s, op in get_session_status().items():
    st.sidebar.markdown(f"**{s}**: {'🟢 OPEN' if op else '🔴 CLOSED'}")

# --- 4. BANNER ---
st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1><p style="color: white; opacity: 0.8;">Sentinel AI • Forex & Crypto Analysis</p></div>', unsafe_allow_html=True)

# --- 5. DATA ENGINE ---
pip_unit, price_fmt, pip_mult, asset_type = get_asset_params(pair)
df_rt = get_realtime_data(pair)
df_d = yf.download(pair, period="1y", interval="1d", progress=False)

if df_rt is not None and not df_rt.empty:
    # Calcolo Bollinger con protezione dinamica (Fix USDCHF/BTC)
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)

    try:
        col_upper = [c for c in df_rt.columns if "BBU" in c.upper()][0]
        col_mid = [c for c in df_rt.columns if "BBM" in c.upper()][0]
        col_lower = [c for c in df_rt.columns if "BBL" in c.upper()][0]
    except:
        st.error("Errore indicatori tecnici.")
        st.stop()
    
    # Grafico Candlestick
    st.subheader(f"📈 Chart 5 Minuti: {pair}")
    plot_df = df_rt.tail(60)
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col_upper], line=dict(color='rgba(173, 216, 230, 0.4)'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col_mid], line=dict(color='gray', dash='dash'), name='Mid BB'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col_lower], line=dict(color='rgba(173, 216, 230, 0.4)'), fill='tonexty', name='Lower BB'))
    
    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    curr_price = float(df_rt['close'].iloc[-1])
    st.metric("Prezzo Live", price_fmt.format(curr_price))

    # --- STRENGTH METER ---
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    s_data = get_currency_strength()      
    if not s_data.empty:
        cols = st.columns(len(s_data))
        for i, (curr, val) in enumerate(s_data.items()):
            bg = "#006400" if val > 0.15 else "#8B0000" if val < -0.15 else "#333333"
            txt = "#00FFCC" if val > 0.15 else "#FF4B4B" if val < -0.15 else "#FFFFFF"
            cols[i].markdown(f"<div style='text-align:center; background:{bg}; padding:10px; border-radius:10px; border:1px solid {txt};'><b style='color:white;'>{curr}</b><br><span style='color:{txt}; font-weight:bold;'>{val:.2f}%</span></div>", unsafe_allow_html=True)
    
    # --- ANALISI AI ---
    if df_d is not None and not df_d.empty:
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
        df_d.columns = [c.lower() for c in df_d.columns]
        df_d['rsi'] = ta.rsi(df_d['close'], length=14)
        df_d['atr'] = ta.atr(df_d['high'], df_d['low'], df_d['close'], length=14)
        
        last_rsi = float(df_d['rsi'].iloc[-1])
        last_atr = float(df_d['atr'].iloc[-1])
        div_sig = detect_divergence(df_d)

        # Inerzia AI (15 candele M5 = 75 min)
        lookback = 15
        y_vals = df_rt['close'].tail(lookback).values
        x_vals = np.arange(len(y_vals)).reshape(-1, 1)
        model = LinearRegression().fit(x_vals, y_vals)
        drift = model.predict([[lookback]])[0] - curr_price
        
        score = 50
        if curr_price < df_rt[col_lower].iloc[-1]: score += 20
        if curr_price > df_rt[col_upper].iloc[-1]: score -= 20
        
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("RSI Daily", f"{last_rsi:.1f}", div_sig)
        c2.metric("Inerzia AI (75m)", f"{drift:.5f}")
        c3.metric("Sentinel Score", f"{score}/100")

        # LOGICA SEGNALI (Fix Doppi)
        if not is_low_liquidity():
            action = "LONG" if (score >= 65 and last_rsi < 60) else "SHORT" if (score <= 35 and last_rsi > 40) else None
            
            # Anti-duplicazione
            last_action = st.session_state['signal_history'].iloc[-1]['Direzione'] if not st.session_state['signal_history'].empty else None
            last_asset = st.session_state['signal_history'].iloc[-1]['Asset'] if not st.session_state['signal_history'].empty else None

            if action and (action != last_action or pair != last_asset):
                sl = curr_price - (1.5 * last_atr) if action == "LONG" else curr_price + (1.5 * last_atr)
                tp = curr_price + (3 * last_atr) if action == "LONG" else curr_price - (3 * last_atr)
                
                risk_cash = balance * (risk_pc / 100)
                dist_pips = abs(curr_price - sl) / pip_unit
                lotti = risk_cash / (dist_pips * pip_mult) if dist_pips > 0 else 0
                
                color = "#00ffcc" if action == "LONG" else "#ff4b4b"
                st.markdown(f"""
                    <div style="border: 2px solid {color}; padding: 20px; border-radius: 15px; background: #0e1117;">
                        <h2 style="color: {color}; margin-top:0;">🚀 SEGNALE: {action}</h2>
                        <p>Entry: {price_fmt.format(curr_price)} | SL: {price_fmt.format(sl)} | TP: {price_fmt.format(tp)}</p>
                        <p style="color:#ffcc00; font-weight:bold;">LOTTI: {lotti:.2f}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                new_sig = pd.DataFrame([{
                    'Orario': datetime.now().strftime("%H:%M:%S"), 
                    'Asset': str(pair), 
                    'Direzione': str(action), 
                    'Prezzo': float(curr_price), 
                    'SL': float(sl), 
                    'TP': float(tp)
                }])
                st.session_state['signal_history'] = pd.concat([st.session_state['signal_history'], new_sig], ignore_index=True)

# Registro Storico
if not st.session_state['signal_history'].empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📜 Storico Segnali")
    st.sidebar.dataframe(st.session_state['signal_history'].tail(5))

time_lib.sleep(1)
st.rerun()

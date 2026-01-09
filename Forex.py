import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, time
import pytz
import time as time_lib
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CONFIGURAZIONE & STILE ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

# CSS per eliminare spazi bianchi in alto
st.markdown("""
    <style>
        .block-container {padding-top: 1rem !important;}
        [data-testid="stSidebar"] > div:first-child {padding-top: 0rem !important;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

rome_tz = pytz.timezone('Europe/Rome')
st_autorefresh(interval=60 * 1000, key="sentinel_refresh")

# Inizializzazione Session State
if 'signal_history' not in st.session_state: 
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
if 'last_alert' not in st.session_state:
    st.session_state['last_alert'] = None
if 'last_scan_status' not in st.session_state:
    st.session_state['last_scan_status'] = "In attesa..."

# --- 2. FUNZIONI TECNICHE ---
def get_now_rome():
    return datetime.now(rome_tz)

def play_notification_sound():
    audio_html = """<audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg"></audio>"""
    st.markdown(audio_html, unsafe_allow_html=True)

def get_session_status():
    now_rome = get_now_rome().time()
    sessions = {"Tokyo 🇯🇵": (time(0,0), time(9,0)), "Londra 🇬🇧": (time(9,0), time(18,0)), "New York 🇺🇸": (time(14,0), time(23,0))}
    return {name: start <= now_rome <= end for name, (start, end) in sessions.items()}

def is_low_liquidity():
    now_rome = get_now_rome().time()
    return time(23, 0) <= now_rome or now_rome <= time(1, 0)

@st.cache_data(ttl=30)
def get_realtime_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def get_asset_params(pair):
    if "-" in pair: return 1.0, "{:.2f}", 1
    if "JPY" in pair: return 0.01, "{:.2f}", 1000
    return 0.0001, "{:.5f}", 10

# --- 3. MOTORE SENTINEL ---
asset_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "BTC-USD": "BTC-USD"}

def run_sentinel():
    for label, ticker in asset_map.items():
        try:
            df_rt_s = yf.download(ticker, period="2d", interval="5m", progress=False)
            if df_rt_s.empty: continue
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            df_rt_s.columns = [c.lower() for c in df_rt_s.columns]

            bb_s = ta.bbands(df_rt_s['close'], length=20, std=2)
            rsi_s = ta.rsi(df_rt_s['close'], length=14).iloc[-1]
            
            c_l = [c for c in bb_s.columns if "BBL" in c.upper()][0]
            c_u = [c for c in bb_s.columns if "BBU" in c.upper()][0]
            
            curr_v = float(df_rt_s['close'].iloc[-1])
            l_bb = float(bb_s[c_l].iloc[-1])
            u_bb = float(bb_s[c_u].iloc[-1])
            
            s_action = None
            if curr_v < l_bb and rsi_s < 35: s_action = "COMPRA"
            elif curr_v > u_bb and rsi_s > 65: s_action = "VENDI"
            
            if s_action:
                hist = st.session_state['signal_history']
                if hist.empty or not ((hist['Asset'] == label) & (hist['DataOra'].str.contains(get_now_rome().strftime("%H:%M")))).any():
                    p_unit, p_fmt, p_mult = get_asset_params(ticker)
                    atr = ta.atr(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14).iloc[-1]
                    sl = curr_v - (1.5 * atr) if s_action == "COMPRA" else curr_v + (1.5 * atr)
                    tp = curr_v + (3 * atr) if s_action == "COMPRA" else curr_v - (3 * atr)
                    
                    new_sig = {'DataOra': get_now_rome().strftime("%d/%m %H:%M:%S"), 'Asset': label, 'Direzione': s_action, 'Prezzo': p_fmt.format(curr_v), 'SL': p_fmt.format(sl), 'TP': p_fmt.format(tp), 'Size': "0.10", 'Stato': 'In Corso'}
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    st.session_state['last_alert'] = new_sig
                    st.rerun()
            st.session_state['last_scan_status'] = f"📡 {label}: Analisi OK"
        except: continue

# --- 4. SIDEBAR ---
st.sidebar.header("🛠 Trading Desk")
run_sentinel()

status = st.session_state['last_scan_status']
if "🚀" in status: st.sidebar.success(status)
else: st.sidebar.info(status)

selected_label = st.sidebar.selectbox("Asset", list(asset_map.keys()))
balance = st.sidebar.number_input("Conto (€)", value=1000)

if st.sidebar.button("🗑️ Reset Cronologia"):
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
    st.rerun()

# --- 5. POPUP ALERT ---
if st.session_state['last_alert']:
    play_notification_sound()
    alert = st.session_state['last_alert']
    st.markdown(f"""
        <div style="position: fixed; top: 40%; left: 50%; transform: translate(-50%, -50%); width: 80vw; max-width: 600px; background: rgba(10,10,30,0.95); z-index: 9999; padding: 25px; border: 3px solid #00ffcc; border-radius: 20px; text-align: center; color: white;">
            <h2 style="color: #00ffcc;">🚀 NUOVO SEGNALE: {alert['Asset']}</h2>
            <h1 style="font-size: 3em;">{alert['Direzione']}</h1>
            <p>Prezzo: {alert['Prezzo']} | SL: {alert['SL']} | TP: {alert['TP']}</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("✅ CONFERMA E CHIUDI", use_container_width=True):
        st.session_state['last_alert'] = None
        st.rerun()

# --- 6. GRAFICO ---
st.title("📊 Forex Momentum Pro AI")
df_rt = get_realtime_data(asset_map[selected_label])

if df_rt is not None:
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    df_rt['rsi'] = ta.rsi(df_rt['close'], length=14)
    
    c_u = [c for c in bb.columns if "BBU" in c.upper()][0]
    c_m = [c for c in bb.columns if "BBM" in c.upper()][0]
    c_l = [c for c in bb.columns if "BBL" in c.upper()][0]
    
    p_df = df_rt.tail(50)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # 1. Upper BB (Solo Linea)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_u], line=dict(color='rgba(173,216,230,0.3)', width=1), name='Upper BB'), row=1, col=1)
    # 2. Middle BB
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_m], line=dict(color='white', width=1, dash='dash'), name='Middle BB'), row=1, col=1)
    # 3. Lower BB (Riempimento verso Middle)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_l], line=dict(color='rgba(173,216,230,0.3)', width=1), fill='tonexty', fillcolor='rgba(173,216,230,0.1)', name='Lower BB'), row=1, col=1)
    # 4. Candele
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['open'], high=p_df['high'], low=p_df['low'], close=p_df['close'], name='Price'), row=1, col=1)

    # RIGHE VERTICALI OGNI 5 MINUTI
    v_lines = pd.date_range(start=p_df.index.min(), end=p_df.index.max(), freq='5min')
    for lt in v_lines:
        fig.add_vline(x=lt.timestamp()*1000, line_width=0.5, line_dash="dot", line_color="rgba(255,255,255,0.1)")
        fig.add_annotation(x=lt, y=-0.15, xref="x", yref="paper", text=lt.strftime('%H:%M'), showarrow=False, font=dict(size=9, color="gray"))

    # RSI
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['rsi'], line=dict(color='#ffcc00'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_color="green", row=2, col=1)

    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=50))
    st.plotly_chart(fig, use_container_width=True)

# --- 7. CRONOLOGIA ---
st.markdown("---")
st.subheader("📜 Cronologia Segnali")
if not st.session_state['signal_history'].empty:
    st.dataframe(st.session_state['signal_history'], use_container_width=True, hide_index=True)
    csv = st.session_state['signal_history'].to_csv(index=False).encode('utf-8')
    st.download_button("📥 Scarica Report CSV", data=csv, file_name="segnali_forex.csv", mime="text/csv")
else:
    st.info("In attesa di segnali...")

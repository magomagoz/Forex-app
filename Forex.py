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
    
# --- 2. FUNZIONI TECNICHE ---
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
    if "-" in pair: return 1.0, "{:.2f}", 1, "CRYPTO"
    if "JPY" in pair: return 0.01, "{:.2f}", 1000, "FOREX" 
    return 0.0001, "{:.5f}", 10, "FOREX"

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi_col = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi_col.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi_col.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DECRESCITA"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 CRESCITA"
    return "Neutrale"

# --- 3. LOGICA DI MONITORAGGIO ---
def update_signal_outcomes():
    if st.session_state['signal_history'].empty: return
    df = st.session_state['signal_history']
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            data = yf.download(asset_map[row['Asset']], period="1d", interval="5m", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                high, low = data['High'].max(), data['Low'].min()
                sl_val, tp_val = float(row['SL']), float(row['TP'])
                if row['Direzione'] == 'COMPRA':
                    if high >= tp_val: df.at[idx, 'Stato'] = '✅ TARGET'
                    elif low <= sl_val: df.at[idx, 'Stato'] = '❌ STOP LOSS'
                else:
                    if low <= tp_val: df.at[idx, 'Stato'] = '✅ TARGET'
                    elif high >= sl_val: df.at[idx, 'Stato'] = '❌ STOP LOSS'
        except: continue

def run_sentinel():
    for label, ticker in asset_map.items():
        try:
            df_rt_scan = yf.download(ticker, period="2d", interval="5m", progress=False)
            df_d_scan = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df_rt_scan.empty or df_d_scan.empty: continue
            if isinstance(df_rt_scan.columns, pd.MultiIndex): df_rt_scan.columns = df_rt_scan.columns.get_level_values(0)
            if isinstance(df_d_scan.columns, pd.MultiIndex): df_d_scan.columns = df_d_scan.columns.get_level_values(0)
            df_rt_scan.columns = [c.lower() for c in df_rt_scan.columns]; df_d_scan.columns = [c.lower() for c in df_d_scan.columns]
            
            bb_scan = ta.bbands(df_rt_scan['close'], length=20, std=2)
            close_val, lower_bb, upper_bb = float(df_rt_scan['close'].iloc[-1]), float(bb_scan.iloc[-1, 0]), float(bb_scan.iloc[-1, 2])
            rsi_scan = ta.rsi(df_d_scan['close'], length=14).iloc[-1]
            atr_scan = ta.atr(df_d_scan['high'], df_d_scan['low'], df_d_scan['close'], length=14).iloc[-1]
            
            scan_action = None
            if close_val < lower_bb and rsi_scan < 40: scan_action = "COMPRA"
            elif close_val > upper_bb and rsi_scan > 60: scan_action = "VENDI"
            
            if scan_action:
                history = st.session_state['signal_history']
                if history.empty or not ((history['Asset'] == label) & (history['Direzione'] == scan_action)).head(1).any():
                    _, p_fmt, _ = get_asset_params(ticker)
                    sl = close_val - (1.5 * atr_scan) if scan_action == "COMPRA" else close_val + (1.5 * atr_scan)
                    tp = close_val + (3 * atr_scan) if scan_action == "COMPRA" else close_val - (3 * atr_scan)
                    new_sig = {'DataOra': datetime.now().strftime("%d/%m %H:%M:%S"), 'Asset': label, 'Direzione': scan_action, 'Prezzo': p_fmt.format(close_val), 'SL': p_fmt.format(sl), 'TP': p_fmt.format(tp), 'Stato': 'In Corso'}
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), history], ignore_index=True)
                    st.session_state['last_alert'] = new_sig
                    st.rerun()
        except: continue

# --- 4. SIDEBAR ---
st.sidebar.header("🛠 Trading Desk (5m)")
asset_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X", "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"}
selected_label = st.sidebar.selectbox("**Asset**", list(asset_map.keys()))
pair = asset_map[selected_label]
balance = st.sidebar.number_input("**Balance (€)**", value=1000)
risk_pc = st.sidebar.slider("**Rischio %**", 0.5, 5.0, 1.0)

if st.sidebar.button("🗑️ Reset Cronologia"):
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Stato'])
    st.rerun()

st.sidebar.subheader("🌍 **Sessioni**")
for s, op in get_session_status().items():
    st.sidebar.markdown(f"**{s}**: {'🟢 OPEN' if op else '🔴 CLOSED'}")

# --- 5. POPUP ---
if st.session_state['last_alert']:
    alert = st.session_state['last_alert']
    st.markdown(f"""<div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background-color: rgba(0,0,0,0.95); z-index: 999999; display: flex; flex-direction: column; justify-content: center; align-items: center; color: white; text-align: center; padding: 20px;"><h1 style="font-size: 4em; color: #00ffcc;">🚀 NUOVO SEGNALE</h1><h2 style="font-size: 3em;">{alert['Asset']} - {alert['Direzione']}</h2><p style="font-size: 2em;">Prezzo: {alert['Prezzo']}</p><p style="font-size: 1.2em; color: gray;">{alert['DataOra']}</p></div>""", unsafe_allow_html=True)
    if st.button("✅ CHIUDI AVVISO"):
        st.session_state['last_alert'] = None
        st.rerun()
    st.stop()

# --- 6. MAIN UI & GRAFICO ---
st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1></div>', unsafe_allow_html=True)

pip_unit, price_fmt, pip_mult, asset_type = get_asset_params(pair)
df_rt = get_realtime_data(pair)
df_d = yf.download(pair, period="1y", interval="1d", progress=False)

if df_rt is not None and not df_rt.empty:
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    col_upper, col_mid, col_lower = [c for c in df_rt.columns if "BBU" in c.upper()][0], [c for c in df_rt.columns if "BBM" in c.upper()][0], [c for c in df_rt.columns if "BBL" in c.upper()][0]
    
    st.subheader(f"📈 Chart: {selected_label}")
    plot_df = df_rt.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='Price'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col_upper], line=dict(color='rgba(173, 216, 230, 0.4)'), name='Upper BB'))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col_lower], line=dict(color='rgba(173, 216, 230, 0.4)'), fill='tonexty', name='Lower BB'))
    fig.update_layout(height=450, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True)
    
    curr_price = float(df_rt['close'].iloc[-1])
    st.metric(f"Prezzo {selected_label}", price_fmt.format(curr_price))

# --- 7. ANALISI AI & LOGICA SEGNALI ---
if df_rt is not None and df_d is not None and not df_d.empty:
    if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
    df_d.columns = [c.lower() for c in df_d.columns]
    df_d['rsi'] = ta.rsi(df_d['close'], length=14); df_d['atr'] = ta.atr(df_d['high'], df_d['low'], df_d['close'], length=14)
    rsi_val, last_atr = float(df_d['rsi'].iloc[-1]), float(df_d['atr'].iloc[-1])
    
    y_vals = df_rt['close'].tail(15).values; x_vals = np.arange(len(y_vals)).reshape(-1, 1)
    model = LinearRegression().fit(x_vals, y_vals); drift = model.predict([[15]])[0] - curr_price
    
    score = 50
    if curr_price < df_rt[col_lower].iloc[-1]: score += 20
    elif curr_price > df_rt[col_upper].iloc[-1]: score -= 20
    
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI Daily", f"{rsi_val:.1f}", detect_divergence(df_d))
    c2.metric("Inerzia AI (75m)", f"{drift:.5f}")
    c3.metric("Sentinel Score", f"{score}/100")

    if not is_low_liquidity():
        action = "COMPRA" if (score >= 65 and rsi_val < 60) else "VENDI" if (score <= 35 and rsi_val > 40) else None
        if action:
            history = st.session_state['signal_history']
            if history.empty or not ((history['Asset'] == selected_label) & (history['Direzione'] == action)).head(1).any():
                sl = curr_price - (1.5 * last_atr) if action == "COMPRA" else curr_price + (1.5 * last_atr)
                tp = curr_price + (3 * last_atr) if action == "COMPRA" else curr_price - (3 * last_atr)
                new_alert = {'DataOra': datetime.now().strftime("%d/%m %H:%M:%S"), 'Asset': selected_label, 'Direzione': action, 'Prezzo': price_fmt.format(curr_price), 'SL': price_fmt.format(sl), 'TP': price_fmt.format(tp), 'Stato': 'In Corso'}
                st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_alert]), history], ignore_index=True)
                st.session_state['last_alert'] = new_alert
                st.rerun()

# --- 8. MOTORI & CRONOLOGIA ---
update_signal_outcomes()
run_sentinel()

st.markdown("---")
st.subheader("⚡ Market Strength Meter")
s_data = get_currency_strength()
if not s_data.empty:
    cols = st.columns(len(s_data))
    for i, (curr, val) in enumerate(s_data.items()):
        cols[i].metric(curr, f"{val:.2f}%")

st.markdown("---")
st.subheader("📜 Cronologia Segnali")
if not st.session_state['signal_history'].empty:
    def style_status(val):
        color = '#ffcc00'
        if '✅' in val: color = '#00ffcc'
        elif '❌' in val: color = '#ff4b4b'
        return f'color: {color}; font-weight: bold'
    st.dataframe(st.session_state['signal_history'].style.applymap(style_status, subset=['Stato']), use_container_width=True)
else:
    st.info("In attesa di segnali...")

st.caption(f"Ultimo Check: {datetime.now().strftime('%H:%M:%S')} | Monitor attivo su 9 Asset.")

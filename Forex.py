import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
import seaborn as sns
import matplotlib.pyplot as plt
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
            AI-Driven Market Analysis & Sentinel System
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 2. FUNZIONI DI SUPPORTO ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "Tokyo 🇯🇵": (time(0,0), time(9,0)), 
        "Londra 🇬🇧": (time(8,0), time(17,0)), 
        "New York 🇺🇸": (time(13,0), time(22,0))
    }
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

if 'prediction_log' not in st.session_state:
    st.session_state['prediction_log'] = None

@st.cache_data(ttl=600)
def get_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty: return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
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

def detect_divergence(df):
    if len(df) < 20: return "Dati Insufficienti"
    price, rsi = df['Close'], df['RSI']
    curr_p, curr_r = price.iloc[-1], rsi.iloc[-1]
    prev_max_p, prev_max_r = price.iloc[-15:-1].max(), rsi.iloc[-15:-1].max()
    prev_min_p, prev_min_r = price.iloc[-15:-1].min(), rsi.iloc[-15:-1].min()
    
    if curr_p > prev_max_p and curr_r < prev_max_r and curr_r > 50:
        return "📉 BEARISH"
    elif curr_p < prev_min_p and curr_r > prev_min_r and curr_r < 50:
        return "📈 BULLISH"
    return "Neutrale"

def get_pip_value(pair):
    if "JPY" in pair: return 0.01, "{:.2f}"
    return 0.0001, "{:.4f}"

# --- 3. SIDEBAR ---
st.sidebar.header("🕹 Control Panel")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)
pip_unit, price_fmt = get_pip_value(pair)

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 4. HEADER ---
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title(f"Analisi & Previsione: {pair}")
    st.caption(f"Ultimo Check: {datetime.now().strftime('%H:%M:%S')}")
with col_h2:
    if st.button("🔄 AGGIORNA"):
        st.cache_data.clear()
        st.rerun()

# --- 5. LOGICA PRINCIPALE ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
	
    # --- 6. ⚡ CURRENCY STRENGTH METER (CORREZIONE RIGA 139) ---
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    strength_data = get_currency_strength()
    
    cols = st.columns(6)
    # Correzione: usiamo l'indice della Series direttamente
    for i, curr_name in enumerate(strength_data.index[:6]):
        val = strength_data[curr_name]
        color_val = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b>{curr_name}</b><br><span style='color:{color_val}; font-size:18px;'>{val:.2f}%</span></div>", unsafe_allow_html=True)
    
    # --- ANALISI VOLATILITÀ ---
    st.subheader("🌋 Analisi Volatilità & Squeeze")
    std = df_d['Close'].rolling(20).std()
    ma = df_d['Close'].rolling(20).mean()
    upper_bb = ma + (2 * std)
    lower_bb = ma - (2 * std)
    atr_val = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=20)
    upper_kc = ma + (1.5 * atr_val)
    lower_kc = ma - (1.5 * atr_val)
    is_sqz = (upper_bb.iloc[-1] < upper_kc.iloc[-1])
    
    if is_sqz:
        st.warning("⚠️ SQUEEZE ATTIVO: Compressione dei prezzi in corso.")
    else:
        st.success("🚀 RELEASE: Il prezzo è in fase di espansione.")

    # --- 7. MODELLO PREDITTIVO AI ---
    st.markdown("---")
    st.subheader("🔮 Modello Predittivo AI (+1h)")
    drift = 0.0
    if len(df_h) > 24:
        recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
        model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
        pred = model.predict(np.array([[24]]))[0][0]
        curr_price = recent_h[-1][0]
        drift = pred - curr_price
        
        cp1, cp2 = st.columns(2)
        cp1.metric("Prezzo Attuale", price_fmt.format(curr_price))
        cp2.metric("Previsione +1h", price_fmt.format(pred), f"{drift:.5f}")
        
    st.line_chart(df_h['Close'].tail(50))

    # --- 8. SETUP OPERATIVO ---
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    df_d['ADX'] = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])['ADX_14']
    
    last_c = df_d['Close'].iloc[-1]
    last_rsi = df_d['RSI'].iloc[-1]
    last_adx = df_d['ADX'].iloc[-1]
    last_atr = df_d['ATR'].iloc[-1]
    div_signal = detect_divergence(df_d)
    
    # --- 9. ORACOLO E SENTINEL ---
    st.subheader("📊 Valutazione Oracle (Confluenza)")
    final_score = 50
    reasons = []
    
    if drift > (pip_unit * 2): 
        final_score += 15; reasons.append("AI Bullish")
    elif drift < -(pip_unit * 2):
        final_score -= 15; reasons.append("AI Bearish")

    if strength_data.index[0] in pair[:3]:
        final_score += 20; reasons.append(f"{pair[:3]} Strong")
    elif strength_data.index[-1] in pair[:3]:
        final_score -= 20; reasons.append(f"{pair[:3]} Weak")

    st.metric("Confluence Score", f"{final_score}/100")
    st.write(f"Motivazioni: {', '.join(reasons)}")

    if final_score >= 80 or final_score <= 20:
        alert_col = "#00ffcc" if final_score >= 80 else "#ff4b4b"
        st.markdown(f"""
            <div style="background-color: {alert_col}; color: black; padding: 25px; border-radius: 15px; text-align: center; font-weight: bold; animation: blinker 1s linear infinite;">
                <h1>🚀 SEGNALE SENTINEL ATTIVO: {final_score}/100</h1>
            </div>
            <style> @keyframes blinker {{ 50% {{ opacity: 0.5; }} }} </style>
            <audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>
        """, unsafe_allow_html=True)

else:
    st.error("Dati non disponibili o mercato chiuso.")

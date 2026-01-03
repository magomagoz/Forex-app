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

# Refresh ogni 120 secondi
st_autorefresh(interval=120 * 1000, key="sentinel_refresh")

# Inizializzazione Session State
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

@st.cache_data(ttl=110)
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
            "AUD 🇦🇺": (returns.get("AUDUSD=X", 0)),
            "CAD 🇨🇦": (-returns.get("USDCAD=X", 0)),
        }
        return pd.Series(strength).sort_values(ascending=False)
    except:
        return pd.Series()

def get_pip_info(pair):
    if "JPY" in pair: return 0.01, "{:.2f}", 1000 
    return 0.0001, "{:.4f}", 10

def detect_divergence(df):
    if len(df) < 20: return "Dati Insufficienti"
    price, rsi = df['Close'], df['RSI']
    curr_p, curr_r = price.iloc[-1], rsi.iloc[-1]
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 BEARISH (Div. Negativa)"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 BULLISH (Div. Positiva)"
    return "Neutrale"

# --- 3. SIDEBAR & TIMER ---
st.sidebar.header("🕹 Trading Desk")
timer_placeholder = st.sidebar.empty()
timer_placeholder.warning("Aggiornamento tra 120s")

pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)

if st.sidebar.button("🔄 AGGIORNA DATI"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    st.sidebar.markdown(f"**{s}**: {'🟢 OPEN' if op else '🔴 CLOSED'}")

# --- 4. BANNER ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', Courier, monospace; letter-spacing: 5px; margin: 0;">📊 FOREX MOMENTUM PRO</h1>
        <p style="color: white; font-size: 14px; opacity: 0.8; margin: 5px 0 0 0;">Sentinel System • AI Drift Engine • Visual Analytics v5.5</p>
    </div>
""", unsafe_allow_html=True)

# --- 5. DATA FETCH ---
pip_unit, price_fmt, pip_mult = get_pip_info(pair)
df_h = get_market_data(pair, "2d", "5m") # Per Grafico Intraday aggiornato
df_d = get_market_data(pair, "1y", "1d") # Per Analisi Daily e Indicatori

if df_h is not None and not df_h.empty and df_d is not None:
    # --- GRAFICO DAILY LIVE ---
    st.subheader(f"📈 Grafico Intraday Aggiornato: {pair}")
    st.area_chart(df_h['Close'].tail(150))
    
    prezzo_attuale = df_h['Close'].iloc[-1]
    diff = prezzo_attuale - df_h['Close'].iloc[-2]
    st.metric("Ultimo Prezzo", price_fmt.format(prezzo_attuale), f"{diff:.5f}")

    # --- STRENGTH METER ---
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    strength_data = get_currency_strength()
    if not strength_data.empty:
        cols = st.columns(6)
        for i, (curr, val) in enumerate(strength_data.items()):
            if i < 6:
                col_c = "#00ffcc" if val > 0 else "#ff4b4b"
                cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b style='color:white;'>{curr}</b><br><span style='color:{col_c}; font-weight:bold;'>{val:.2f}%</span></div>", unsafe_allow_html=True)

    # --- AI PREDICTION ENGINE ---
    st.markdown("---")
    st.subheader("🔮 Previsione AI Inerziale (+1h)")
    lookback = 24
    recent_prices = df_h['Close'].tail(lookback).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(lookback).reshape(-1, 1), recent_prices)
    pred_price = model.predict(np.array([[lookback]]))[0][0]
    drift = pred_price - prezzo_attuale

    # Indicatori Tecnici Daily
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    df_d['ADX'] = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])['ADX_14']
    last_rsi = df_d['RSI'].iloc[-1]
    last_atr = df_d['ATR'].iloc[-1]
    last_adx = df_d['ADX'].iloc[-1]
    div_sig = detect_divergence(df_d)

    # Sentinel Score
    score = 50
    if drift > (pip_unit * 2): score += 25
    elif drift < -(pip_unit * 2): score -= 25
    if not strength_data.empty:
        if strength_data.index[0] in pair[:3]: score += 25
        elif strength_data.index[-1] in pair[:3]: score -= 25

    m1, m2, m3 = st.columns(3)
    m1.metric("RSI (Daily)", f"{last_rsi:.1f}", div_sig)
    m2.metric("Inerzia AI (+1h)", price_fmt.format(pred_price), f"{drift:.5f}")
    m3.metric("Sentinel Score", f"{score}/100")

    # --- LOGICA SEGNALE & MONEY MANAGEMENT ---
    st.markdown("---")
    if is_low_liquidity():
        st.warning("⚠️ FILTRO LIQUIDITÀ: Mercato instabile (Rollover). Scansione in pausa.")
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
                <h2 style="color: {color}; margin-top:0;">🚀 SEGNALE SENTINEL: {action}</h2>
                <table style="width:100%; color: white; font-size: 18px;">
                    <tr>
                        <td><b>Entry:</b> {price_fmt.format(prezzo_attuale)}</td>
                        <td><b>Stop Loss:</b> {price_fmt.format(sl)}</td>
                        <td><b>Take Profit:</b> {price_fmt.format(tp)}</td>
                    </tr>
                    <tr style="color: #ffcc00;">
                        <td><b>Rischio:</b> ${risk_cash:.2f}</td>
                        <td><b>Distanza SL:</b> {dist_pips:.1f} pips</td>
                        <td><b>SIZE CONSIGLIATA:</b> {lotti:.2f} Lotti</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)
        
        # Log History
        new_row = pd.DataFrame([{'Orario': datetime.now().strftime("%H:%M:%S"), 'Asset': pair, 'Direzione': action, 'Prezzo': prezzo_attuale, 'SL': sl, 'TP': tp}])
        st.session_state['signal_history'] = pd.concat([st.session_state['signal_history'], new_row], ignore_index=True)
    else:
        st.info("🔎 Sentinel in scansione... Nessun setup ad alta probabilità al momento.")

    # --- ANALISI EXTRA ---
    with st.expander("📊 Correlazione & Volatilità"):
        st.write(f"ADX (Forza Trend): {last_adx:.1f}")
        st.write(f"Distanza ATR: {last_atr:.5f}")

    if not st.session_state['signal_history'].empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Storico Segnali")
        st.sidebar.dataframe(st.session_state['signal_history'].tail(5))
else:
    st.error("Errore nel caricamento dati. Riprova tra pochi secondi.")

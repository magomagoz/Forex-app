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

# --- FUNZIONI TECNICHE ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "Tokyo 🇯🇵": (time(0,0), time(9,0)), 
        "Londra 🇬🇧": (time(8,0), time(17,0)), 
        "New York 🇺🇸": (time(13,0), time(22,0))
    }
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

def is_low_liquidity():
    # Filtro per evitare il rollover (23:00 - 01:00) dove lo spread è altissimo
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
    return pd.Series(strength).sort_values(ascending=True)

def get_pip_info(pair):
    if "JPY" in pair: return 0.01, "{:.2f}", 1000 
    return 0.0001, "{:.4f}", 10

# --- BANNER ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', Courier, monospace; letter-spacing: 5px; margin: 0;">
            📊 FOREX MOMENTUM PRO
        </h1>
        <p style="color: white; font-size: 14px; opacity: 0.8; margin: 5px 0 0 0;">
            Sentinel System • AI Drift Engine • Risk Calculator v5.0
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
st.sidebar.header("🕹 Trading Desk")
pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)

if st.sidebar.button("🔄 AGGIORNA DATI"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 3. DATA ENGINE ---
pip_unit, price_fmt, pip_mult = get_pip_info(pair)
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    # --- STRENGTH METER ---
    st.subheader("⚡ Currency Strength Meter")
    s_series = get_currency_strength()
    display_strength = s_series.sort_values(ascending=False)
    cols = st.columns(6)
    for i, curr in enumerate(display_strength.index[:6]):
        val = display_strength[curr]
        col_c = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b style='color:white;'>{curr}</b><br><span style='color:{col_c}; font-weight:bold;'>{val:.2f}%</span></div>", unsafe_allow_html=True)

    # --- AI ANALYSIS ---
    st.markdown("---")
    lookback = 24
    recent_h = df_h['Close'].tail(lookback).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(lookback).reshape(-1, 1), recent_h)
    pred_price = model.predict(np.array([[lookback]]))[0][0]
    drift = pred_price - recent_h[-1][0]

    # Indicatori
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    last_c, last_rsi, last_atr = df_d['Close'].iloc[-1], df_d['RSI'].iloc[-1], df_d['ATR'].iloc[-1]

    # Sentinel Score
    score = 50
    if drift > (pip_unit * 2): score += 25
    elif drift < -(pip_unit * 2): score -= 25
    if display_strength.index[0] in pair[:3]: score += 25
    elif display_strength.index[-1] in pair[:3]: score -= 25

    m1, m2, m3 = st.columns(3)
    m1.metric("Prezzo Attuale", price_fmt.format(last_c))
    m2.metric("Inerzia AI (+1h)", price_fmt.format(pred_price), f"{drift:.5f}")
    m3.metric("Sentinel Score", f"{score}/100")

    # --- LOGICA SEGNALE ---
    st.markdown("---")
    if is_low_liquidity():
        st.warning("⚠️ FILTRO LIQUIDITÀ: Operatività sconsigliata durante il rollover (23:00-01:00 UTC).")
        action = None
    else:
        action = "LONG" if (score >= 75 and last_rsi < 60) else "SHORT" if (score <= 25 and last_rsi > 40) else None

    if action:
        sl = last_c - (1.5 * last_atr) if action == "LONG" else last_c + (1.5 * last_atr)
        tp = last_c + (3 * last_atr) if action == "LONG" else last_c - (3 * last_atr)
        
        # Calcolo Lotti
        risk_cash = balance * (risk_pc / 100)
        dist_pips = abs(last_c - sl) / pip_unit
        lotti = risk_cash / (dist_pips * pip_mult) if dist_pips > 0 else 0
        
        color = "#00ffcc" if action == "LONG" else "#ff4b4b"
        st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 20px; border-radius: 15px; background: #0e1117;">
                <h2 style="color: {color}; margin-top:0;">🚀 SEGNALE RILEVATO: {action}</h2>
                <table style="width:100%; color: white; font-size: 18px;">
                    <tr>
                        <td><b>Entry:</b> {price_fmt.format(last_c)}</td>
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
        
        # Audio e Log
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)
        new_row = pd.DataFrame([{'Orario': datetime.now().strftime("%H:%M:%S"), 'Asset': pair, 'Direzione': action, 'Prezzo': last_c, 'SL': sl, 'TP': tp}])
        if st.session_state['signal_history'].empty or st.session_state['signal_history'].iloc[-1]['Orario'] != new_row.iloc[0]['Orario']:
            st.session_state['signal_history'] = pd.concat([st.session_state['signal_history'], new_row], ignore_index=True)
    else:
        st.info("🔎 Sentinel in scansione... Nessun setup ad alta probabilità.")

    # Grafico
    st.line_chart(df_h['Close'].tail(50))
    
    # Registro in Sidebar
    if not st.session_state['signal_history'].empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Registro Sessione")
        st.sidebar.dataframe(st.session_state['signal_history'].tail(5))
        csv = st.session_state['signal_history'].to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Scarica CSV", csv, "segnali_sentinel.csv", "text/csv")

else:
    st.error("Errore nel caricamento dei dati.")

# --- GESTIONE TIMER VISIVO (In fondo per non bloccare il caricamento) ---
# Questo simula il countdown visivo tra i refresh di 120s
timer_placeholder.warning(f"Aggiornamento tra 120s")

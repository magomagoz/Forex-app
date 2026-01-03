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
import time as time_lib
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

# Refresh ogni 120 secondi (2 minuti)
st_autorefresh(interval=120 * 1000, key="sentinel_refresh")

# Inizializzazione Session State
if 'signal_history' not in st.session_state:
    st.session_state['signal_history'] = pd.DataFrame(columns=['Orario', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP'])

if 'prediction_log' not in st.session_state:
    st.session_state['prediction_log'] = None

# --- FUNZIONI TECNICHE (Spostate in alto per essere disponibili) ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "Tokyo 🇯🇵": (time(0,0), time(9,0)), 
        "Londra 🇬🇧": (time(8,0), time(17,0)), 
        "New York 🇺🇸": (time(13,0), time(22,0))
    }
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

@st.cache_data(ttl=110)
def get_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df is None or df.empty: return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
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

def detect_divergence(df):
    if len(df) < 20: return "Dati Insufficienti"
    price, rsi = df['Close'], df['RSI']
    curr_p, curr_r = price.iloc[-1], rsi.iloc[-1]
    prev_max_p, prev_max_r = price.iloc[-15:-1].max(), rsi.iloc[-15:-1].max()
    prev_min_p, prev_min_r = price.iloc[-15:-1].min(), rsi.iloc[-15:-1].min()
    
    if curr_p > prev_max_p and curr_r < prev_max_r and curr_r > 50:
        return "📉 BEARISH (Div. Negativa)"
    elif curr_p < prev_min_p and curr_r > prev_min_r and curr_r < 50:
        return "📈 BULLISH (Div. Positiva)"
    return "Neutrale"

def get_pip_info(pair):
    if "JPY" in pair: return 0.01, "{:.2f}", 1000 
    return 0.0001, "{:.4f}", 10

def calculate_squeeze(df):
    length, mult_bb, mult_kc = 20, 2, 1.5
    bb = ta.bbands(df['Close'], length=length, std=mult_bb)
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=length, scalar=mult_kc)
    is_sqz = (bb.iloc[:, 2] < kc.iloc[:, 2]) & (bb.iloc[:, 0] > kc.iloc[:, 0])
    return is_sqz.iloc[-1], is_sqz

@st.cache_data(ttl=600)
def get_correlation_matrix(pairs_list):
    combined_data = pd.DataFrame()
    for p in pairs_list:
        df = yf.download(p, period="60d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            combined_data[p] = df['Close']
    return combined_data.corr()

# --- BANNER DI TESTATA ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', Courier, monospace; letter-spacing: 5px; margin: 0;">
            📊 FOREX MOMENTUM PRO
        </h1>
        <p style="color: white; font-size: 14px; opacity: 0.8; margin: 5px 0 0 0;">
            Sentinel System • AI Drift Engine • Risk Management v4.0
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR & COUNTDOWN ---
st.sidebar.header("🛠 Trading Desk")
container_timer = st.sidebar.empty()
container_timer.info("Prossimo Update: 120s")

pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 3. DATA FETCH ---
pip_unit, price_fmt, pip_mult = get_pip_info(pair)
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    # --- HEADER ---
    st.title(f"Analisi: {pair}")
    st.caption(f"Ultimo Check: {datetime.now().strftime('%H:%M:%S')}")

    # --- STRENGTH METER ---
    st.subheader("⚡ Currency Strength Meter")
    strength_series = get_currency_strength()
    display_strength = strength_series.sort_values(ascending=False)
    cols = st.columns(6)
    for i, curr in enumerate(display_strength.index[:6]):
        val = display_strength[curr]
        col_c = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"""
            <div style='text-align:center; border:1px solid #444; border-radius:10px; padding:12px; background:#1e1e1e;'>
                <b style='color: white; font-size: 14px;'>{curr}</b><br>
                <span style='color:{col_c}; font-size: 16px; font-weight: bold;'>{val:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # --- AI PREDICTION ---
    st.markdown("---")
    lookback = 24
    recent_h = df_h['Close'].tail(lookback).values.reshape(-1, 1)
    X_model = np.arange(len(recent_h)).reshape(-1, 1)
    model = LinearRegression().fit(X_model, recent_h)
    pred_price = model.predict(np.array([[lookback]]))[0][0]
    drift = pred_price - recent_h[-1][0]

    # --- INDICATORI TECNICI ---
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    df_d['ADX'] = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])['ADX_14']
    last_c, last_rsi, last_adx, last_atr = df_d['Close'].iloc[-1], df_d['RSI'].iloc[-1], df_d['ADX'].iloc[-1], df_d['ATR'].iloc[-1]
    div_sig = detect_divergence(df_d)
    is_sqz, _ = calculate_squeeze(df_d)

    # --- SENTINEL SCORE ---
    final_score = 50
    reasons = []
    if drift > (pip_unit * 2): final_score += 25; reasons.append("AI Bullish")
    elif drift < -(pip_unit * 2): final_score -= 25; reasons.append("AI Bearish")
    if display_strength.index[0] in pair[:3]: final_score += 25; reasons.append(f"{pair[:3]} Forte")
    elif display_strength.index[-1] in pair[:3]: final_score -= 25; reasons.append(f"{pair[:3]} Debole")

    # Metrics Display
    m1, m2, m3 = st.columns(3)
    m1.metric("Prezzo Attuale", price_fmt.format(last_c))
    m2.metric("Inerzia AI (+1h)", price_fmt.format(pred_price), f"{drift:.5f}")
    m3.metric("Sentinel Score", f"{final_score}/100")

    # --- SEGNALE OPERATIVO ---
    st.markdown("---")
    action = "LONG" if (final_score >= 75 and last_rsi < 65) else "SHORT" if (final_score <= 25 and last_rsi > 35) else None

    if action:
        sl = last_c - (1.5 * last_atr) if action == "LONG" else last_c + (1.5 * last_atr)
        tp = last_c + (3 * last_atr) if action == "LONG" else last_c - (3 * last_atr)
        risk_cash = balance * (risk_pc / 100)
        dist_pips = abs(last_c - sl) / pip_unit
        lotti = risk_cash / (dist_pips * pip_mult) if dist_pips > 0 else 0
        
        color = "#00ffcc" if action == "LONG" else "#ff4b4b"
        st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 20px; border-radius: 15px; background: #0e1117;">
                <h2 style="color: {color}; margin-top:0;">🚀 SEGNALE SENTINEL: {action}</h2>
                <table style="width:100%; color: white; font-size: 18px;">
                    <tr>
                        <td><b>Entry:</b> {price_fmt.format(last_c)}</td>
                        <td><b>Stop Loss:</b> {price_fmt.format(sl)}</td>
                        <td><b>Take Profit:</b> {price_fmt.format(tp)}</td>
                    </tr>
                    <tr style="color: #ffcc00;">
                        <td><b>Rischio:</b> ${risk_cash:.2f}</td>
                        <td><b>Distanza SL:</b> {dist_pips:.1f} pips</td>
                        <td><b>SIZE:</b> {lotti:.2f} Lotti</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
        st.toast(f"🚨 SEGNALE {action} RILEVATO", icon="🎯")
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)
        
        # Log History
        new_row = pd.DataFrame([{'Orario': datetime.now().strftime("%H:%M:%S"), 'Asset': pair, 'Direzione': action, 'Prezzo': last_c, 'SL': sl, 'TP': tp}])
        if st.session_state['signal_history'].empty or st.session_state['signal_history'].iloc[-1]['Orario'] != new_row.iloc[0]['Orario']:
            st.session_state['signal_history'] = pd.concat([st.session_state['signal_history'], new_row], ignore_index=True)
    else:
        st.info("🔎 Sentinel in scansione... Nessun setup ad alta probabilità.")

    # --- GRAFICI E STRUMENTI ---
    st.line_chart(df_h['Close'].tail(50))
    
    col_inf1, col_inf2 = st.columns(2)
    with col_inf1:
        with st.expander("📊 Strategia Daily Details"):
            st.write(f"RSI: {last_rsi:.1f} | ADX: {last_adx:.1f}")
            st.write(f"Divergenza: {div_sig}")
            st.write(f"Squeeze: {'ATTIVO' if is_sqz else 'RELEASE'}")
    with col_inf2:
        with st.expander("📊 Correlazione Asset"):
            corr = get_correlation_matrix(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"])
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="RdYlGn", ax=ax_corr)
            st.pyplot(fig_corr)

    # Sidebar History
    if not st.session_state['signal_history'].empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Registro Segnali")
        st.sidebar.dataframe(st.session_state['signal_history'].tail(5))
        csv = st.session_state['signal_history'].to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Scarica CSV", csv, "segnali.csv", "text/csv")

else:
    st.error("Dati non disponibili o errore API.")

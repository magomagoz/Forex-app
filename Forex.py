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

if 'prediction_log' not in st.session_state:
    st.session_state['prediction_log'] = None

# --- BANNER DI TESTATA ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); 
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 25px; border: 1px solid #00ffcc;">
        <h1 style="color: #00ffcc; font-family: 'Courier New', Courier, monospace; letter-spacing: 5px; margin: 0;">
            📊 FOREX MOMENTUM PRO
        </h1>
        <p style="color: white; font-size: 14px; opacity: 0.8; margin: 5px 0 0 0;">
            Sentinel System • Visual Analytics v3.0 • AI Drift Engine
        </p>
    </div>
""", unsafe_allow_html=True)

# --- 2. FUNZIONI TECNICHE ---
def get_session_status():
    now_utc = datetime.now(pytz.utc).time()
    sessions = {
        "Tokyo 🇯🇵": (time(0,0), time(9,0)), 
        "Londra 🇬🇧": (time(8,0), time(17,0)), 
        "New York 🇺🇸": (time(13,0), time(22,0))
    }
    return {name: start <= now_utc <= end for name, (start, end) in sessions.items()}

@st.cache_data(ttl=300)
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

def get_pip_value(pair):
    if "JPY" in pair: return 0.01, "{:.2f}"
    return 0.0001, "{:.4f}"

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

# --- 3. SIDEBAR ---
st.sidebar.header("🛠 Trading Desk")
pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)
pip_unit, price_fmt = get_pip_value(pair)

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 4. HEADER & DATA FETCH ---
col_head1, col_head2 = st.columns([5, 1])
with col_head1:
    st.title(f"Analisi: {pair}")
    st.caption(f"Ultimo Check: {datetime.now().strftime('%H:%M:%S')}")
with col_head2:
    if st.button("🔄 AGGIORNA"):
        st.cache_data.clear()
        st.rerun()

df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    
    # --- 5. STRENGTH METER & CHART ---
    st.subheader("⚡ Currency Strength Meter")
    strength_series = get_currency_strength()
    
    # Box Orizzontali
    display_strength = strength_series.sort_values(ascending=False)
    cols = st.columns(6)
    for i, curr in enumerate(display_strength.index[:6]):
        val = display_strength[curr]
        col_c = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"""
            <div style='text-align:center; border:1px solid #444; border-radius:10px; padding:12px; background:#1e1e1e;'>
                <b style='color: white; font-size: 16px;'>{curr}</b><br>
                <span style='color:{col_c}; font-size: 18px; font-weight: bold;'>{val:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("📊 Analisi Comparativa Forza", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        colors = ['#00ffcc' if x > 0 else '#ff4b4b' for x in strength_series.values]
        strength_series.plot(kind='barh', color=colors, ax=ax)
        ax.tick_params(colors='white')
        st.pyplot(fig)

    # --- 6. AI PREDICTION ENGINE ---
    st.markdown("---")
    st.subheader("🔮 Modello Predittivo AI (+1h)")
    
    lookback = 24
    recent_data = df_h['Close'].tail(lookback).values
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data.reshape(-1, 1)
    
    model = LinearRegression().fit(X, y)
    predicted_price = model.predict(np.array([[lookback]]))[0][0]
    current_price = y[-1][0]
    drift = predicted_price - current_price

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Prezzo Attuale", price_fmt.format(current_price))
    col_p2.metric("Previsione AI", price_fmt.format(predicted_price), f"{drift:.5f}")
    
    # Reality Check
    with st.expander("🧪 Reality Check"):
        if st.button("📸 Salva Previsione"):
            st.session_state['prediction_log'] = {'time': datetime.now().strftime('%H:%M'), 'pred': predicted_price, 'pair': pair}
        if st.session_state['prediction_log']:
            log = st.session_state['prediction_log']
            st.write(f"Ultimo salvataggio: {log['pred']:.5f} alle {log['time']}")

    st.line_chart(pd.DataFrame({'Storico': recent_data, 'Trend': model.predict(X).flatten()}))

    # --- 7. ORACLE SCORE & SENTINEL ---
    final_score = 50
    reasons = []
    
    if drift > (pip_unit * 2): final_score += 20; reasons.append("Inerzia Rialzista")
    elif drift < -(pip_unit * 2): final_score -= 20; reasons.append("Inerzia Ribassista")
    
    if display_strength.index[0] in pair[:3]: final_score += 20; reasons.append(f"{pair[:3]} Forte")
    elif display_strength.index[-1] in pair[:3]: final_score -= 20; reasons.append(f"{pair[:3]} Debole")

    if final_score >= 80 or final_score <= 20:
        st.toast("🚨 SEGNALE SENTINEL ATTIVO", icon="🎯")
        st.error(f"SENTINEL ALERT: Confluenza ({final_score}/100) - {', '.join(reasons)}")
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)

    # --- 8. SETUP OPERATIVO DAILY ---
    st.markdown("---")
    st.subheader("🎯 Analisi Strategica (Daily)")
    
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    df_d['ADX'] = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])['ADX_14']
    
    is_sqz, sqz_series = calculate_squeeze(df_d)
    div_sig = detect_divergence(df_d)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI", f"{df_d['RSI'].iloc[-1]:.1f}")
    c2.metric("ADX", f"{df_d['ADX'].iloc[-1]:.1f}")
    c3.metric("Divergenza", div_sig)
    c4.metric("Squeeze", "ATTIVO" if is_sqz else "RELEASE")

    # Trading Signal Logic
    last_close = df_d['Close'].iloc[-1]
    last_atr = df_d['ATR'].iloc[-1]
    action = None
    if (div_sig == "📈 BULLISH" or df_d['RSI'].iloc[-1] < 35) and df_d['ADX'].iloc[-1] > 20: action = "LONG"
    elif (div_sig == "📉 BEARISH" or df_d['RSI'].iloc[-1] > 65) and df_d['ADX'].iloc[-1] > 20: action = "SHORT"

    if action:
        sl = last_close - (2 * last_atr) if action == "LONG" else last_close + (2 * last_atr)
        tp = last_close + (4 * last_atr) if action == "LONG" else last_close - (4 * last_atr)
        st.success(f"**SEGNALE {action}** | Entry: {price_fmt.format(last_close)} | SL: {price_fmt.format(sl)} | TP: {price_fmt.format(tp)}")
    
    with st.expander("📊 Correlazione Asset"):
        corr = get_correlation_matrix(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"])
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="RdYlGn", ax=ax_corr)
        st.pyplot(fig_corr)

else:
    st.error("Connessione dati fallita. Ricarica.")

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time  # AGGIUNTO 'time' per get_session_status
import seaborn as sns
import matplotlib.pyplot as plt
import pytz
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIGURAZIONE ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")
# Autorefresh già presente, rimosso il duplicato a fondo pagina per pulizia
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
    # Orari sessioni (UTC)
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
        if df.empty: return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Errore API Dati per {ticker}: {e}")
        return None

def get_currency_strength():
    tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"]
    data = yf.download(tickers, period="2d", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    
    returns = data.pct_change().fillna(0).iloc[-1] * 100    
    strength = {
        "USD 🇺🇸": -(returns.mean()), # Proxy semplificata per stabilità
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

@st.cache_data(ttl=3600)
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
st.sidebar.header("🕹 Control Panel")
pair = st.sidebar.selectbox("Seleziona Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"], key="main_pair")
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)
pip_unit, price_fmt = get_pip_value(pair)

st.sidebar.markdown("---")
status = get_session_status()
for s, op in status.items():
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
	
    # --- 6. ⚡ CURRENCY STRENGTH METER ---
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    strength_data = get_currency_strength() # CORRETTO: assegnato a strength_data
    
    cols = st.columns(6)
    for i, (curr, val) in enumerate(strength_data.items()[:6]):
        color = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"<div style='text-align:center; border:1px solid #444; border-radius:10px; padding:10px; background:#1e1e1e;'><b>{curr}</b><br><span style='color:{color};'>{val:.2f}%</span></div>", unsafe_allow_html=True)
    
    c_str1, c_str2 = st.columns([2, 1])
    with c_str1:
        fig_str, ax_str = plt.subplots(figsize=(10, 4))
        colors = ['#00ffcc' if x > 0 else '#ff4b4b' for x in strength_data.values]
        strength_data.plot(kind='barh', color=colors, ax=ax_str)
        ax_str.set_title("Forza Relativa Valute")
        st.pyplot(fig_str)
    with c_str2:
        st.success(f"Valuta più Forte: {strength_data.index[0]}")
        st.error(f"Valuta più Debole: {strength_data.index[-1]}")

    # --- ANALISI VOLATILITÀ ---
    st.subheader("🌋 Analisi Volatilità & Squeeze")
    std = df_d['Close'].rolling(20).std()
    ma = df_d['Close'].rolling(20).mean()
    upper_bb = ma + (2 * std)
    lower_bb = ma - (2 * std)
    atr_val = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=20)
    upper_kc = ma + (1.5 * atr_val)
    lower_kc = ma - (1.5 * atr_val)
    is_sqz = (upper_bb.iloc[-1] < upper_kc.iloc[-1]) # Definizione per scorecard
    
    if is_sqz:
        st.warning("⚠️ SQUEEZE ATTIVO: Il mercato sta comprimendo i prezzi.")
    else:
        st.success("🚀 RELEASE: Momentum in fase di espansione.")

    # --- 7. MODELLO PREDITTIVO AI ---
    st.markdown("---")
    st.subheader("🔮 Modello Predittivo AI (+1h)")
    drift = 0.0
    if len(df_h) > 24:
        recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
        model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
        pred = model.predict(np.array([[24]]))[0][0]
        curr_h = recent_h[-1][0]
        drift = pred - curr_h
        
        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Prezzo Attuale", price_fmt.format(curr_h))
        cp2.metric("Previsione +1h", price_fmt.format(pred), f"{drift:.5f}")
        
        if st.button("📸 Salva Reality Check"):
            st.session_state.prediction_log = {"time": datetime.now().strftime("%H:%M"), "pred": pred, "pair": pair}
        
    st.line_chart(df_h['Close'].tail(50))

    # --- 8. SETUP OPERATIVO DAILY ---
    st.markdown("---")
    st.subheader("🎯 Setup Operativo")
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    df_d['ADX'] = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])['ADX_14']
    
    last_c = df_d['Close'].iloc[-1]
    last_rsi = df_d['RSI'].iloc[-1]
    last_adx = df_d['ADX'].iloc[-1]
    last_atr = df_d['ATR'].iloc[-1]
    div = detect_divergence(df_d)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RSI (14)", f"{last_rsi:.1f}")
    m2.metric("ADX (14)", f"{last_adx:.1f}")
    m3.metric("Divergenza", div)
    m4.metric("ATR", price_fmt.format(last_atr))

    action = None
    if (div == "📈 BULLISH" or last_rsi < 30) and last_adx > 20: action = "LONG"
    elif (div == "📉 BEARISH" or last_rsi > 70) and last_adx > 20: action = "SHORT"

    if action:
        risk_val = balance * (risk_pc / 100)
        sl = last_c - (2 * last_atr) if action == "LONG" else last_c + (2 * last_atr)
        tp = last_c + (4 * last_atr) if action == "LONG" else last_c - (4 * last_atr)
        pips_sl = abs(last_c - sl) / pip_unit
        lots = risk_val / (pips_sl * (10 if "JPY" not in pair else 9))
        st.success(f"**Segnale {action}** | Entry: {price_fmt.format(last_c)} | SL: {price_fmt.format(sl)} | TP: {price_fmt.format(tp)}")
        st.info(f"Size Suggerita: {lots:.2f} Lotti per rischiare ${risk_val:.2f}")

    # --- 9. DASHBOARD DI SINTESI ---
    st.subheader("📊 Valutazione Opportunità (Confluenza)")
    final_score = 50
    reasons = []
    
    if drift > (pip_unit * 2): 
        final_score += 15
        reasons.append("AI: Inerzia Rialzista")
    elif drift < -(pip_unit * 2):
        final_score -= 15
        reasons.append("AI: Inerzia Ribassista")

    if strength_data.index[0] in pair[:3]:
        final_score += 20
        reasons.append(f"Strength: {pair[:3]} Dominante")
    elif strength_data.index[-1] in pair[:3]:
        final_score -= 20
        reasons.append(f"Strength: {pair[:3]} Debole")

    if not is_sqz:
        final_score += 10
        reasons.append("Volatilità: Espansione")
    
    st.metric("Oracle Score", f"{final_score}/100")
    st.write(f"Confluenze: {', '.join(reasons)}")

    # --- 10. ALERT SENTINEL ---
    if final_score >= 80 or final_score <= 20:
        alert_msg = "🔥 ALTA PROBABILITÀ LONG" if final_score >= 80 else "⚠️ FORTE PRESSIONE SHORT"
        st.markdown(f"""
            <style>
            @keyframes blink {{ 0% {{opacity: 1;}} 50% {{opacity: 0.3;}} 100% {{opacity: 1;}} }}
            .sentinel-alert {{
                background-color: {"#00ffcc" if final_score >= 80 else "#ff4b4b"};
                color: black; padding: 20px; border-radius: 10px; text-align: center;
                animation: blink 1s infinite; font-weight: bold;
            }}
            </style>
            <div class="sentinel-alert"><h1>{alert_msg}</h1></div>
            <audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>
            """, unsafe_allow_html=True)

    # --- 11. CORRELAZIONE ---
    with st.expander("📊 Matrice di Correlazione"):
        corr_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
        corr = get_correlation_matrix(corr_pairs)
        fig_c, ax_c = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="RdYlGn", ax=ax_c)
        st.pyplot(fig_c)
else:
    st.error("Impossibile recuperare i dati. Verifica la connessione o il Ticker.")

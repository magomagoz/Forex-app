import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

# Inizializzazione Session State per il Reality Check
if 'prediction_log' not in st.session_state:
    st.session_state['prediction_log'] = None

# --- 2. DEFINIZIONE FUNZIONI TECNICHE ---

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

def calculate_squeeze(df):
    length, mult_bb, mult_kc = 20, 2, 1.5
    bb = ta.bbands(df['Close'], length=length, std=mult_bb)
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=length, scalar=mult_kc)
    
    # Selezione sicura colonne
    lower_bb, upper_bb = bb.iloc[:, 0], bb.iloc[:, 2]
    lower_kc, upper_kc = kc.iloc[:, 0], kc.iloc[:, 2]

    is_sqz = (upper_bb < upper_kc) & (lower_bb > lower_kc)
    return is_sqz.iloc[-1], is_sqz

def get_correlation_matrix(pairs_list):
    combined_data = pd.DataFrame()
    for p in pairs_list:
        df = yf.download(p, period="60d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            combined_data[p] = df['Close']
    return combined_data.corr()

def get_currency_strength():
    tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"]
    data = yf.download(tickers, period="2d", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    returns = data.pct_change().iloc[-1] * 100
    
    strength = {
        "USD": (-returns["EURUSD=X"] - returns["GBPUSD=X"] + returns["USDJPY=X"] - returns["AUDUSD=X"] + returns["USDCAD=X"] + returns["USDCHF=X"] - returns["NZDUSD=X"]) / 7,
        "EUR": (returns["EURUSD=X"] + returns["EURJPY=X"] + returns["EURGBP=X"]) / 3,
        "GBP": (returns["GBPUSD=X"] + returns["GBPJPY=X"] - returns["EURGBP=X"]) / 3,
        "JPY": (-returns["USDJPY=X"] - returns["EURJPY=X"] - returns["GBPJPY=X"]) / 3,
        "AUD": (returns["AUDUSD=X"]) / 1,
        "CAD": (-returns["USDCAD=X"]) / 1,
    }
    return pd.Series(strength).sort_values(ascending=False)

def detect_divergence(df):
    if len(df) < 20: return "Dati Insufficienti"
    price, rsi = df['Close'], df['RSI']
    curr_p, curr_r = price.iloc[-1], rsi.iloc[-1]
    prev_max_p, prev_max_r = price.iloc[-15:-1].max(), rsi.iloc[-15:-1].max()
    prev_min_p, prev_min_r = price.iloc[-15:-1].min(), rsi.iloc[-15:-1].min()
    
    if curr_p > prev_max_p and curr_r < prev_max_r and curr_r > 50:
        return "📉 BEARISH (Divergenza Negativa)"
    elif curr_p < prev_min_p and curr_r > prev_min_r and curr_r < 50:
        return "📈 BULLISH (Divergenza Positiva)"
    return "Neutrale"

def get_pip_value(pair):
    if "JPY" in pair: return 0.01, "{:.2f}"
    return 0.0001, "{:.4f}"

# --- 3. SIDEBAR (TRADING DESK) ---
st.sidebar.header("🛠 Trading Desk")
pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)
pip_unit, price_fmt = get_pip_value(pair)

# --- 4. HEADER ---
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.title(f"Analisi & Previsione: {pair}")
    st.caption(f"Ultimo Check: {datetime.now().strftime('%H:%M:%S')}")
with col_h2:
    if st.button("🔄 AGGIORNA"):
        st.cache_data.clear()
        st.rerun()

# --- 5. DOWNLOAD DATI PRINCIPALI ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None:
    # --- 6. CURRENCY STRENGTH METER ---
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    strength_data = get_currency_strength()
    c_str1, c_str2 = st.columns([2, 1])
    with c_str1:
        fig_str, ax_str = plt.subplots(figsize=(8, 3))
        strength_data.plot(kind='barh', color=['green' if x > 0 else 'red' for x in strength_data.values], ax=ax_str)
        st.pyplot(fig_str)
    with c_str2:
        st.success(f"Top: {strength_data.index[0]}")
        st.error(f"Worst: {strength_data.index[-1]}")

    # --- 7. MODELLO PREDITTIVO ---
    st.markdown("---")
    st.subheader("🔮 Modello Predittivo AI (+1h)")
    drift = 0.0 # Inizializzazione per scorecard
    if df_h is not None and len(df_h) > 24:
        recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
        model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
        pred = model.predict(np.array([[24]]))[0][0]
        curr_h = recent_h[-1][0]
        drift = pred - curr_h
        
        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Prezzo Attuale", price_fmt.format(curr_h))
        cp2.metric("Previsione +1h", price_fmt.format(pred), f"{drift:.5f}")
        
        if st.button("📸 Salva per Reality Check"):
            st.session_state.prediction_log = {"time": datetime.now().strftime("%H:%M"), "pred": pred, "pair": pair}
        
        st.line_chart(pd.DataFrame(recent_h, columns=['Prezzo']))

    # --- 8. ANALISI SQUEEZE & VOLATILITÀ ---
    st.markdown("---")
    st.subheader("🌋 Analisi Volatilità & Squeeze")
    is_sqz, sqz_series = calculate_squeeze(df_d)
    cq1, cq2 = st.columns([1, 2])
    with cq1:
        if is_sqz: st.warning("⚠️ SQUEEZE: Caricamento Volatilità")
        else: st.success("🚀 RELEASE: Momentum in corso")
    with cq2:
        st.line_chart(sqz_series.tail(50).astype(int))

    # --- 9. SETUP OPERATIVO DAILY ---
    st.markdown("---")
    st.subheader("🎯 Setup Operativo")
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    df_d['ADX'] = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])['ADX_14']
    
    last_c, last_rsi, last_adx, last_atr = df_d['Close'].iloc[-1], df_d['RSI'].iloc[-1], df_d['ADX'].iloc[-1], df_d['ATR'].iloc[-1]
    div = detect_divergence(df_d)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RSI", f"{last_rsi:.1f}")
    m2.metric("ADX", f"{last_adx:.1f}")
    m3.metric("Divergenza", div)
    m4.metric("ATR", price_fmt.format(last_atr))

    action = None
    if (div == "📈 BULLISH" or last_rsi < 35) and last_adx > 20: action = "LONG"
    elif (div == "📉 BEARISH" or last_rsi > 65) and last_adx > 20: action = "SHORT"

    if action:
        risk_val = balance * (risk_pc / 100)
        sl = last_c - (2 * last_atr) if action == "LONG" else last_c + (2 * last_atr)
        tp = last_c + (4 * last_atr) if action == "LONG" else last_c - (4 * last_atr)
        pips_sl = abs(last_c - sl) / pip_unit
        lots = risk_val / (pips_sl * (10 if "JPY" not in pair else 9))
        
        st.success(f"**Segnale {action}** | Entry: {price_fmt.format(last_c)} | SL: {price_fmt.format(sl)} | TP: {price_fmt.format(tp)}")
        st.info(f"Size: {lots:.2f} Lotti per rischiare ${risk_val:.2f}")

    # --- 10. DASHBOARD DI SINTESI (SCORECARD) ---
    st.markdown("---")
    st.subheader("📊 Valutazione Opportunità (Confluenza)")
    score = 50
    reasons = []
    
    if drift > (pip_unit * 5): 
        score += 15
        reasons.append("AI: Inerzia Rialzista")
    elif drift < -(pip_unit * 5):
        score -= 15
        reasons.append("AI: Inerzia Ribassista")

    if strength_data.index[0] in pair[:3]:
        score += 20
        reasons.append(f"Strength: {pair[:3]} Dominante")
    elif strength_data.index[-1] in pair[:3]:
        score -= 20
        reasons.append(f"Strength: {pair[:3]} Debole")

    if not is_sqz:
        score += 10
        reasons.append("Volatilità: Release")
    else:
        reasons.append("Volatilità: Squeeze")

    if last_rsi < 40: score += 5
    elif last_rsi > 60: score -= 5

    # --- SISTEMA DI ALERT & SENTINEL ---
    if score >= 80 or score <= 20:
        alert_msg = "🔥 ALTA PROBABILITÀ RILEVATA!" if score >= 80 else "⚠️ FORTE PRESSIONE RIBASSISTA!"
        
        # Suono di notifica (usiamo un link a un suono pulito di sistema)
        audio_url = "https://www.soundjay.com/buttons/beep-07a.mp3"
        
        st.markdown(f"""
            <style>
            @keyframes blink {{ 0% {{opacity: 1;}} 50% {{opacity: 0.3;}} 100% {{opacity: 1;}} }}
            .sentinel-alert {{
                background-color: {color_score};
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                animation: blink 1s infinite;
                border: 4px solid white;
            }}
            </style>
            <div class="sentinel-alert">
                <h1>{alert_msg}</h1>
                <h2>Punteggio Confluenza: {score}/100</h2>
            </div>
            <audio autoplay>
                <source src="{audio_url}" type="audio/mpeg">
            </audio>
            """, unsafe_allow_html=True)

    # --- AUTO-REFRESH (Ogni 300 secondi = 5 minuti) ---
    # Questo pezzo di codice forza l'app a ricaricarsi da sola
    from streamlit_autorefresh import st_autorefresh
    count = st_autorefresh(interval=300 * 1000, key="sentinel_refresh")
    
    # --- 11. CORRELAZIONE ---
    with st.expander("📊 Vedi Matrice di Correlazione"):
        corr = get_correlation_matrix(["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"])
        fig_c, ax_c = plt.subplots()
        sns.heatmap(corr, annot=True, cmap="RdYlGn", ax=ax_c)
        st.pyplot(fig_c)
else:
    st.error("Dati non disponibili.")

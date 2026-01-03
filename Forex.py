import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time

# --- 1. CONFIGURAZIONE PAGINA E STATO ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

# Inizializzazione Session State per il Reality Check
if 'prediction_log' not in st.session_state:
    st.session_state['prediction_log'] = None

# --- 2. FUNZIONI CORE (DATA & ANALYSIS) ---

@st.cache_data(ttl=600)  # Cache di 10 minuti per evitare ban da Yahoo
def get_market_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        # Fix per MultiIndex di yfinance recente
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty: return None
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Errore API Dati: {e}")
        return None

def detect_divergence(df):
    """Logica avanzata per divergenze RSI su 14 periodi"""
    if len(df) < 20: return "Dati Insufficienti"
    
    # Prezzi e Indicatori recenti
    price = df['Close']
    rsi = df['RSI']
    
    # Swing High/Low logic (Semplificata per velocità)
    curr_p = price.iloc[-1]
    curr_r = rsi.iloc[-1]
    prev_max_p = price.iloc[-15:-1].max()
    prev_max_r = rsi.iloc[-15:-1].max()
    prev_min_p = price.iloc[-15:-1].min()
    prev_min_r = rsi.iloc[-15:-1].min()
    
    if curr_p > prev_max_p and curr_r < prev_max_r and curr_r > 50:
        return "📉 BEARISH (Prezzo sale, Momentum scende)"
    elif curr_p < prev_min_p and curr_r > prev_min_r and curr_r < 50:
        return "📈 BULLISH (Prezzo scende, Momentum sale)"
    
    return "Neutrale / Trend Following"

def get_pip_value(pair):
    """Restituisce il valore di 1 pip e la formattazione corretta"""
    if "JPY" in pair:
        return 0.01, "{:.2f}"
    else:
        return 0.0001, "{:.4f}"

# --- 3. INTERFACCIA LATERALE (TRADING DESK) ---
st.sidebar.header("🛠 Trading Desk")
pair = st.sidebar.selectbox("Asset", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
balance = st.sidebar.number_input("Balance Conto ($)", value=10000, step=1000)
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 1.0)

# Gestione JPY per formattazione
pip_unit, price_fmt = get_pip_value(pair)

# --- 4. HEADER & AGGIORNAMENTO ---
col1, col2 = st.columns([5, 1])
with col1:
    st.title(f"Analisi & Previsione: {pair}")
    st.caption(f"Ultimo Check: {datetime.now().strftime('%H:%M:%S')}")
with col2:
    if st.button("🔄 AGGIORNA"):
        st.cache_data.clear()
        st.rerun()

# --- 5. LOGICA PREVISIONALE (HOURLY / INTRADAY) ---
st.markdown("---")
st.subheader("🔮 Modello Predittivo (Prossima Ora)")

# Scarico dati orari per il modello ML
df_h = get_market_data(pair, period="5d", interval="1h")

if df_h is not None and len(df_h) > 24:
    # Preparazione dati per Regressione Lineare (ultime 24 ore)
    lookback = 24
    recent_data = df_h['Close'].tail(lookback).values
    X = np.arange(len(recent_data)).reshape(-1, 1)
    y = recent_data.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Previsione t+1
    next_index = np.array([[lookback]])
    predicted_price = model.predict(next_index)[0][0]
    current_price = y[-1][0]
    drift = predicted_price - current_price
    
    # Visualizzazione Previsione
    col_pred1, col_pred2, col_pred3 = st.columns(3)
    
    col_pred1.metric("Prezzo Attuale", price_fmt.format(current_price))
    col_pred2.metric("Previsione +1h", price_fmt.format(predicted_price), 
                     f"{drift:.5f} (Momentum Drift)")
    
    strength = "ALTA" if abs(drift) > (pip_unit * 10) else "BASSA"
    direction = "RIALZISTA" if drift > 0 else "RIBASSISTA"
    col_pred3.info(f"Inerzia: **{direction}** (Forza: {strength})")
    
    # --- MODULO REALITY CHECK (Didattico) ---
    with st.expander("🧪 Test Accuratezza Modello (Reality Check)"):
        st.write("Salva la previsione attuale e torna tra un'ora per vedere se il modello aveva ragione.")
        
        if st.button("📸 Salva Previsione Corrente"):
            st.session_state['prediction_log'] = {
                'time': datetime.now().strftime('%H:%M'),
                'predicted': predicted_price,
                'start_price': current_price,
                'pair': pair
            }
            st.success("Previsione Salvata nel registro temporaneo!")
            
        # Verifica se esiste un log
        if st.session_state['prediction_log'] and st.session_state['prediction_log']['pair'] == pair:
            log = st.session_state['prediction_log']
            st.write(f"**Previsione Salvata alle {log['time']}:** {price_fmt.format(log['predicted'])}")
            
            # Calcolo errore
            error_pips = abs(current_price - log['predicted']) / pip_unit
            accuracy_color = "green" if error_pips < 10 else "red"
            st.markdown(f"Scostamento attuale: :**{accuracy_color}[{error_pips:.1f} pips]** dal target.")

    # Grafico Proiezione
    chart_data = pd.DataFrame({
        'Storico (24h)': pd.Series(recent_data.flatten()),
        'Trend Line': pd.Series(model.predict(X).flatten())
    })
    st.line_chart(chart_data)

else:
    st.warning("Dati orari insufficienti per il modello predittivo.")

# --- 6. ANALISI STRATEGICA (DAILY / SWING) ---
st.markdown("---")
st.subheader("🎯 Setup Operativo (Daily)")

df_d = get_market_data(pair, period="1y", interval="1d")

if df_d is not None:
    # Calcolo Indicatori
    df_d['RSI'] = ta.rsi(df_d['Close'], length=14)
    df_d['ATR'] = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=14)
    adx = ta.adx(df_d['High'], df_d['Low'], df_d['Close'])
    df_d['ADX'] = adx['ADX_14']
    
    # Ultimi valori
    last_close = df_d['Close'].iloc[-1]
    last_atr = df_d['ATR'].iloc[-1]
    last_rsi = df_d['RSI'].iloc[-1]
    last_adx = df_d['ADX'].iloc[-1]
    signal_div = detect_divergence(df_d)
    
    # Dashboard Indicatori
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RSI (Momentum)", f"{last_rsi:.1f}")
    c2.metric("ATR (Volatilità)", price_fmt.format(last_atr))
    c3.metric("ADX (Forza Trend)", f"{last_adx:.1f}")
    c4.metric("Analisi Divergenza", signal_div)
    
    # --- GENERAZIONE SEGNALE DI TRADING ---
    st.markdown("#### 💡 Suggerimento Operativo")
    
    trend_state = "Laterale"
    if last_adx > 25: trend_state = "Trend Definito"
    
    # Logica Decisionale
    action = None
    if "BULLISH" in signal_div or (last_rsi < 40 and last_adx > 20):
        action = "LONG"
        sl_price = last_close - (2 * last_atr)
        tp_price = last_close + (3 * last_atr) # RR 1:1.5 prudenziale
        color = "success"
    elif "BEARISH" in signal_div or (last_rsi > 60 and last_adx > 20):
        action = "SHORT"
        sl_price = last_close + (2 * last_atr)
        tp_price = last_close - (3 * last_atr)
        color = "error"
    
    if action:
        # Calcolo Size
        risk_amount = balance * (risk_pc / 100)
        dist_pips = abs(last_close - sl_price) / pip_unit
        
        # Formula approssimata per lotti standard (1 lotto = 100k units)
        # Assumendo conto in USD. Per precisione assoluta servirebbe tasso di cambio cross.
        # Valore pip stimato: 10$ per coppie USD standard, variabile per cross.
        val_pip_standard = 10 if "JPY" not in pair else 9 # approx
        lot_size = risk_amount / (dist_pips * val_pip_standard)
        
        container = st.container(border=True)
        container.subheader(f"Segnale: {action}")
        
        c_sig1, c_sig2 = container.columns(2)
        c_sig1.write(f"**Entry:** {price_fmt.format(last_close)}")
        c_sig1.write(f"**Stop Loss:** {price_fmt.format(sl_price)}")
        c_sig1.write(f"**Take Profit:** {price_fmt.format(tp_price)}")
        
        c_sig2.info(f"💰 **Risk Management:**\n\n"
                    f"- Capitale a rischio: ${risk_amount:.2f}\n"
                    f"- Distanza SL: {dist_pips:.1f} pips\n"
                    f"- **Size Consigliata:** {lot_size:.2f} Lotti")
    else:
        st.info("🚧 Nessun setup ad alta probabilità rilevato. Il mercato è in fase di attesa o consolidamento. Meglio non operare.")

else:
    st.error("Impossibile caricare i dati Daily. Riprova più tardi.")









import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
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
            Sentinel System • AI Confluence • Notification v2.0
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
    except:
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

def get_pip_value(pair):
    if "JPY" in pair: return 0.01, "{:.2f}"
    return 0.0001, "{:.4f}"

# --- 3. SIDEBAR ---
st.sidebar.header("🕹 Trading Desk")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
pip_unit, price_fmt = get_pip_value(pair)

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 4. LOGICA SENTINEL ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    # --- CALCOLO AI & DRIFT ---
    recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
    pred_1h = model.predict(np.array([[24]]))[0][0]
    drift = pred_1h - recent_h[-1][0]
    
    # --- CALCOLO SQUEEZE ---
    std = df_d['Close'].rolling(20).std()
    ma = df_d['Close'].rolling(20).mean()
    upper_bb = ma + (2 * std)
    lower_bb = ma - (2 * std)
    atr_val = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=20)
    upper_kc = ma + (1.5 * atr_val)
    is_sqz = (upper_bb.iloc[-1] < upper_kc.iloc[-1])

    # --- CALCOLO FORZA VALUTE ---
    strength_data = get_currency_strength()

    # --- 5. ORACLE SCORE & NOTIFICHE TESTUALI ---
    final_score = 50
    reasons = []
    
    if drift > (pip_unit * 2): final_score += 20; reasons.append("AI Bullish Inerzia")
    elif drift < -(pip_unit * 2): final_score -= 20; reasons.append("AI Bearish Inerzia")
    
    if strength_data.index[0] in pair[:3]: final_score += 20; reasons.append(f"Forza: {pair[:3]} Dominante")
    elif strength_data.index[-1] in pair[:3]: final_score -= 20; reasons.append(f"Forza: {pair[:3]} Debole")

    # --- NOTIFICA PERSISTENTE IN ALTO ---
    if final_score >= 80:
        st.toast(f"🚀 SEGNALE LONG RILEVATO: {pair}", icon="🔥")
        st.error(f"⚠️ SENTINEL ALERT: Confluenza Rialzista ({final_score}/100) - {', '.join(reasons)}")
    elif final_score <= 20:
        st.toast(f"📉 SEGNALE SHORT RILEVATO: {pair}", icon="⚠️")
        st.error(f"⚠️ SENTINEL ALERT: Confluenza Ribassista ({final_score}/100) - {', '.join(reasons)}")

    # --- 6. VISUALIZZAZIONE ---
    st.subheader("⚡ Currency Strength Meter")
    cols = st.columns(6)
    for i, curr in enumerate(strength_data.index[:6]):
        val = strength_data[curr]
        col_c = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"<div style='text-align:center; border:1px solid #333; border-radius:10px; padding:10px; background:#111;'><b>{curr}</b><br><span style='color:{col_c};'>{val:.2f}%</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Prezzo", price_fmt.format(recent_h[-1][0]))
    m2.metric("AI Prediction (+1h)", price_fmt.format(pred_1h), f"{drift:.5f}")
    m3.metric("Oracle Score", f"{final_score}/100")

    st.line_chart(df_h['Close'].tail(50))
    
    if is_sqz:
        st.warning("🌋 SQUEEZE ATTIVO: Possibile movimento esplosivo imminente.")
    
    # Audio Alert
    if final_score >= 80 or final_score <= 20:
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)

else:
    st.info("In attesa di dati freschi dal mercato...")









import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
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
            Sentinel System • Font Optimized for iPad
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
    except:
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

def get_pip_value(pair):
    if "JPY" in pair: return 0.01, "{:.2f}"
    return 0.0001, "{:.4f}"

# --- 3. SIDEBAR ---
st.sidebar.header("🕹 Trading Desk")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
pip_unit, price_fmt = get_pip_value(pair)

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 4. LOGICA SENTINEL ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    # Calcoli AI e Squeeze (stessi della versione precedente)
    recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
    pred_1h = model.predict(np.array([[24]]))[0][0]
    drift = pred_1h - recent_h[-1][0]
    
    std = df_d['Close'].rolling(20).std()
    ma = df_d['Close'].rolling(20).mean()
    upper_bb = ma + (2 * std)
    lower_bb = ma - (2 * std)
    atr_val = ta.atr(df_d['High'], df_d['Low'], df_d['Close'], length=20)
    upper_kc = ma + (1.5 * atr_val)
    is_sqz = (upper_bb.iloc[-1] < upper_kc.iloc[-1])

    strength_data = get_currency_strength()

    # --- 5. ORACLE SCORE ---
    final_score = 50
    reasons = []
    if drift > (pip_unit * 2): final_score += 20; reasons.append("AI Bullish")
    elif drift < -(pip_unit * 2): final_score -= 20; reasons.append("AI Bearish")
    if strength_data.index[0] in pair[:3]: final_score += 20; reasons.append(f"{pair[:3]} Strong")
    elif strength_data.index[-1] in pair[:3]: final_score -= 20; reasons.append(f"{pair[:3]} Weak")

    # Alert Box
    if final_score >= 80 or final_score <= 20:
        st.error(f"⚠️ SENTINEL ALERT ({final_score}/100): {', '.join(reasons)}")
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)

    # --- 6. VISUALIZZAZIONE BOX VALUTE (FONT BIANCO) ---
    st.subheader("⚡ Currency Strength Meter")
    cols = st.columns(6)
    for i, curr in enumerate(strength_data.index[:6]):
        val = strength_data[curr]
        col_c = "#00ffcc" if val > 0 else "#ff4b4b"
        # AGGIUNTO color: white per il nome della valuta
        cols[i].markdown(f"""
            <div style='text-align:center; border:1px solid #444; border-radius:10px; padding:12px; background:#1e1e1e; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);'>
                <b style='color: white; font-size: 16px; letter-spacing: 1px;'>{curr}</b><br>
                <span style='color:{col_c}; font-size: 20px; font-weight: bold;'>{val:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Prezzo", price_fmt.format(recent_h[-1][0]))
    m2.metric("AI Pred. (+1h)", price_fmt.format(pred_1h), f"{drift:.5f}")
    m3.metric("Oracle Score", f"{final_score}/100")

    st.line_chart(df_h['Close'].tail(50))
    
    if is_sqz:
        st.warning("🌋 SQUEEZE ATTIVO: Compressione volatilità.")

else:
    st.info("Sincronizzazione dati...")










import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
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
            Sentinel System • Visual Analytics v3.0
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
    except:
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
    return pd.Series(strength).sort_values(ascending=True) # Ordinamento per grafico a barre

def get_pip_value(pair):
    if "JPY" in pair: return 0.01, "{:.2f}"
    return 0.0001, "{:.4f}"

# --- 3. SIDEBAR ---
st.sidebar.header("🕹 Trading Desk")
pair = st.sidebar.selectbox("Pair", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "BTC-USD"])
pip_unit, price_fmt = get_pip_value(pair)

st.sidebar.markdown("---")
status_sessions = get_session_status()
for s, op in status_sessions.items():
    color = "🟢" if op else "🔴"
    st.sidebar.markdown(f"**{s}**: {color} {'OPEN' if op else 'CLOSED'}")

# --- 4. LOGICA SENTINEL ---
df_d = get_market_data(pair, "1y", "1d")
df_h = get_market_data(pair, "5d", "1h")

if df_d is not None and df_h is not None:
    # Calcoli AI
    recent_h = df_h['Close'].tail(24).values.reshape(-1, 1)
    model = LinearRegression().fit(np.arange(24).reshape(-1, 1), recent_h)
    pred_1h = model.predict(np.array([[24]]))[0][0]
    drift = pred_1h - recent_h[-1][0]
    
    # Calcolo Forza Valute
    strength_series = get_currency_strength()

    # --- 5. VISUALIZZAZIONE BOX VALUTE (FONT BIANCO) ---
    st.subheader("⚡ Currency Strength Meter")
    cols = st.columns(6)
    # Mostriamo le top 6 in ordine decrescente nei box
    display_strength = strength_series.sort_values(ascending=False)
    for i, curr in enumerate(display_strength.index[:6]):
        val = display_strength[curr]
        col_c = "#00ffcc" if val > 0 else "#ff4b4b"
        cols[i].markdown(f"""
            <div style='text-align:center; border:1px solid #444; border-radius:10px; padding:12px; background:#1e1e1e;'>
                <b style='color: white; font-size: 16px;'>{curr}</b><br>
                <span style='color:{col_c}; font-size: 20px; font-weight: bold;'>{val:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # --- 6. GRAFICO A BARRE COMPARATIVO ---
    st.markdown(" ")
    with st.expander("📊 Analisi Comparativa Forza Valute", expanded=True):
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117') # Sfondo scuro come Streamlit
        ax.set_facecolor('#0e1117')
        
        colors = ['#00ffcc' if x > 0 else '#ff4b4b' for x in strength_series.values]
        bars = ax.barh(strength_series.index, strength_series.values, color=colors)
        
        # Estetica grafico
        ax.tick_params(axis='y', colors='white', labelsize=12)
        ax.tick_params(axis='x', colors='white')
        ax.spines['bottom'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#444')
        ax.axvline(0, color='white', linewidth=0.8, linestyle='--')
        
        st.pyplot(fig)

    # --- 7. SCORECARD & AI ---
    st.markdown("---")
    final_score = 50
    if drift > (pip_unit * 2): final_score += 20
    elif drift < -(pip_unit * 2): final_score -= 20
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Prezzo", price_fmt.format(recent_h[-1][0]))
    m2.metric("AI Pred. (+1h)", price_fmt.format(pred_1h), f"{drift:.5f}")
    m3.metric("Oracle Score", f"{final_score}/100")

    if final_score >= 80 or final_score <= 20:
        st.toast("🚨 SEGNALE OPERATIVO RILEVATO", icon="🎯")
        st.markdown(f'<audio autoplay><source src="https://www.soundjay.com/buttons/beep-07a.mp3" type="audio/mpeg"></audio>', unsafe_allow_html=True)

    st.line_chart(df_h['Close'].tail(50))

else:
    st.info("Caricamento dati in corso...")


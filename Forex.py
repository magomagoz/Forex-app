import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
from streamlit_autorefresh import st_autorefresh

# Aggiorna l'app ogni 5 minuti (300.000 millisecondi)
#st_autorefresh(interval=300000, key="datarefresh")

# 1. Funzione con Cache (scade ogni 10 minuti per non sovraccaricare)
@st.cache_data(ttl=600)
def get_clean_data(ticker):
    df = yf.download(ticker, period="2y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

# 2. Layout per il tasto di aggiornamento e timestamp
col_title, col_btn = st.columns([4, 1])

with col_title:
    st.title(f"Analisi Momentum: {pair}")

with col_btn:
    if st.button("🔄 AGGIORNA DATI"):
        st.cache_data.clear()  # Pulisce la cache per forzare il nuovo download
        st.rerun()

# Mostra l'ultimo aggiornamento
st.caption(f"Ultimo aggiornamento: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Forex Momentum Pro", layout="wide")

def get_clean_data(ticker):
    try:
        # Periodo 2y per avere abbastanza dati per medie mobili e indicatori
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty:
            return None
        
        # FIX: yfinance restituisce MultiIndex se non gestito
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Errore nel download dei dati: {e}")
        return None

def detect_divergence(df):
    """
    Rileva divergenze semplici tra Prezzo e RSI nelle ultime 10 sessioni
    """
    if len(df) < 15: return "Dati insufficienti"
    
    current_close = df['Close'].iloc[-1]
    prev_max_close = df['Close'].iloc[-11:-1].max()
    prev_min_close = df['Close'].iloc[-11:-1].min()
    
    current_rsi = df['RSI'].iloc[-1]
    prev_max_rsi = df['RSI'].iloc[-11:-1].max()
    prev_min_rsi = df['RSI'].iloc[-11:-1].min()
    
    # Divergenza Bearish (Prezzo sale, RSI scende)
    if current_close > prev_max_close and current_rsi < prev_max_rsi:
        return "Divergenza Bearish 📉"
    # Divergenza Bullish (Prezzo scende, RSI sale)
    elif current_close < prev_min_close and current_rsi > prev_min_rsi:
        return "Divergenza Bullish 📈"
    
    return "Nessuna Divergenza"

# --- SIDEBAR: INPUT UTENTE ---
st.sidebar.title("🛠 Configurazione Trader")
pair = st.sidebar.selectbox("Coppia Valutaria", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "EURGBP=X"])
balance = st.sidebar.number_input("Capitale Portafoglio ($)", value=10000)
risk_percent = st.sidebar.slider("Rischio per Operazione (%)", 0.1, 5.0, 1.0)

# --- MAIN APP ---
st.title(f"Analisi Momentum Giornaliero: {pair}")

data = get_clean_data(pair)

if data is not None:
    # --- CALCOLO INDICATORI ---
    data['RSI'] = ta.rsi(data['Close'], length=14)
    data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
    adx_df = ta.adx(data['High'], data['Low'], data['Close'], length=14)
    data['ADX'] = adx_df['ADX_14']

    # Valori attuali
    last_price = data['Close'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    last_adx = data['ADX'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    div_signal = detect_divergence(data)

    # --- VISUALIZZAZIONE METRICHE ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prezzo Attuale", f"{last_price:.4f}")
    col2.metric("RSI (14)", f"{last_rsi:.2f}")
    col3.metric("ADX (Trend Strength)", f"{last_adx:.2f}")
    col4.metric("Segnale Divergenza", div_signal)

    st.divider()

    # --- LOGICA DI TRADING & RISK MANAGEMENT ---
    st.subheader("🎯 Strategia e Punti di Entrata")
    
    # Esempio logica combinata: Momentum + Divergenza
    if "Bullish" in div_signal and last_rsi < 40:
        st.success("✅ SEGNALE BUY: Possibile inversione rialzista (Divergenza + Ipervenduto)")
        
        # Calcolo Livelli
        stop_loss = last_price - (last_atr * 2)
        take_profit = last_price + (last_atr * 4)
        
        # Calcolo Size (1 pip = 0.0001 per la maggior parte dei cross)
        risk_amount = balance * (risk_percent / 100)
        pip_distance = abs(last_price - stop_loss)
        # Semplificazione per lotti standard (10$ a pip per 1 lotto su EURUSD)
        position_size = risk_amount / (pip_distance * 100000) 
        
        st.write(f"**Entry Price:** {last_price:.4f}")
        st.write(f"**Stop Loss (2xATR):** {stop_loss:.4f}")
        st.write(f"**Take Profit (RR 1:2):** {take_profit:.4f}")
        st.info(f"💰 **Size Suggerita:** {position_size:.2f} Lotti per rischiare {risk_amount:.2f}$")

    elif "Bearish" in div_signal and last_rsi > 60:
        st.error("⚠️ SEGNALE SELL: Possibile inversione ribassista (Divergenza + Ipercomprato)")
        
        stop_loss = last_price + (last_atr * 2)
        take_profit = last_price - (last_atr * 4)
        
        risk_amount = balance * (risk_percent / 100)
        pip_distance = abs(last_price - stop_loss)
        position_size = risk_amount / (pip_distance * 100000)

        st.write(f"**Entry Price:** {last_price:.4f}")
        st.write(f"**Stop Loss (2xATR):** {stop_loss:.4f}")
        st.write(f"**Take Profit (RR 1:2):** {take_profit:.4f}")
        st.info(f"💰 **Size Suggerita:** {position_size:.2f} Lotti per rischiare {risk_amount:.2f}$")

    else:
        st.info("Market Watch: Nessuna divergenza rilevante al momento. Il momentum attuale è guidato dal trend principale.")

    # --- GRAFICO ---
    st.line_chart(data['Close'].tail(50))
else:
    st.warning("In attesa dei dati di mercato...")

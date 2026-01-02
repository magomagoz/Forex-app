import streamlit as st
import pandas_ta as ta
import yfinance as download
import yfinance as yf

st.title("Forex Momentum Analyzer 📈")

# Selezione Valuta
pair = st.selectbox("Seleziona Coppia", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"])

# Download Dati
data = download(pair, period="1y", interval="1d")

# Calcolo Indicatori
data['RSI'] = ta.rsi(data['Close'], length=14)
data['ADX'] = ta.adx(data['High'], data['Low'], data['Close'])['ADX_14']

# Logica di Trading (Esempio)
last_rsi = data['RSI'].iloc[-1]
last_adx = data['ADX'].iloc[-1]

if last_rsi > 60 and last_adx > 25:
    st.success(f"MOMENTUM RIALZISTA rilevato su {pair}. Cerca entrate Long sui pullback.")
elif last_rsi < 40 and last_adx > 25:
    st.error(f"MOMENTUM RIBASSISTA rilevato su {pair}. Cerca entrate Short.")
else:
    st.warning("Mercato in fase laterale. Attendi segnali più chiari.")

# --- FUNZIONE CALCOLO DIVERGENZE (Semplificata) ---
def check_divergence(df):
    # Consideriamo le ultime 10 candele per identificare i picchi
    current_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-10:-1].max()
    
    current_rsi = df['RSI'].iloc[-1]
    prev_rsi = df['RSI'].iloc[-10:-1].max()
    
    # Esempio Divergenza Bearish
    if current_close > prev_close and current_rsi < prev_rsi:
        return "Divergenza Bearish 📉"
    # Esempio Divergenza Bullish
    elif current_close < df['Close'].iloc[-10:-1].min() and current_rsi > df['RSI'].iloc[-10:-1].min():
        return "Divergenza Bullish 📈"
    return "Nessuna Divergenza Chiara"

# --- INTERFACCIA STREAMLIT ---
st.sidebar.header("Parametri Risk Management")
risk_pc = st.sidebar.slider("Rischio per operazione (%)", 0.5, 3.0, 1.0)
balance = st.sidebar.number_input("Capitale Totale ($)", value=10000)

# Calcolo ATR e Indicatori
data['RSI'] = ta.rsi(data['Close'], length=14)
data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)

div_signal = check_divergence(data)
last_atr = data['ATR'].iloc[-1]
last_price = data['Close'].iloc[-1]

# --- OUTPUT TRADING DESK ---
st.subheader(f"Analisi Tecnica: {div_signal}")

if "Bullish" in div_signal:
    # Calcolo Stop Loss basato su ATR (2 * ATR sotto il prezzo)
    stop_loss = last_price - (last_atr * 2)
    take_profit = last_price + (last_atr * 4) # Risk/Reward 1:2
    
    st.write(f"**Suggerimento Entry:** Buy a mercato ({last_price:.4f})")
    st.write(f"**Stop Loss Dinamico (ATR):** {stop_loss:.4f}")
    st.write(f"**Take Profit Suggerito:** {take_profit:.4f}")
    
    # Calcolo Size
    risk_amount = balance * (risk_pc / 100)
    pip_risk = abs(last_price - stop_loss)
    # Assumendo EURUSD (0.0001 = 1 pip)
    position_size = risk_amount / (pip_risk * 10) 
    st.info(f"Size consigliata per rischiare {risk_amount}$: {position_size:.2f} lotti")
    


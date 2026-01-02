import streamlit as st
import yfinance as yf
import pandas_ta as ta

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

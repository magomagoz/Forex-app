import streamlit as st
import pandas_ta as ta
import yfinance as download

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

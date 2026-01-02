import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt

def get_correlation_matrix(pairs_list):
    """Scarica i dati per più coppie e calcola la correlazione"""
    combined_data = pd.DataFrame()
    
    for p in pairs_list:
        df = yf.download(p, period="60d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        combined_data[p] = df['Close']
    
    return combined_data.corr()

# --- SEZIONE DA AGGIUNGERE NELL'INTERFACCIA ---
st.markdown("---")
st.subheader("📊 Matrice di Correlazione (Sentiment di Mercato)")

with st.expander("Analizza Correlazioni tra Major"):
    major_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X"]
    
    with st.spinner("Calcolo correlazioni in corso..."):
        corr_matrix = get_correlation_matrix(major_pairs)
        
        # Creazione del grafico con Matplotlib/Seaborn
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", center=0, ax=ax)
        plt.title("Correlazione a 60 giorni")
        st.pyplot(fig)
    
    st.info("""
    **Come leggere la mappa:**
    * **Vicino a 1 (Verde):** Le coppie si muovono insieme. 
    * **Vicino a -1 (Rosso):** Le coppie si muovono in direzioni opposte.
    * **Vicino a 0:** Movimenti indipendenti.
    """)

# --- 1. CONFIGURAZIONE PAGINA E STATO ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

# Inizializzazione Session State per il Reality Check
if 'prediction_log' not in st.session_state:
    st.session_state['prediction_log'] = None

def get_currency_strength():
    # Definiamo un set di coppie "cross" per isolare le valute
    tickers = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", 
        "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURJPY=X", 
        "GBPJPY=X", "EURGBP=X"
    ]
    data = yf.download(tickers, period="2d", interval="1d", progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    
    # Calcolo variazione percentuale
    returns = data.pct_change().iloc[-1] * 100
    
    # Isollamento delle singole valute (Semplificato)
    strength = {
        "USD": (-returns["EURUSD=X"] - returns["GBPUSD=X"] + returns["USDJPY=X"] - returns["AUDUSD=X"] + returns["USDCAD=X"] + returns["USDCHF=X"] - returns["NZDUSD=X"]) / 7,
        "EUR": (returns["EURUSD=X"] + returns["EURJPY=X"] + returns["EURGBP=X"]) / 3,
        "GBP": (returns["GBPUSD=X"] + returns["GBPJPY=X"] - returns["EURGBP=X"]) / 3,
        "JPY": (-returns["USDJPY=X"] - returns["EURJPY=X"] - returns["GBPJPY=X"]) / 3,
        "AUD": (returns["AUDUSD=X"]) / 1, # Semplificato
        "CAD": (-returns["USDCAD=X"]) / 1,
    }
    return pd.Series(strength).sort_values(ascending=False)

# --- VISUALIZZAZIONE IN STREAMLIT ---
st.markdown("---")
st.subheader("⚡ Currency Strength Meter (Intraday)")

with st.spinner("Calcolo forza valute..."):
    strength_data = get_currency_strength()
    
    # Creazione di colonne per i primi e gli ultimi
    col_str1, col_str2 = st.columns([2, 1])
    
    with col_str1:
        # Grafico a barre orizzontali
        fig_str, ax_str = plt.subplots(figsize=(8, 4))
        colors = ['green' if x > 0 else 'red' for x in strength_data.values]
        strength_data.plot(kind='barh', color=colors, ax=ax_str)
        ax_str.set_title("Forza Relativa %")
        ax_str.grid(axis='x', linestyle='--', alpha=0.7)
        st.pyplot(fig_str)

    with col_str2:
        st.write("**Top Momentum:**")
        st.success(f"🥇 {strength_data.index[0]}")
        st.write("**Worst Momentum:**")
        st.error(f"😴 {strength_data.index[-1]}")

st.info("**Strategia Professionale:** Cerca di accoppiare la valuta più forte (Top) con quella più debole (Worst). Quella coppia avrà il momentum più pulito e prevedibile.")

def calculate_squeeze(df):
    # Parametri standard: Bollinger (20, 2), Keltner (20, 1.5)
    length = 20
    mult_bb = 2
    mult_kc = 1.5
    
    # Bollinger Bands
    bb = ta.bbands(df['Close'], length=length, std=mult_bb)
    # Keltner Channels
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=length, scalar=mult_kc)
    
    # Logica di Squeeze
    # Se la banda superiore di BB è minore della superiore di KC 
    # E la banda inferiore di BB è maggiore della inferiore di KC
    is_sqz = (bb[f'BBU_{length}_{mult_bb}.0'] < kc[f'KCUe_{length}_{mult_kc}']) & \
             (bb[f'BBL_{length}_{mult_bb}.0'] > kc[f'KCLe_{length}_{mult_kc}'])
    
    return is_sqz.iloc[-1], is_sqz

# --- SEZIONE UI STREAMLIT ---
st.markdown("---")
st.subheader("🌋 Analisi Volatilità & Squeeze")

is_squeezing, squeeze_series = calculate_squeeze(df_d)

col_sqz1, col_sqz2 = st.columns([1, 2])

with col_sqz1:
    if is_squeezing:
        st.warning("⚠️ SQUEEZE ATTIVO: Il mercato sta comprimendo volatilità. Grande movimento in arrivo.")
    else:
        st.success("🚀 RELEASE: Il momentum è in fase di espansione.")

with col_sqz2:
    # Mostriamo un piccolo grafico dello stato di squeeze (ultimi 50 giorni)
    # 1 = Squeeze, 0 = No Squeeze
    st.line_chart(squeeze_series.tail(50).astype(int))
    st.caption("1.0 indica compressione (Squeeze), 0.0 indica espansione del prezzo.")

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


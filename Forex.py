import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, time
import pytz
import time as time_lib
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

def send_telegram_msg(msg):
    """Invia un avviso istantaneo su Telegram"""
    token = "IL_TUO_BOT_TOKEN" # Sostituisci con il tuo token
    chat_id = "IL_TUO_CHAT_ID" # Sostituisci con il tuo ID
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        requests.get(url, params=params, timeout=5)
    except Exception as e:
        print(f"Errore Telegram: {e}")

# --- 1. CONFIGURAZIONE & REFRESH ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

st.markdown("""
    <style>
        /* Riduce lo spazio sopra il titolo principale */
        .main .block-container {padding-top: 1rem !important;}
        /* Riduce lo spazio sopra la sidebar */
        [data-testid="stSidebar"] > div:first-child {padding-top: 0rem !important;}
        /* Rende l'header di Streamlit meno ingombrante */
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Definizione Fuso Orario Roma
rome_tz = pytz.timezone('Europe/Rome')
asset_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X", "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"}

# Refresh automatico ogni 60 secondi
st_autorefresh(interval=60 * 1000, key="sentinel_refresh")

if 'signal_history' not in st.session_state: 
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
if 'last_alert' not in st.session_state:
    st.session_state['last_alert'] = None
if 'last_scan_status' not in st.session_state:
    st.session_state['last_scan_status'] = "In attesa..."

# --- 2. FUNZIONI TECNICHE ---
def get_now_rome():
    return datetime.now(rome_tz)

def play_notification_sound():
    audio_html = """
        <audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg"></audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def get_session_status():
    now_rome = get_now_rome().time()
    sessions = {
        "Tokyo 🇯🇵": (time(0,0), time(9,0)), 
        "Londra 🇬🇧": (time(9,0), time(18,0)), 
        "New York 🇺🇸": (time(14,0), time(23,0))
    }
    return {name: start <= now_rome <= end for name, (start, end) in sessions.items()}

def is_low_liquidity():
    now_rome = get_now_rome().time()
    return time(23, 0) <= now_rome or now_rome <= time(1, 0)

@st.cache_data(ttl=30)
def get_realtime_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, timeout=10)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def get_currency_strength():
    try:
        # Lista asset completa
        forex = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURCHF=X","EURJPY=X", "GBPJPY=X", "GBPCHF=X","EURGBP=X"]
        crypto = ["BTC-USD", "ETH-USD"]
        
        # Scarichiamo 5 giorni per essere sicuri di avere dati anche nel weekend
        data = yf.download(forex + crypto, period="5d", interval="1d", progress=False, timeout=15)
        
        if data is None or data.empty: 
            return pd.Series(dtype=float)

        # Gestione colonne MultiIndex (fix per versioni recenti di yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            # Se esiste il livello 'Close', usiamo quello
            if 'Close' in data.columns.get_level_values(0):
                close_data = data['Close']
            elif 'Close' in data.columns.get_level_values(1):
                 # Caso invertito (Ticker, Price)
                 close_data = data.xs('Close', axis=1, level=1)
            else:
                 # Tentativo generico di prendere la chiusura
                 close_data = data['Close']
        else:
            close_data = data['Close'] if 'Close' in data else data

        # Pulizia dati: riempiamo i buchi e prendiamo l'ultima variazione valida
        close_data = close_data.ffill().dropna()
        
        if len(close_data) < 2: return pd.Series(dtype=float)

        # Calcolo variazione percentuale
        returns = close_data.pct_change().iloc[-1] * 100
        
        # Calcolo forza valute (usando .get per evitare crash se manca un ticker)
        strength = {
            "USD 🇺🇸": (-returns.get("EURUSD=X",0) - returns.get("GBPUSD=X",0) + returns.get("USDJPY=X",0) - returns.get("AUDUSD=X",0) + returns.get("USDCAD=X",0) + returns.get("USDCHF=X",0) - returns.get("NZDUSD=X",0)) / 7,
            "EUR 🇪🇺": (returns.get("EURUSD=X",0) + returns.get("EURJPY=X",0) + returns.get("EURGBP=X",0) + returns.get("EURCHF=X", 0)) / 4,
            "GBP 🇬🇧": (returns.get("GBPUSD=X",0) + returns.get("GBPJPY=X",0) - returns.get("EURGBP=X",0) + returns.get("GBPCHF=X", 0)) / 4,
            "JPY 🇯🇵": (-returns.get("USDJPY=X",0) - returns.get("EURJPY=X",0) - returns.get("GBPJPY=X",0)) / 3,
            "CHF 🇨🇭": (-returns.get("USDCHF=X",0) - returns.get("EURCHF=X",0) - returns.get("GBPCHF=X",0)) / 3,
            "AUD 🇦🇺": returns.get("AUDUSD=X", 0),
            "CAD 🇨🇦": -returns.get("USDCAD=X", 0),
            "BTC ₿": returns.get("BTC-USD", 0),
            "ETH 💎": returns.get("ETH-USD", 0)
        }
        return pd.Series(strength).sort_values(ascending=False)
    except Exception as e:
        # Se vuoi vedere l'errore nel terminale per debug, togli il commento sotto:
        # print(f"Errore Currency Strength: {e}")
        return pd.Series(dtype=float)

def get_asset_params(pair):
    if "-" in pair: return 1.0, "{:.2f}", 1, "CRYPTO"
    if "JPY" in pair: return 0.01, "{:.2f}", 1000, "FOREX" 
    return 0.0001, "{:.5f}", 10, "FOREX"

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi_col = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi_col.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi_col.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DECRESCITA"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 CRESCITA"
    return "Neutrale"

def get_win_rate():
    if st.session_state['signal_history'].empty:
        return "Nessun dato"
    df = st.session_state['signal_history']
    total = len(df[df['Stato'] != 'In Corso'])
    if total == 0: return "In attesa di chiusure..."
    wins = len(df[df['Stato'] == '✅ TARGET'])
    wr = (wins / total) * 100
    return f"Win Rate: {wr:.1f}% ({wins}/{total})"

# --- 3. MOTORI DI BACKGROUND ---
def update_signal_outcomes():
    if st.session_state['signal_history'].empty: return
    df = st.session_state['signal_history']
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            data = yf.download(asset_map[row['Asset']], period="1d", interval="1m", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                high, low = data['High'].max(), data['Low'].min()
                sl_v, tp_v = float(row['SL']), float(row['TP'])
                if row['Direzione'] == 'COMPRA':
                    if high >= tp_v: df.at[idx, 'Stato'] = '✅ TARGET'
                    elif low <= sl_v: df.at[idx, 'Stato'] = '❌ STOP LOSS'
                else:
                    if low <= tp_v: df.at[idx, 'Stato'] = '✅ TARGET'
                    elif high >= sl_v: df.at[idx, 'Stato'] = '❌ STOP LOSS'
        except: continue

def run_sentinel():
    for label, ticker in asset_map.items():
        try:
            df_rt_s = yf.download(ticker, period="2d", interval="1m", progress=False)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df_rt_s.empty or df_d_s.empty: continue
                
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            if isinstance(df_d_s.columns, pd.MultiIndex): df_d_s.columns = df_d_s.columns.get_level_values(0)
            df_rt_s.columns = [c.lower() for c in df_rt_s.columns]
            df_d_s.columns = [c.lower() for c in df_d_s.columns]
            
            bb_s = ta.bbands(df_rt_s['close'], length=20, std=2)
            c_l = [c for c in bb_s.columns if "BBL" in c.upper()][0]
            c_u = [c for c in bb_s.columns if "BBU" in c.upper()][0]
            
            c_v = float(df_rt_s['close'].iloc[-1])
            l_bb = float(bb_s[c_l].iloc[-1])
            u_bb = float(bb_s[c_u].iloc[-1])
            
            rsi_s = ta.rsi(df_d_s['close'], length=14).iloc[-1]
            atr_s = ta.atr(df_d_s['high'], df_d_s['low'], df_d_s['close'], length=14).iloc[-1]
            adx_df = ta.adx(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1]
              
            s_action = None
            if c_v < l_bb and rsi_s < 45 and curr_adx < 30: s_action = "COMPRA"
            elif c_v > u_bb and rsi_s > 55 and curr_adx < 30: s_action = "VENDI"
            
            if s_action:
                hist = st.session_state['signal_history']
                if hist.empty or not ((hist['Asset'] == label) & (hist['Direzione'] == s_action)).head(1).any():
                    p_unit, p_fmt, p_mult = get_asset_params(ticker)[:3]
                    sl = c_v - (1.5 * atr_s) if s_action == "COMPRA" else c_v + (1.5 * atr_s)
                    tp = c_v + (3 * atr_s) if s_action == "COMPRA" else c_v - (3 * atr_s)
                    risk_val = balance * (risk_pc / 100)
                    dist_p = abs(c_v - sl) * p_mult
                    sz = risk_val / (dist_p * 10) if dist_p > 0 else 0
                    
                    new_sig = {'DataOra': get_now_rome().strftime("%d/%m/%Y %H:%M:%S"), 'Asset': label, 'Direzione': s_action, 'Prezzo': p_fmt.format(c_v), 'SL': p_fmt.format(sl), 'TP': p_fmt.format(tp), 'Size': f"{sz:.2f}", 'Stato': 'In Corso'}
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    st.session_state['last_alert'] = new_sig
                    st.rerun()
            st.session_state['last_scan_status'] = f"✅ {label} analizzato"
        except Exception as e:
            st.session_state['last_scan_status'] = f"⚠️ Errore su {label}"
            continue

# ESECUZIONE MOTORI
update_signal_outcomes()
run_sentinel()

# --- 4. SIDEBAR ---
st.sidebar.header("🛠 Trading Desk (1m)")
if "start_time" not in st.session_state: st.session_state.start_time = time_lib.time()
countdown = 60 - int(time_lib.time() - st.session_state.start_time) % 60
st.sidebar.markdown(f"⏳ **Prossimo Scan: {countdown}s**")

st.sidebar.subheader("📡 Sentinel Status")
status = st.session_state.get('last_scan_status', 'In attesa...')
st.sidebar.code(status)

selected_label = st.sidebar.selectbox("**Asset**", list(asset_map.keys()))
pair = asset_map[selected_label]
balance = st.sidebar.number_input("**Conto (€)**", value=1000)
risk_pc = st.sidebar.slider("**Rischio %**", 0.5, 5.0, 1.0)

st.sidebar.subheader("🌍 Sessioni di Mercato")
for s_name, is_open in get_session_status().items():
    color = "🟢" if is_open else "🔴"
    status_text = "OPEN" if is_open else "CLOSED"
    st.sidebar.markdown(f"{color} **{s_name}**: {status_text}")

# Visualizzazione Win Rate in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 **Performance Oggi**")
wr = get_win_rate()
if wr:
    st.sidebar.info(wr)

# --- MODIFICA SIDEBAR: RESET CON SICUREZZA ---
st.sidebar.markdown("---")
#st.sidebar.subheader("⚙️ Gestione Dati")
# Usiamo un popover per evitare click accidentali
with st.sidebar.popover("🗑️ **Reset Cronologia**"):
    st.warning("Sei sicuro? Questa azione cancellerà tutti i segnali salvati.")
    if st.button("SÌ, CANCELLA ORA"):
        st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
        st.session_state['last_alert'] = None
        st.rerun()

st.sidebar.markdown("---")

# --- 5. POPUP ALERT (RE-FIXED) ---
if st.session_state['last_alert']:
    play_notification_sound()
    alert = st.session_state['last_alert']

    # Popup HTML con z-index leggermente ridotto per non coprire i tasti Streamlit
    st.markdown(f"""
        <div style="position: fixed; top: 45%; left: 50%; transform: translate(-50%, -50%); width: 85vw; max-width: 700px; background-color: rgba(15, 12, 41, 0.98); z-index: 999; display: flex; flex-direction: column; justify-content: center; align-items: center; color: white; text-align: center; padding: 30px; border: 4px solid #00ffcc; border-radius: 30px; box-shadow: 0 0 50px rgba(0, 255, 204, 0.5); backdrop-filter: blur(10px); font-family: sans-serif;">
            <h1 style="font-size: 2.2em; color: #00ffcc; margin-bottom:5px;">🚀 SEGNALE RILEVATO</h1>
            <p style="font-size: 0.9em; color: #888;">{alert['DataOra']}</p>
            <hr style="width: 80%; border: 0.5px solid #333; margin: 15px 0;">
            <h2 style="font-size: 2.5em; margin: 10px 0;">{alert['Asset']} <span style="color:{'#00ffcc' if alert['Direzione'] == 'COMPRA' else '#ff4b4b'}">{alert['Direzione']}</span></h2>
            <div style="background: rgba(34, 34, 34, 0.8); padding: 20px; border-radius:20px; border: 1px solid #ffcc00; width: 90%; margin: 15px 0;">
                <p style="font-size: 2em; color: #ffcc00; font-weight: bold; margin:0;">SIZE: {alert['Size']}</p>
                <p style="font-size: 1.2em; margin: 5px 0;">Entry: {alert['Prezzo']}</p>
                <p style="font-size: 0.9em; color: #aaa;">SL: {alert['SL']} | TP: {alert['TP']}</p>
            </div>
            <p style="color: #666; font-style: italic; font-size: 0.8em;">Clicca il tasto qui sotto per chiudere</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Il bottone deve essere fuori dal markdown ma visibile
    if st.button("✅ CHIUDI E TORNA AL MONITORAGGIO", use_container_width=True, type="primary"):
        st.session_state['last_alert'] = None
        st.rerun()

# --- 6. HEADER E GRAFICO AVANZATO (Con RSI) ---
st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1><p style="color: white; opacity: 0.8; margin:0;">Sentinel AI Engine • Forex & Crypto Analysis</p></div>', unsafe_allow_html=True)

p_unit, price_fmt, p_mult, a_type = get_asset_params(pair)
df_rt = get_realtime_data(pair) 
df_d = yf.download(pair, period="1y", interval="1d", progress=False)

if df_rt is not None and not df_rt.empty and df_d is not None and not df_d.empty:
    # 1. PRE-CALCOLO INDICATORI (Fondamentale per evitare NameError)
    if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
    df_d.columns = [c.lower() for c in df_d.columns]
    
    # Calcolo indicatori Real Time (5m)
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    df_rt['rsi'] = ta.rsi(df_rt['close'], length=14)
    
    # Calcolo indicatori Daily (per score e divergenza)
    df_d['rsi'] = ta.rsi(df_d['close'], length=14)
    df_d['atr'] = ta.atr(df_d['high'], df_d['low'], df_d['close'], length=14)
          
    # 1. Definizioni Colonne Bande
    c_up = [c for c in df_rt.columns if "BBU" in c.upper()][0]
    c_mid = [c for c in df_rt.columns if "BBM" in c.upper()][0]
    c_low = [c for c in df_rt.columns if "BBL" in c.upper()][0]
    
    # 2. Calcolo Variabili e Score (FONDAMENTALE PER EVITARE ERRORI)
    curr_p = float(df_rt['close'].iloc[-1])
    curr_rsi = float(df_rt['rsi'].iloc[-1])
    rsi_val = float(df_d['rsi'].iloc[-1]) 
    last_atr = float(df_d['atr'].iloc[-1])
    
    # Calcolo Score
    score = 50 + (20 if curr_p < df_rt[c_low].iloc[-1] else -20 if curr_p > df_rt[c_up].iloc[-1] else 0)

    # 3. Creazione Grafico
    p_df = df_rt.tail(60)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])
    
    # Candele
    fig.add_trace(go.Candlestick(
        x=p_df.index, open=p_df['open'], high=p_df['high'], 
        low=p_df['low'], close=p_df['close'], name='Prezzo'
    ), row=1, col=1)
    
    # Banda Superiore (Solo riga)
    fig.add_trace(go.Scatter(
        x=p_df.index, y=p_df[c_up], 
        line=dict(color='rgba(173, 216, 230, 0.6)', width=1), 
        name='Upper BB'
    ), row=1, col=1)
    
    # Banda Mediana (Base per il riempimento)
    fig.add_trace(go.Scatter(
        x=p_df.index, y=p_df[c_mid], 
        line=dict(color='rgba(255, 255, 255, 0.3)', width=1), 
        name='BBM (Mediana)'
    ), row=1, col=1)
    
    # Banda Inferiore (Riempita in Celeste verso la Mediana)
    fig.add_trace(go.Scatter(
        x=p_df.index, y=p_df[c_low], 
        line=dict(color='rgba(0, 191, 255, 0.6)', width=1.5), 
        fill='tonexty', 
        fillcolor='rgba(0, 191, 255, 0.15)', 
        name='Lower BB (Buy Zone)'
    ), row=1, col=1)


    # RSI
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['rsi'], line=dict(color='#ffcc00', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00ff00", row=2, col=1)

    
    # RSI (Riga 2)
    #fig.add_trace(go.Scatter(x=p_df.index, y=p_df['rsi'], line=dict(color='#ffcc00', width=2), name='RSI'), row=2, col=1)
    #fig.add_hline(y=75, line_dash="dot", line_color="red", row=2, col=1)
    #fig.add_hline(y=25, line_dash="dot", line_color="#00ff00", row=2, col=1)
    #fig.add_hrect(y0=25, y1=75, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)

    # --- AGGIUNTA GRIGLIA VERTICALE (OGNI 10 MINUTI) ---
    # Scansioniamo l'indice delle candele visibili
    for t in p_df.index:
        # Se il minuto è multiplo di 10 (0, 10, 20, 30...)
        if t.minute % 10 == 0:
            fig.add_vline(
                x=t, 
                line_width=1, 
                line_dash="dot", 
                line_color="rgba(0, 0, 0, 0.15)" # Bianco molto trasparente
            )
        
    # Layout e Visualizzazione
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # 4. Metriche
    c_met1, c_met2 = st.columns(2)
    c_met1.metric(label=f"Prezzo {selected_label}", value=price_fmt.format(curr_p))
    c_met2.metric(label="RSI (5m)", value=f"{curr_rsi:.1f}", delta="Ipercomprato" if curr_rsi > 70 else "Ipervenduto" if curr_rsi < 30 else "Neutro", delta_color="inverse")
    
    st.caption(f"📢 RSI Daily: {rsi_val:.1f} | Divergenza: {detect_divergence(df_d)}")

    # --- VISUALIZZAZIONE METRICHE AI & ADX ---
    adx_df_ai = ta.adx(df_rt['high'], df_rt['low'], df_rt['close'], length=14)
    curr_adx_ai = adx_df_ai['ADX_14'].iloc[-1]

    st.markdown("---")
    st.subheader("🕵️ Sentinel Market Analysis")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("RSI Daily", f"{rsi_val:.1f}", detect_divergence(df_d))
    col_b.metric("Sentinel Score", f"{score}/100")
    adx_emoji = "🔴" if curr_adx_ai > 30 else "🟡" if curr_adx_ai > 20 else "🟢"
    col_c.metric("Forza Trend (ADX)", f"{curr_adx_ai:.1f}", adx_emoji)

    st.markdown("### 📊 Guida alla Volatilità (ADX)")
    adx_guide = pd.DataFrame([
        {"Valore": "0 - 20", "Stato": "🟢 Laterale", "Affidabilità": "MASSIMA"},
        {"Valore": "20 - 30", "Stato": "🟡 In formazione", "Affidabilità": "MEDIA"},
        {"Valore": "30+", "Stato": "🔴 Trend Forte", "Affidabilità": "BASSA"}
    ])

    def highlight_adx(row):
        if curr_adx_ai <= 20 and "0 - 20" in row['Valore']: return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
        elif 20 < curr_adx_ai <= 30 and "20 - 30" in row['Valore']: return ['background-color: rgba(255, 255, 0, 0.2)'] * len(row)
        elif curr_adx_ai > 30 and "30+" in row['Valore']: return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
        return [''] * len(row)

    st.table(adx_guide.style.apply(highlight_adx, axis=1))

# --- 7. CURRENCY STRENGTH ---
# Deve essere fuori da ogni IF, allineato tutto a sinistra
st.markdown("---")
st.subheader("⚡ Currency Strength Meter")
s_data = get_currency_strength()

if not s_data.empty:
    cols = st.columns(len(s_data))
    for i, (curr, val) in enumerate(s_data.items()):
        bg = "#006400" if val > 0.15 else "#8B0000" if val < -0.15 else "#333333"
        txt_c = "#00FFCC" if val > 0.15 else "#FF4B4B" if val < -0.15 else "#FFFFFF"
        cols[i].markdown(
            f"<div style='text-align:center; background:{bg}; padding:6px; border-radius:8px; border:1px solid {txt_c}; min-height:80px;'>"
            f"<b style='color:white; font-size:0.8em;'>{curr}</b><br>"
            f"<span style='color:{txt_c};'>{val:.2f}%</span></div>", 
            unsafe_allow_html=True
        )
else:
    st.info("⏳ Caricamento dati macro in corso...")

# --- 8. ANALISI AI E GENERAZIONE SEGNALI ---
# Questa sezione usa i dati già calcolati nella Sezione 6

if df_rt is not None and df_d is not None and not df_d.empty:
    # Controlliamo se siamo in una fascia oraria con liquidità sufficiente
    if not is_low_liquidity():
        # LOGICA SEGNALE: Incrocio tra Sentinel Score (Sentiment) e RSI Daily (Trend)
        action = "COMPRA" if (score >= 65 and rsi_val < 60) else "VENDI" if (score <= 35 and rsi_val > 40) else None
        
        if action:
            hist = st.session_state['signal_history']
            
            # Evitiamo di duplicare lo stesso segnale se è già presente e attivo
            if hist.empty or not ((hist['Asset'] == selected_label) & (hist['Direzione'] == action) & (hist['Stato'] == 'In Corso')).any():
                
                # Calcolo SL e TP basato sulla volatilità (ATR)
                sl = curr_p - (1.5 * last_atr) if action == "COMPRA" else curr_p + (1.5 * last_atr)
                tp = curr_p + (3 * last_atr) if action == "COMPRA" else curr_p - (3 * last_atr)
                
                # Calcolo della Size (Lotti) in base al rischio impostato
                dist_p = abs(curr_p - sl) * p_mult
                size = (balance * (risk_pc / 100)) / (dist_p * 10) if dist_p > 0 else 0
                
                # Creazione del nuovo record segnale
                new_a = {
                    'DataOra': get_now_rome().strftime("%d/%m %H:%M:%S"), 
                    'Asset': selected_label, 
                    'Direzione': action, 
                    'Prezzo': price_fmt.format(curr_p), 
                    'SL': price_fmt.format(sl), 
                    'TP': price_fmt.format(tp), 
                    'Size': f"{size:.2f}", 
                    'Stato': 'In Corso'
                }
                
                # Aggiornamento stato e trigger Alert
                st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_a]), hist], ignore_index=True)
                st.session_state['last_alert'] = new_a
                st.rerun()

# --- FOOTER INFORMATIVO ---
st.markdown("---")
st.info(f"🛰️ **Sentinel AI Engine Attiva**: Monitoraggio in corso su {len(asset_map)} asset in tempo reale (1m).")
st.caption(f"Ultimo aggiornamento globale: {get_now_rome().strftime('%d/%m/%Y %H:%M:%S')}")

# --- 9. CRONOLOGIA SEGNALI (CORRETTA) ---
st.markdown("---")
st.subheader("📜 Cronologia Segnali")

if not st.session_state['signal_history'].empty:
    # Creiamo una copia per la visualizzazione
    display_df = st.session_state['signal_history'].copy()
    
    def style_status(val):
        color = '#00ffcc' if '✅' in val else '#ff4b4b' if '❌' in val else '#ffcc00'
        return f'color: {color}; font-weight: bold'
    
    # Visualizzazione tabella
    st.dataframe(
        display_df.style.map(style_status, subset=['Stato']), 
        use_container_width=True,
        hide_index=True
    )
    
    # TASTO DOWNLOAD (Come richiesto)
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Esporta Cronologia (CSV)",
        csv,
        "cronologia_forex.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.write("Nessun segnale rilevato finora. In attesa di opportunità...")

st.markdown("---")

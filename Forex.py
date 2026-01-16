import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, time
import pytz
import time as time_lib
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os

# --- 1. CONFIGURAZIONE & LAYOUT ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")

# CSS Migliorato
st.markdown("""
    <style>
        .block-container {padding-top: 1rem !important;}
        [data-testid="stSidebar"] > div:first-child {padding-top: 0rem !important;}
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background-color: rgba(0,0,0,0) !important;} 
        
        /* Stile Tasti */
        div.stButton > button {
            border-radius: 8px !important;
            font-weight: bold;
            width: 100%;
        }
        
        /* Colori Tabella */
        [data-testid="stDataFrame"] {
            border: 1px solid #333;
        }
    </style>
""", unsafe_allow_html=True)

# Definizione Fuso Orario Roma
rome_tz = pytz.timezone('Europe/Rome')
asset_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X", "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"}

# Refresh automatico ogni 30 secondi
st_autorefresh(interval=30 * 1000, key="sentinel_refresh")

# --- INIZIALIZZAZIONE STATO (Session State) ---
if 'signal_history' not in st.session_state: 
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
if 'last_alert' not in st.session_state:
    st.session_state['last_alert'] = None
if 'last_scan_status' not in st.session_state:
    st.session_state['last_scan_status'] = "In attesa..."

# --- 2. FUNZIONI TECNICHE ---

def send_telegram_msg(msg):
    """Invia un avviso istantaneo su Telegram"""
    token = "8235666467:AAGCsvEhlrzl7bH537bJTjsSwQ3P3PMRW10" 
    chat_id = "7191509088" 
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        requests.get(url, params=params, timeout=5)
    except Exception as e:
        print(f"Errore Telegram: {e}")

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
        forex = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURCHF=X","EURJPY=X", "GBPJPY=X", "GBPCHF=X","EURGBP=X"]
        crypto = ["BTC-USD", "ETH-USD"]
        data = yf.download(forex + crypto, period="5d", interval="1d", progress=False, timeout=15)
        
        if data is None or data.empty: 
            return pd.Series(dtype=float)

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0): close_data = data['Close']
            else: close_data = data['Close'] if 'Close' in data else data
        else:
            close_data = data['Close'] if 'Close' in data else data

        close_data = close_data.ffill().dropna()
        if len(close_data) < 2: return pd.Series(dtype=float)

        returns = close_data.pct_change().iloc[-1] * 100
        
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
    except Exception:
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

def update_signal_outcomes():
    """Controlla se i segnali aperti hanno raggiunto TP o SL"""
    if st.session_state['signal_history'].empty: return
    df = st.session_state['signal_history']
    
    updates_made = False
    # Itera solo su quelli in corso
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            ticker = asset_map[row['Asset']]
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                # Prendiamo i dati più recenti per controllare
                high, low = data['High'].max(), data['Low'].min()
                sl_v, tp_v = float(row['SL']), float(row['TP'])
                
                new_status = None
                if row['Direzione'] == 'COMPRA':
                    if high >= tp_v: new_status = '✅ TARGET'
                    elif low <= sl_v: new_status = '❌ STOP LOSS'
                else: # VENDI
                    if low <= tp_v: new_status = '✅ TARGET'
                    elif high >= sl_v: new_status = '❌ STOP LOSS'
                
                if new_status:
                    df.at[idx, 'Stato'] = new_status
                    updates_made = True
        except: continue
    
    if updates_made:
        st.session_state['signal_history'] = df

def run_sentinel():
    """Scansiona tutti gli asset definiti in asset_map"""
    assets = list(asset_map.items())
    for label, ticker in assets:
        try:
            # Scarichiamo i dati necessari per l'analisi
            df_rt_s = yf.download(ticker, period="2d", interval="1m", progress=False)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df_rt_s.empty or df_d_s.empty: continue
            
            # Normalizzazione colonne
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            df_rt_s.columns = [c.lower() for c in df_rt_s.columns]
            df_d_s.columns = [c.lower() for c in df_d_s.columns]

            # Calcolo indicatori Sentinel
            bb_s = ta.bbands(df_rt_s['close'], length=20, std=2)
            c_low = [c for c in bb_s.columns if "BBL" in c.upper()][0]
            c_up = [c for c in bb_s.columns if "BBU" in c.upper()][0]
            
            curr_v = float(df_rt_s['close'].iloc[-1])
            low_bb = float(bb_s[c_low].iloc[-1])
            up_bb = float(bb_s[c_up].iloc[-1])
            
            rsi_d = ta.rsi(df_d_s['close'], length=14).iloc[-1]
            atr_d = ta.atr(df_d_s['high'], df_d_s['low'], df_d_s['close'], length=14).iloc[-1]
            adx_df = ta.adx(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1]

            # Logica Segnale
            s_action = None
            if curr_v < low_bb and rsi_d < 45 and curr_adx < 30: s_action = "COMPRA"
            elif curr_v > up_bb and rsi_d > 55 and curr_adx < 30: s_action = "VENDI"

            if s_action:
                hist = st.session_state['signal_history']
                # Evita duplicati se c'è già un segnale "In Corso" per lo stesso asset
                if hist.empty or not ((hist['Asset'] == label) & (hist['Stato'] == 'In Corso')).any():
                    p_unit, p_fmt, p_mult = get_asset_params(ticker)[:3]
                    sl = curr_v - (1.5 * atr_d) if s_action == "COMPRA" else curr_v + (1.5 * atr_d)
                    tp = curr_v + (3 * atr_d) if s_action == "COMPRA" else curr_v - (3 * atr_d)
                    
                    # Calcolo Size automatica (basata sul rischio impostato in sidebar)
                    risk_val = balance * (risk_pc / 100)
                    dist_p = abs(curr_v - sl) * p_mult
                    sz = risk_val / (dist_p * 10) if dist_p > 0 else 0
                    
                    new_sig = {
                        'DataOra': get_now_rome().strftime("%H:%M:%S"),
                        'Asset': label, 
                        'Direzione': s_action, 
                        'Prezzo': p_fmt.format(curr_v), 
                        'SL': p_fmt.format(sl), 
                        'TP': p_fmt.format(tp), 
                        'Size': f"{sz:.2f}", 
                        'Stato': 'In Corso'
                    }

        
                    
                    
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    st.session_state['last_alert'] = new_sig
                    send_telegram_msg(f"🚀 *{s_action}* {label}\nPrezzo: {new_sig['Prezzo']}")
                    st.rerun()

            now = get_now_rome().strftime("%H:%M:%S")
            st.session_state['last_scan_status'] = f"🔍 {label}: Analisi completata ({now})"

            #st.session_state['last_scan_status'] = f"✅ {label} Analizzato"
        except Exception as e:
        # Invece di un errore generico, mostriamo cosa è andato storto (es. timeout o dati mancanti)
            error_type = "Timeout" if "timeout" in str(e).lower() else "Dati non disp."
            st.session_state['last_scan_status'] = f"⚠️ {label}: {error_type}"
            continue


def get_win_rate():
    if st.session_state['signal_history'].empty:
        return "Nessun dato"
    df = st.session_state['signal_history']
    # Consideriamo conclusi solo quelli che non sono "In Corso"
    closed_trades = df[df['Stato'] != 'In Corso']
    total = len(closed_trades)
    
    if total == 0: return "In attesa di chiusure..."
    
    wins = len(closed_trades[closed_trades['Stato'] == '✅ TARGET'])
    wr = (wins / total) * 100
    return f"Win Rate: {wr:.1f}% ({wins}/{total})"

# --- 5. POPUP ALERT ---
if st.session_state['last_alert']:
    play_notification_sound()
    alert = st.session_state['last_alert']
    main_color = "#00ffcc" if alert['Direzione'] == 'COMPRA' else "#ff4b4b"

    # Overlay Scuro e stile bottone
    st.markdown(f"""
        <style>
            .full-overlay {{
                position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                background: rgba(0,0,0,0.85); z-index: 9999;
                display: flex; flex-direction: column; align-items: center; justify-content: center;
            }}
            .close-trigger {{
                margin-top: 20px; /* Spazio sotto il popup */
                z-index: 10001;
            }}
            /* Stile forzato del bottone Streamlit */
            div.stButton > button:first-child {{
                background-color: #111 !important;
                color: white !important;
                border: 2px solid {main_color} !important;
                padding: 10px 40px !important;
                font-weight: bold !important;
            }}
        </style>
        <div class="full-overlay"></div>
    """, unsafe_allow_html=True)

    # Contenitore Grafico Popup
    st.markdown(f"""
    <div style="position: fixed; top: 45%; left: 50%; transform: translate(-50%, -50%); 
                width: 85%; max-width: 500px; background: #111; 
                border: 3px solid {main_color}; border-radius: 20px; 
                padding: 25px; text-align: center; z-index: 10000;
                box-shadow: 0 0 40px {main_color}33;">
        <h2 style="color: {main_color}; margin:0;">{alert['Asset']}</h2>
        <h1 style="color: white; font-size: 3em; margin: 10px 0;">🚀 {alert['Direzione']}</h1>
        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px;">
            <p style="color: #FFD700; font-size: 1.2em; margin:0;">SIZE: {alert['Size']} LOTTI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pulsante di chiusura posizionato dinamicamente sotto
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Questo placeholder invisibile spinge il bottone in primo piano sopra l'overlay
        st.markdown('<div style="height: 65vh;"></div>', unsafe_allow_html=True)
        if st.button("❌ CHIUDI E TORNA AL MONITOR", key="btn_close_final"):
            st.session_state['last_alert'] = None
            st.rerun()
    
    st.stop()

# --- 3. ESECUZIONE AGGIORNAMENTO DATI (PRIMA DELLA GUI) ---
# Importante: Aggiorniamo i risultati TP/SL prima di disegnare la sidebar
update_signal_outcomes()

# --- 4. SIDEBAR ---
st.sidebar.header("🛠 Trading Desk (30s)")

# Countdown Testuale e Barra Rossa Animata
st.sidebar.markdown("⏳ **Prossimo Scan**")

# CSS per la barra che si riempie in 30 secondi
st.sidebar.markdown("""
    <style>
        @keyframes progressFill {
            0% { width: 0%; }
            100% { width: 100%; }
        }
        .container-bar {
            width: 100%; background-color: #222; border-radius: 5px;
            height: 12px; margin-bottom: 25px; border: 1px solid #444; overflow: hidden;
        }
        .red-bar {
            height: 100%; background-color: #ff4b4b; width: 0%;
            animation: progressFill 30s linear infinite;
            box-shadow: 0 0 10px #ff4b4b;
        }
    </style>
    <div class="container-bar"><div class="red-bar"></div></div>
""", unsafe_allow_html=True)

st.sidebar.subheader("📡 Sentinel Status")
status = st.session_state.get('last_scan_status', 'In attesa...')

# Usiamo un contenitore con colore dinamico
if "⚠️" in status:
    st.sidebar.error(status)
elif "🔍" in status:
    st.sidebar.success(status)
else:
    st.sidebar.info(status)

# Parametri Input
selected_label = st.sidebar.selectbox("**Asset**", list(asset_map.keys()))
pair = asset_map[selected_label]
balance = st.sidebar.number_input("**Conto (€)**", value=1000)
risk_pc = st.sidebar.slider("**Rischio %**", 0.5, 5.0, 1.0)

st.sidebar.markdown("---")
# ... (restante codice sidebar: sessioni, win rate, reset)
st.sidebar.subheader("🌍 Sessioni di Mercato")
for s_name, is_open in get_session_status().items():
    color = "🟢" if is_open else "🔴"
    status_text = "APERTO" if is_open else "CHIUSO"
    st.sidebar.markdown(f"**{s_name}** <small>({status_text})</small>      {color}",
unsafe_allow_html=True)

# Win Rate Sidebar - Ora mostrerà i dati aggiornati
st.sidebar.markdown("---")
st.sidebar.subheader("🏆 **Performance Oggi**")
wr = get_win_rate()
if wr:
    st.sidebar.info(wr)

# Reset Sidebar
st.sidebar.markdown("---")
with st.sidebar.popover("🗑️ **Reset Cronologia**"):
    st.warning("Sei sicuro? Questa azione cancellerà tutti i segnali salvati.")
    if st.button("SÌ, CANCELLA ORA"):
        st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
        st.session_state['last_alert'] = None
        st.rerun()

st.sidebar.markdown("---")

# --- 5. POPUP ALERT (RE-DESIGNED) ---
if st.session_state['last_alert']:
    play_notification_sound()
    alert = st.session_state['last_alert']
    main_color = "#00ffcc" if alert['Direzione'] == 'COMPRA' else "#ff4b4b"

    # Overlay Totale
    st.markdown(f"""
        <style>
            .full-screen-overlay {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0,0,0,0.92); z-index: 9990;
            }}
            .popup-card {{
                position: fixed; top: 40%; left: 50%; transform: translate(-50%, -50%);
                width: 85%; max-width: 450px; background: #111;
                border: 3px solid {main_color}; border-radius: 20px;
                padding: 30px; text-align: center; z-index: 9995;
                box-shadow: 0 0 50px {main_color}44;
            }}
            /* Container per il tasto chiudi proprio sotto il popup */
            .button-container {{
                position: fixed; top: 70%; left: 50%; transform: translateX(-50%);
                z-index: 10000; width: 300px;
            }}
        </style>
        <div class="full-screen-overlay"></div>
        <div class="popup-card">
            <h2 style="color: {main_color}; margin:0; letter-spacing:2px;">AI ALERT</h2>
            <h1 style="color: white; font-size: 3.5em; margin: 10px 0;">{alert['Asset']}</h1>
            <h2 style="color: white; background: {main_color}; border-radius: 10px; padding: 5px; color: black;">
                🚀 {alert['Direzione']}
            </h2>
            <div style="margin-top: 20px; display: flex; justify-content: space-around;">
                <div><small style="color:#888;">ENTRY</small><br><b style="font-size:1.2em;">{alert['Prezzo']}</b></div>
                <div><small style="color:#888;">TARGET</small><br><b style="color:{main_color}; font-size:1.2em;">{alert['TP']}</b></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Pulsante Chiudi Effettivo (Streamlit)
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("❌ CHIUDI E TORNA AL MONITOR", key="close_final_btn"):
        st.session_state['last_alert'] = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# --- 6. BODY PRINCIPALE ---
# Assicuriamoci che le variabili balance e risk_pc esistano già (create nella sidebar)
if 'signal_history' in st.session_state:
    run_sentinel()

# Banner logic
banner_path = "banner1.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)
else:
    st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1><p style="color: white; opacity: 0.8; margin:0;">Sentinel AI Engine • Forex & Crypto Analysis</p></div>', unsafe_allow_html=True)

st.info(f"🛰️ **Sentinel AI Engine Attiva**: Monitoraggio in corso su {len(asset_map)} asset (7 Forex e 2 Crypto) in tempo reale (30s).")
st.caption(f"Ultimo aggiornamento globale: {get_now_rome().strftime('%d/%m/%Y %H:%M:%S')}")

st.markdown("---")
#st.subheader("📈 Grafico in tempo reale (30s) con BB e RSI")
st.subheader(f"📈 Grafico {selected_label} (30s) con BB e RSI")

p_unit, price_fmt, p_mult, a_type = get_asset_params(pair)
df_rt = get_realtime_data(pair) 
df_d = yf.download(pair, period="1y", interval="1d", progress=False)

if df_rt is not None and not df_rt.empty and df_d is not None and not df_d.empty:
    
    # Pulizia dati
    if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
    df_d.columns = [c.lower() for c in df_d.columns]
    
    # Calcolo indicatori
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    df_rt['rsi'] = ta.rsi(df_rt['close'], length=14)
    df_d['rsi'] = ta.rsi(df_d['close'], length=14)
    df_d['atr'] = ta.atr(df_d['high'], df_d['low'], df_d['close'], length=14)
          
    c_up = [c for c in df_rt.columns if "BBU" in c.upper()][0]
    c_mid = [c for c in df_rt.columns if "BBM" in c.upper()][0]
    c_low = [c for c in df_rt.columns if "BBL" in c.upper()][0]
    
    curr_p = float(df_rt['close'].iloc[-1])
    curr_rsi = float(df_rt['rsi'].iloc[-1])
    rsi_val = float(df_d['rsi'].iloc[-1]) 
    last_atr = float(df_d['atr'].iloc[-1])
    
    score = 50 + (20 if curr_p < df_rt[c_low].iloc[-1] else -20 if curr_p > df_rt[c_up].iloc[-1] else 0)

    # --- COSTRUZIONE GRAFICO ---
    p_df = df_rt.tail(60)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])
    
    # Candele
    fig.add_trace(go.Candlestick(
        x=p_df.index, open=p_df['open'], high=p_df['high'], 
        low=p_df['low'], close=p_df['close'], name='Prezzo'
    ), row=1, col=1)
    
    # Bande Bollinger
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_up], line=dict(color='rgba(0, 191, 255, 0.6)', width=1), name='Upper BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_mid], line=dict(color='rgba(0, 0, 0, 0.3)', width=1), name='BBM'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_low], line=dict(color='rgba(0, 191, 255, 0.6)', width=1), fill='tonexty', fillcolor='rgba(0, 191, 255, 0.15)', name='Lower BB'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['rsi'], line=dict(color='#ffcc00', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00ff00", row=2, col=1)

    # --- AGGIUNTA GRIGLIA VERTICALE (OGNI 10 MINUTI) ---
    for t in p_df.index:
        if t.minute % 10 == 0:
            fig.add_vline(x=t, line_width=0.5, line_dash="solid", line_color="rgba(0, 0, 0, 0.3)", layer="below")

    # Layout Grafico
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    # 4. Metriche Base
    c_met1, c_met2 = st.columns(2)
    c_met1.metric(label=f"Prezzo {selected_label}", value=price_fmt.format(curr_p))
    c_met2.metric(label="RSI (5m)", value=f"{curr_rsi:.1f}", delta="Ipercomprato" if curr_rsi > 70 else "Ipervenduto" if curr_rsi < 30 else "Neutro", delta_color="inverse")
    
    st.caption(f"📢 RSI Daily: {rsi_val:.1f} | Divergenza: {detect_divergence(df_d)}")

    # --- VISUALIZZAZIONE METRICHE AVANZATE (ADX & AI) ---
    adx_df_ai = ta.adx(df_rt['high'], df_rt['low'], df_rt['close'], length=14)
    curr_adx_ai = adx_df_ai['ADX_14'].iloc[-1]

    st.markdown("---")
    st.subheader("🕵️ Sentinel Market Analysis")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("RSI Daily", f"{rsi_val:.1f}", detect_divergence(df_d))
    col_b.metric("Sentinel Score", f"{score}/100")
    adx_emoji = "🔴" if curr_adx_ai > 30 else "🟡" if curr_adx_ai > 20 else "🟢"
    col_c.metric("Forza Trend (ADX)", f"{curr_adx_ai:.1f}", adx_emoji)

    # --- TABELLA GUIDA ADX COLORATA ---
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

# --- 8. FOOTER & CRONOLOGIA ---
st.markdown("---")
st.subheader("📜 Cronologia Segnali")

if not st.session_state['signal_history'].empty:
    display_df = st.session_state['signal_history'].copy()
    
    def style_status(val):
        color = '#00ffcc' if '✅' in val else '#ff4b4b' if '❌' in val else '#ffcc00'
        return f'color: {color}; font-weight: bold'
    
    st.dataframe(
        display_df.style.map(style_status, subset=['Stato']), 
        use_container_width=True,
        hide_index=True
    )
    
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Esporta CSV", csv, "cronologia_forex.csv", "text/csv")
else:
    st.write("Nessun segnale rilevato finora. In attesa di opportunità...")

st.markdown("---")

# IN FONDO AL FILE
if __name__ == "__main__":
    # Esegui la scansione solo se la sidebar è già stata renderizzata
    if 'signal_history' in st.session_state:
        run_sentinel()

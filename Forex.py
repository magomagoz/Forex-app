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

# Refresh automatico ogni 60 secondi
st_autorefresh(interval=60 * 1000, key="sentinel_refresh")

# --- 2. FUNZIONI TECNICHE ---
def save_history_permanently():
    """Salva la cronologia attuale su un file fisico CSV"""
    try:
        if 'signal_history' in st.session_state and not st.session_state['signal_history'].empty:
            st.session_state['signal_history'].to_csv("permanent_signals_db.csv", index=False)
    except Exception as e:
        print(f"Errore salvataggio file: {e}")

def load_history_from_csv():
    if os.path.exists("permanent_signals_db.csv"):
        try:
            df = pd.read_csv("permanent_signals_db.csv")
            # Lista aggiornata con le nuove colonne monetarie e di protezione
            expected_cols = ['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 
                             'Stato', 'Investimento €', 'Risultato €', 'Stato_Prot', 'Protezione']
            for col in expected_cols:
                if col not in df.columns: 
                    df[col] = "0.00" if "€" in col else "Standard"
            return df
        except:
            return pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Stato', 'Investimento €', 'Risultato €', 'Stato_Prot', 'Protezione'])
    return pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Stato', 'Investimento €', 'Risultato €', 'Stato_Prot', 'Protezione'])

def send_telegram_msg(msg):
    token = "8235666467:AAGCsvEhlrzl7bH537bJTjsSwQ3P3PMRW10" 
    chat_id = "7191509088" 
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code != 200:
            st.toast(f"Errore Telegram: {r.status_code}", icon="⚠️")
    except Exception as e:
        print(f"Errore: {e}")

def get_now_rome():
    return datetime.now(rome_tz)

def play_notification_sound():
    audio_html = """
        <audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg"></audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def play_close_sound():
    # Un suono più breve e "cash register" per le chiusure
    audio_html = """
        <audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2017/2017-preview.mp3" type="audio/mpeg"></audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

def play_safe_sound():
    # Un suono tipo "scatto metallico" o "ding" per indicare la messa in sicurezza
    audio_html = """
        <audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2021/2021-preview.mp3" type="audio/mpeg"></audio>
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

@st.cache_data(ttl=60)
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
    if st.session_state['signal_history'].empty: return
    df = st.session_state['signal_history']
    
    # Parametri simulazione broker (Spread/Commissioni approx)
    COMMISSIONE_APPROX = 0.02 
    
    updates_made = False
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            ticker = asset_map[row['Asset']]
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): 
                    data.columns = data.columns.get_level_values(0)
                
                # Dati prezzi attuali
                current_high = float(data['High'].iloc[-1])
                current_low = float(data['Low'].iloc[-1])
                current_close = float(data['Close'].iloc[-1])
                
                # Valori convertiti
                entry_v = float(str(row['Prezzo']).replace(',', '.'))
                sl_v = float(str(row['SL']).replace(',', '.')) 
                tp_v = float(str(row['TP']).replace(',', '.'))
                investimento = float(str(row['Investimento €']).replace(',', '.'))
                
                new_status = None
                risultato_finale = 0.0

                # --- LOGICA TARGET DINAMICO (PROTEZIONE PROFITTO) ---
                dist_target = abs(tp_v - entry_v)
                progresso = abs(current_close - entry_v)
                
                # Se profitto > 25% del percorso e non abbiamo ancora protetto
                if progresso >= (dist_target * 0.25) and row.get('Stato_Prot') != 'Garantito':
                    if row['Direzione'] == 'COMPRA':
                        nuovo_sl = entry_v + (dist_target * 0.20)
                    else:
                        nuovo_sl = entry_v - (dist_target * 0.20)
                    
                    # Aggiorniamo lo Stop Loss nel DataFrame per bloccare il profitto
                    df.at[idx, 'SL'] = f"{nuovo_sl:.5f}" if "JPY" not in row['Asset'] else f"{nuovo_sl:.2f}"
                    df.at[idx, 'Stato_Prot'] = 'Garantito'
                    updates_made = True
                    play_safe_sound() 
                    send_telegram_msg(f"🛡️ **TARGET DINAMICO ATTIVATO**\n{row['Asset']}: Il profitto è ora blindato al 20%!")

                # --- CONTROLLO CHIUSURA (TP o SL aggiornato) ---
                if row['Direzione'] == 'COMPRA':
                    if current_high >= tp_v: 
                        new_status = '✅ TARGET'
                        risultato_finale = (investimento * 2.0) - COMMISSIONE_APPROX
                    elif current_low <= sl_v: 
                        # Se lo SL era "Garantito", calcoliamo il profitto del 20%
                        if row.get('Stato_Prot') == 'Garantito':
                            new_status = '🛡️ SL DINAMICO'
                            risultato_finale = (investimento * 0.40) - COMMISSIONE_APPROX # 20% del movimento = approx 40% dell'investimento se RR è 1:2
                        else:
                            new_status = '❌ STOP LOSS'
                            risultato_finale = -(investimento + COMMISSIONE_APPROX)
                        
                elif row['Direzione'] == 'VENDI':
                    if current_low <= tp_v: 
                        new_status = '✅ TARGET'
                        risultato_finale = (investimento * 2.0) - COMMISSIONE_APPROX
                    elif current_high >= sl_v: 
                        if row.get('Stato_Prot') == 'Garantito':
                            new_status = '🛡️ SL DINAMICO'
                            risultato_finale = (investimento * 0.40) - COMMISSIONE_APPROX
                        else:
                            new_status = '❌ STOP LOSS'
                            risultato_finale = -(investimento + COMMISSIONE_APPROX)
                
                if new_status:
                    df.at[idx, 'Stato'] = new_status
                    df.at[idx, 'Risultato €'] = f"{risultato_finale:+.2f}"
                    updates_made = True
                    play_close_sound()
                    
                    msg = f"🔔 **IQ OPTION - TRADE CHIUSO**\nAsset: {row['Asset']}\nEsito: {new_status}\nNetto: {risultato_finale:+.2f}€"
                    send_telegram_msg(msg)
                    
        except Exception as e:
            continue 
        
    if updates_made:
        st.session_state['signal_history'] = df
        save_history_permanently()

def run_sentinel():
    """Scansiona tutti gli asset definiti in asset_map"""
    # Usiamo i nomi corretti recuperati dallo stato o dai widget
    current_balance = st.session_state.get('balance_val', 1000)
    current_risk = st.session_state.get('risk_val', 1.0)
    
    assets = list(asset_map.items())
    for label, ticker in assets:
        try:
            # 1. DOWNLOAD DATI
            df_rt_s = yf.download(ticker, period="2d", interval="1m", progress=False)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df_rt_s.empty or df_d_s.empty: continue
            
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            if isinstance(df_d_s.columns, pd.MultiIndex): df_d_s.columns = df_d_s.columns.get_level_values(0)
            
            df_rt_s.columns = [c.lower() for c in df_rt_s.columns]
            df_d_s.columns = [c.lower() for c in df_d_s.columns]

            # 2. CALCOLO INDICATORI
            bb_s = ta.bbands(df_rt_s['close'], length=20, std=2)
            c_low = [c for c in bb_s.columns if "BBL" in c.upper()][0]
            c_up = [c for c in bb_s.columns if "BBU" in c.upper()][0]
            
            curr_v = float(df_rt_s['close'].iloc[-1])
            low_bb = float(bb_s[c_low].iloc[-1])
            up_bb = float(bb_s[c_up].iloc[-1])
            
            rsi_d = ta.rsi(df_d_s['close'], length=14).iloc[-1]

            # Sostituisci il calcolo ATR e ADX con i nomi minuscoli:
            atr_d = ta.atr(df_d_s['high'], df_d_s['low'], df_d_s['close'], length=14).iloc[-1]
            adx_df = ta.adx(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1]

            s_action = None
            if curr_v < low_bb and rsi_d < 60 and curr_adx < 45: 
                s_action = "COMPRA"
            elif curr_v > up_bb and rsi_d > 40 and curr_adx < 45: 
                s_action = "VENDI"

            if s_action:
                hist = st.session_state['signal_history']
                
                # 1. Controllo se c'è un trade APERTO (In Corso)
                is_running = not hist.empty and ((hist['Asset'] == label) & (hist['Stato'] == 'In Corso')).any()
                
                # 2. Controllo TEMPORALE (Nessun segnale duplicato negli ultimi 30 minuti)
                recent_signals = []
                if not hist.empty:
                    # Filtriamo i segnali dell'asset corrente
                    asset_hist = hist[hist['Asset'] == label]
                    if not asset_hist.empty:
                        # Convertiamo l'orario del segnale in oggetto datetime per il calcolo
                        last_sig_time_str = asset_hist.iloc[0]['DataOra']
                        last_sig_time = datetime.strptime(last_sig_time_str, "%H:%M:%S").time()
                        now_time = get_now_rome().time()
                        
                        # Calcolo differenza approssimativa in minuti
                        diff_min = (now_time.hour * 60 + now_time.minute) - (last_sig_time.hour * 60 + last_sig_time.minute)
                        
                        # Se il segnale è avvenuto meno di 30 minuti fa, lo ignoriamo
                        if 0 <= diff_min < 30:
                            s_action = None 

                if not is_running:
                    p_unit, p_fmt, p_mult, _ = get_asset_params(ticker)
                    
                    # 1. Calcolo Tecnico basato sulla volatilità (ATR)
                    atr_factor = 1.5
                    sl_dist_tecnica = atr_d * atr_factor
                    
                    if s_action == "COMPRA":
                        sl_proposto = curr_v - sl_dist_tecnica
                        tp = curr_v + (sl_dist_tecnica * 2) # Reward 1:2
                    else:
                        sl_proposto = curr_v + sl_dist_tecnica
                        tp = curr_v - (sl_dist_tecnica * 2)

                    # 2. APPLICAZIONE REGOLA PROTEZIONE 50% (Hard Limit)
                    # Il rischio massimo per singola operazione è risk_val. 
                    # Se lo SL tecnico supera il 50% di risk_val, lo accorciamo.
                    risk_val = current_balance * (current_risk / 100)
                    max_loss_allowed = risk_val * 0.50 # Limite 50% dell'investito
                    
                    # Calcoliamo quanto dista lo SL in termini monetari
                    distanza_punti = abs(curr_v - sl_proposto)
                    perdita_monetaria_tecnica = distanza_punti * p_mult * 10 # Approx valore pip/punto
                    
                    is_protected = False # Inizialmente falso
                    if perdita_monetaria_tecnica > max_loss_allowed:
                        distanza_massima_punti = max_loss_allowed / (p_mult * 10)
                        sl = curr_v - distanza_massima_punti if s_action == "COMPRA" else curr_v + distanza_massima_punti
                        is_protected = True # La protezione è entrata in funzione
                    else:
                        sl = sl_proposto
                
                    new_sig = {
                        'DataOra': get_now_rome().strftime("%H:%M:%S"),
                        'Asset': label, 
                        'Direzione': s_action, 
                        'Prezzo': p_fmt.format(curr_v), 
                        'TP': p_fmt.format(tp), 
                        'SL': p_fmt.format(sl), 
                        'Protezione': "ATTIVA (50%)" if is_protected else "Standard",
                        'Stato_Prot': 'In Attesa',
                        'Stato': 'In Corso',
                        'Investimento €': f"{risk_val:.2f}",
                        'Risultato €': "0.00"  # Inizialmente neutro
                    }
                    
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    save_history_permanently()
                    st.session_state['last_alert'] = new_sig

                    # FIX: Chiusura corretta della stringa e della parentesi
                    telegram_text = (f"🚀 *{s_action}* {label}\n"
                                     f"Entry: {new_sig['Prezzo']}\n"
                                     f"TP: {new_sig['TP']}\n"
                                     f"SL: {new_sig['SL']}\n"
                                     f"Investimento: {new_sig['Investimento €']}€")
                    
                    send_telegram_msg(telegram_text)

            st.session_state['last_scan_status'] = f"🟢 {get_now_rome().strftime('%H:%M:%S')} - {label}: OK"
            time_lib.sleep(0.5) 

        except Exception as e:
            st.session_state['last_scan_status'] = f"🔴 Errore {label}: {str(e)}"
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

# --- INIZIALIZZAZIONE STATO (Session State) ---
if 'signal_history' not in st.session_state: 
    st.session_state['signal_history'] = load_history_from_csv()
if 'sentinel_logs' not in st.session_state:
    st.session_state['sentinel_logs'] = []
if 'last_alert' not in st.session_state:
    st.session_state['last_alert'] = None
if 'last_scan_status' not in st.session_state:
    st.session_state['last_scan_status'] = "In attesa..."

# --- 3. ESECUZIONE AGGIORNAMENTO DATI (PRIMA DELLA GUI) ---
# Importante: Aggiorniamo i risultati TP/SL prima di disegnare la sidebar
update_signal_outcomes()

def get_equity_data():
    """Calcola l'andamento del saldo applicando il rischio scelto ai trade chiusi"""
    initial_balance = balance 
    equity_curve = [initial_balance]
    
    if st.session_state['signal_history'].empty:
        return pd.Series(equity_curve)
    
    # Ordiniamo dal più vecchio al più recente per la curva temporale
    df_sorted = st.session_state['signal_history'].iloc[::-1]
    current_bal = initial_balance
    
    for _, row in df_sorted.iterrows():
        # Applichiamo il rischio scelto sulla barra al saldo attuale
        risk_amount = current_bal * (risk_pc / 100)
        
        if row['Stato'] == '✅ TARGET':
            # Simuliamo un profitto con Reward Ratio 1:2
            current_bal += (risk_amount * 2) 
        elif row['Stato'] == '❌ STOP LOSS':
            # Perdita fissa della quota rischio
            current_bal -= risk_amount
            
        equity_curve.append(current_bal)
        
    return pd.Series(equity_curve)

# --- 4. ESECUZIONE SENTINEL ---
# Assicuriamoci che lo scanner giri solo se lo stato è inizializzato
if 'signal_history' in st.session_state:
    run_sentinel()

# --- 5. SIDEBAR ---
st.sidebar.header("🛠 Trading Desk (1m)")

# Countdown Testuale e Barra Rossa Animata
st.sidebar.markdown("⏳ **Prossimo Scan**")

# CSS per la barra che si riempie in 60 secondi
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
            animation: progressFill 60s linear infinite;
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
balance = st.sidebar.number_input("**Conto (€)**", value=1000, key="balance_val")
risk_pc = st.sidebar.slider("**Investimento %**", 0.5, 5.0, 2.0, step=0.5, key="risk_val")

# --- Sotto il widget risk_pc ---
st.sidebar.markdown(
    """
    <div style='background-color: rgba(255, 152, 0, 0.1); 
                border: 1px solid #ff9800; 
                padding: 10px; 
                border-radius: 5px; 
                margin-top: 10px;'>
        <span style='color: #ff9800; font-weight: bold; font-size: 0.85em;'>
            🟠 IQOption Mode: ATTIVA
        </span><br>
        <small style='color: #888; font-size: 0.75em;'>
            Commissioni e Spread simulati inclusi.
        </small>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- CALCOLO INVESTIMENTO SIMULATO ---
investimento_simulato = balance * (risk_pc / 100)
saldo_residuo = balance - investimento_simulato

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Gestione Capitale")
#col_cap1, col_cap2 = st.sidebar.columns(2)
#col_cap1.metric("Conto", f"€ {balance:.2f}")
#col_cap2.metric("Investimento", f"€ {investimento_simulato:.2f}")

st.sidebar.metric("Conto iniziale", f"€ {balance:.2f}")
st.sidebar.metric("Investimento per operazione", f"€ {investimento_simulato:.2f}")


#st.sidebar.info(f"💳 **Saldo Attuale Operativo**: € {saldo_residuo:.2f}")

st.sidebar.markdown("---")

# --- SIDEBAR PERFORMANCE ---
st.sidebar.subheader("🏆 Performance")

equity_series = get_equity_data()
current_equity = equity_series.iloc[-1]
initial_bal = balance if balance > 0 else 1000
total_return = ((current_equity - initial_bal) / initial_bal) * 100

# Calcolo Drawdown
max_val = equity_series.max()
dd = ((current_equity - max_val) / max_val) * 100 if max_val > 0 else 0

# Visualizzazione Metriche
st.sidebar.metric("Saldo Attuale Operativo", f"€ {current_equity:.2f}", delta=f"{total_return}%")
st.sidebar.metric("Drawdown Massimo", f"{dd:.2f}%", delta_color="inverse")

# Grafico Equity (Piccolo e pulito)
#fig_equity = go.Figure()
#fig_equity.add_trace(go.Scatter(y=equity_series, mode='lines', fill='tozeroy', line=dict(color='#00ffcc')))
#fig_equity.update_layout(height=100, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
#st.sidebar.plotly_chart(fig_equity, use_container_width=True, config={'displayModeBar': False})

# Dettagli operazione selezionata (se presente)
active_trades = st.session_state['signal_history'][st.session_state['signal_history']['Stato'] == 'In Corso']
if not active_trades.empty:
    st.sidebar.warning("⚡ Ultima Operazione Attiva")
    last_t = active_trades.iloc[0]
    st.sidebar.write(f"Asset: **{last_t['Asset']}**")
    st.sidebar.write(f"SL: `{last_t['SL']}` | TP: `{last_t['TP']}`")

st.sidebar.markdown("---")
# ... (restante codice sidebar: sessioni, win rate, reset)
st.sidebar.subheader("🌍 Sessioni di Mercato")
for s_name, is_open in get_session_status().items():
    color = "🟢" if is_open else "🔴"
    status_text = "APERTO" if is_open else "CHIUSO"
    st.sidebar.markdown(f"**{s_name}** <small>: {status_text}</small> {color}",
unsafe_allow_html=True)
   
# Reset Sidebar
st.sidebar.markdown("---")
with st.sidebar.popover("🗑️ **Reset Cronologia**"):
    st.warning("Sei sicuro? Questa azione cancellerà tutti i segnali salvati.")

    if st.button("SÌ, CANCELLA ORA"):
        st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
        save_history_permanently() # Questo sovrascrive il file CSV con uno vuoto
        st.rerun()

st.sidebar.markdown("---")

if st.sidebar.button("TEST ALERT"):
    st.session_state['last_alert'] = {
        'Asset': 'TEST/EUR', 'Direzione': 'COMPRA', 'Prezzo': '1.0000',
        'TP': '1.0100', 'SL': '0.9900', 'Protezione': 'Standard'
    }
    if 'alert_start_time' in st.session_state: del st.session_state['alert_start_time']
    st.rerun()

st.sidebar.markdown("---")

# --- 5. POPUP ALERT (VERSIONE FINALE COMPATIBILE) ---
if st.session_state.get('last_alert'):
    if 'alert_start_time' not in st.session_state:
        st.session_state['alert_start_time'] = time_lib.time()
        play_notification_sound()

    elapsed = time_lib.time() - st.session_state['alert_start_time']
    countdown = max(0, int(30 - elapsed))
    
    if elapsed > 30:
        st.session_state['last_alert'] = None
        if 'alert_start_time' in st.session_state: del st.session_state['alert_start_time']
        st.rerun()

    if st.session_state.get('last_alert'):
        alert = st.session_state['last_alert']
        main_color = "#00ffcc" if alert['Direzione'] == 'COMPRA' else "#ff4b4b"

        # Creiamo un overlay che non blocca i click di sistema ma mostra il messaggio
        st.markdown(f"""
            <div style="position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.9); z-index:1000; backdrop-filter:blur(10px); display:flex; justify-content:center; align-items:center;">
                <div style="width:90%; max-width:400px; background:#111; border:3px solid {main_color}; border-radius:25px; padding:30px; text-align:center; box-shadow:0 0 50px {main_color}44;">
                    <div style="color:{main_color}; font-weight:bold; letter-spacing:3px; font-size:0.9em;">AI SENTINEL</div>
                    <div style="font-size:3.5em; font-weight:900; color:white; margin:10px 0; line-height:1;">{alert['Asset']}</div>
                    <div style="background:{main_color}; color:black; padding:10px 20px; border-radius:10px; font-weight:900; font-size:1.8em; margin:15px 0; display:inline-block;">{alert['Direzione']}</div>
                    
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:15px; margin:20px 0; text-align:left; border-top:1px solid #333; padding-top:20px; color:white;">
                        <div><small style="color:#888;">PRICE</small><br><b>{alert['Prezzo']}</b></div>
                        <div style="text-align:right;"><small style="color:#888;">TARGET</small><br><b style="color:{main_color};">{alert['TP']}</b></div>
                        <div><small style="color:#888;">STOP LOSS</small><br><b style="color:#ff4b4b;">{alert['SL']}</b></div>
                        <div style="text-align:right;"><small style="color:#888;">CLOSE</small><br><b>{countdown}s</b></div>
                    </div>
                    
                    <a href="/" target="_self" style="text-decoration:none;">
                        <div style="background:#333; color:white; padding:15px; border-radius:12px; font-weight:bold; cursor:pointer; border:1px solid #555;">
                            CHIUDI E TORNA AL GRAFICO
                        </div>
                    </a>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Pulsante di "sicurezza" standard di Streamlit nel caso il link sopra fallisse
        if st.button("CHIUDI MANUALE (BACKUP)", key="close_manual"):
            st.session_state['last_alert'] = None
            if 'alert_start_time' in st.session_state: del st.session_state['alert_start_time']
            st.rerun()

# --- 7. BODY PRINCIPALE ---
# Banner logic
banner_path = "banner1.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)
else:
    st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1><p style="color: white; opacity: 0.8; margin:0;">Sentinel AI Engine • Forex & Crypto Analysis</p></div>', unsafe_allow_html=True)

st.info(f"🛰️ **Sentinel AI Attiva**: Monitoraggio in corso su {len(asset_map)} asset (7 Forex e 2 Crypto) in tempo reale (1m).")
st.caption(f"Ultimo aggiornamento globale: {get_now_rome().strftime('%d/%m/%Y %H:%M:%S')}")

st.markdown("---")
#st.subheader("📈 Grafico in tempo reale")
st.subheader(f"📈 Grafico {selected_label} (1m) con BB e RSI")

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

    # --- TABELLA GUIDA ADX COLORATA (FULL WIDTH) ---
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

    # 1. Applichiamo lo stile e nascondiamo l'indice
    # 2. Aggiungiamo 'set_table_attributes' per forzare la larghezza al 100%
    styled_adx_html = (adx_guide.style
                       .apply(highlight_adx, axis=1)
                       .hide(axis='index')
                       .set_table_attributes('style="width:100%; border-collapse: collapse; text-align: left;"')
                       .to_html())

    # Visualizziamo con unsafe_allow_html
    st.markdown(styled_adx_html, unsafe_allow_html=True)

# --- 8. CURRENCY STRENGTH ---
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

# --- 9. CRONOLOGIA SEGNALI (CORREZIONE ERRORI) ---
st.markdown("---")
st.subheader("📜 Cronologia Segnali")

# Inizializziamo display_df sempre, anche se vuoto, per evitare NameError
display_df = pd.DataFrame()

if not st.session_state['signal_history'].empty:
    display_df = st.session_state['signal_history'].copy()
    
    # Invertiamo l'ordine per vedere i più recenti in alto
    display_df = display_df.iloc[::-1]

    try:
        # Usiamo map invece di applymap (nuova sintassi Pandas)
        st.dataframe(
            display_df.style.map(style_status, subset=['Stato']),
            use_container_width=True,
            hide_index=True,
            column_order=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'TP', 'SL', 'Stato', 'Risultato €']
        )
        
        # Bottone download - ora display_df esiste sicuramente
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Esporta CSV", csv_data, "trading_history.csv", "text/csv")
        
    except Exception as e:
        # Se lo stile fallisce, mostra la tabella semplice invece di crashare
        st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("Nessun segnale registrato.")

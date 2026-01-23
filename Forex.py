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

# --- COSTANTI DI MERCATO ---
SIMULATED_SPREAD = 0.0005  # Rappresenta lo 0,05%

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
    cols = ['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Stato', 'Investimento €', 'Risultato €', 'Costo Spread €', 'Stato_Prot', 'Protezione']
    if os.path.exists("permanent_signals_db.csv"):
        try:
            df = pd.read_csv("permanent_signals_db.csv")
            # Forza la presenza di tutte le colonne necessarie
            for col in cols:
                if col not in df.columns: df[col] = "0.00"
            return df
        except:
            return pd.DataFrame(columns=cols)
    # Se il file non esiste, crea un DataFrame vuoto con le colonne giuste
    return pd.DataFrame(columns=cols)

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

def style_status(val):
    if val == '✅ TARGET': return 'background-color: rgba(0, 255, 204, 0.2); color: #00ffcc;'
    if val == '❌ STOP LOSS': return 'background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b;'
    if val == '🛡️ SL DINAMICO': return 'background-color: rgba(255, 165, 0, 0.2); color: #ffa500;'
    
    # Se il valore è numerico e positivo/negativo (per la colonna Risultato)
    try:
        num = float(str(val).replace('+', ''))
        if num > 0: return 'color: #00ffcc; font-weight: bold;'
        if num < 0: return 'color: #ff4b4b; font-weight: bold;'
    except:
        pass
    return ''

def get_trailing_params(asset_name):
    """
    Ritorna: (Step1_BE, Step2_Save, SL_Iniziale_Percent)
    Forex: Scaglioni stretti per micro-movimenti
    Crypto: Scaglioni larghi per alta volatilità
    """
    if any(x in asset_name for x in ["BTC", "ETH"]):
        # Parametri Crypto
        return 5.0, 10.0, -10.0  # +5% -> BE, +10% -> +5%, Inizio -10%
    else:
        # Parametri Forex
        return 0.5, 1.0, -2.0    # +0.5% -> BE, +1.0% -> +0.5%, Inizio -2% (Forex a -10% è troppo lontano)

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
    """
    Restituisce: (unità_minima, formato_prezzo, moltiplicatore_reale, tipo)
    """
    if "BTC" in pair or "ETH" in pair:
        # Per Crypto: 1 punto = 1 Dollaro
        return 1.0, "{:.2f}", 1, "CRYPTO"
    elif "JPY" in pair:
        # Per JPY: 0.01 = 1 punto
        return 0.01, "{:.3f}", 100, "FOREX_JPY"
    else:
        # Per Forex standard (EURUSD ecc): 0.0001 = 1 punto (PIP)
        return 0.0001, "{:.5f}", 10000, "FOREX_STD"

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi_col = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi_col.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi_col.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DECRESCITA"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 CRESCITA"
    return "Neutrale"
    
# --- 2. FUNZIONI TECNICHE (AGGIORNATE) ---
def update_signal_outcomes():
    if st.session_state['signal_history'].empty: return
    df = st.session_state['signal_history']
    updates_made = False
    
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            ticker = asset_map[row['Asset']]
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            if data.empty: continue
            
            if isinstance(data.columns, pd.MultiIndex): 
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.lower() for c in data.columns]
            
            # --- RECUPERO DATI OPERAZIONE ---
            current_price = float(data['close'].iloc[-1])
            entry_v = float(str(row['Prezzo']).replace(',', '.'))
            investimento = float(str(row['Investimento €']).replace(',', '.'))
            direzione = row['Direzione']
            tp_v = float(str(row['TP']).replace(',', '.'))
            current_sl = float(str(row['SL']).replace(',', '.'))
            
            # --- CALCOLO GAIN (Basato su prezzo già "spread-ato") ---
            if direzione == 'COMPRA':
                percent_gain = ((current_price - entry_v) / entry_v) * 100
            else:
                percent_gain = ((entry_v - current_price) / entry_v) * 100

            new_sl = current_sl
            status_prot = row.get('Stato_Prot', 'Iniziale')

            # --- LOGICA TRAILING STOP ---
            if percent_gain >= 5.0 and 'Iniziale' in status_prot:
                new_sl = entry_v
                status_prot = 'BE (0%)'
                play_safe_sound()
                send_telegram_msg(f"🛡️ {row['Asset']}: SL a Pareggio!")
            elif percent_gain >= 10.0 and 'BE' in status_prot:
                new_sl = entry_v * 1.05 if direzione == 'COMPRA' else entry_v * 0.95
                status_prot = 'Safe (+5%)'
                play_safe_sound()

            # Aggiornamento fisico SL se cambiato
            if new_sl != current_sl:
                _, p_fmt, _, _ = get_asset_params(row['Asset'])
                df.at[idx, 'SL'] = p_fmt.format(new_sl)
                df.at[idx, 'Stato_Prot'] = status_prot
                updates_made = True

            # --- CONTROLLO CHIUSURA ---
            target_hit = (direzione == 'COMPRA' and current_price >= tp_v) or (direzione == 'VENDI' and current_price <= tp_v)
            stop_hit = (direzione == 'COMPRA' and current_price <= new_sl) or (direzione == 'VENDI' and current_price >= new_sl)

            if target_hit or stop_hit:
                esito = '✅ TARGET' if target_hit else ('🛡️ SL DINAMICO' if 'Iniziale' not in status_prot else '❌ STOP LOSS')
                df.at[idx, 'Stato'] = esito
                final_profit = investimento * (percent_gain / 100)
                df.at[idx, 'Risultato €'] = f"{final_profit:+.2f}"
                updates_made = True
                play_close_sound()
                send_telegram_msg(f"🏁 CHIUSO: {row['Asset']}\nEsito: {esito}\nNetto: {final_profit:+.2f}€")
                    
        except Exception as e:
            continue 
        
    if updates_made:
        st.session_state['signal_history'] = df
        save_history_permanently()

def run_sentinel():
    # --- AUTO-PULIZIA CRONOLOGIA (Gestione Memoria) ---
    if not st.session_state['signal_history'].empty:
        # Se superiamo i 50 record, teniamo solo i 50 più recenti
        if len(st.session_state['signal_history']) > 50:
            st.session_state['signal_history'] = st.session_state['signal_history'].head(50)
            save_history_permanently()

    # Azzera i popup aperti all'inizio di ogni nuovo ciclo di scansione
    st.session_state['last_alert'] = None
    if 'alert_notified' in st.session_state: 
        del st.session_state['alert_notified']
    
    #"""Scansiona tutti gli asset e popola il Debug Monitor"""
    current_balance = st.session_state.get('balance_val', 1000)
    current_risk = st.session_state.get('risk_val', 1.0)
    
    # Lista per il monitoraggio live nella sidebar
    debug_list = []
    
    assets = list(asset_map.items())
    for label, ticker in assets:
        try:
            # 1. SCARICO DATI (Maggiore tolleranza errori)
            df_rt_s = yf.download(ticker, period="2d", interval="1m", progress=False)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df_rt_s.empty or df_d_s.empty: 
                debug_list.append(f"🔴 {label}: No Data")
                continue
            
            # Pulizia Colonne ROBUSTA
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            if isinstance(df_d_s.columns, pd.MultiIndex): df_d_s.columns = df_d_s.columns.get_level_values(0)
            
            # Rinominiamo esplicitamente per pandas_ta
            df_rt_s.columns = [c.lower() for c in df_rt_s.columns]
            df_d_s.columns = [c.lower() for c in df_d_s.columns]

            # 2. CALCOLO INDICATORI
            bb_s = ta.bbands(df_rt_s['close'], length=20, std=2)
            if bb_s is None: continue # Skip se errore calcolo

            c_low = [c for c in bb_s.columns if "BBL" in c.upper()][0]
            c_up = [c for c in bb_s.columns if "BBU" in c.upper()][0]
            
            curr_v = float(df_rt_s['close'].iloc[-1])
            low_bb = float(bb_s[c_low].iloc[-1])
            up_bb = float(bb_s[c_up].iloc[-1])
            
            rsi_d = ta.rsi(df_d_s['close'], length=14).iloc[-1]
            
            adx_df = ta.adx(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1] if adx_df is not None else 0

            # 3. CONDIZIONI DI INGRESSO (Mean Reversion)
            s_action = None
            
            # Debug Status
            dist_low = curr_v - low_bb
            dist_up = up_bb - curr_v
            
            # Logica: Prezzo SOTTO banda bassa o SOPRA banda alta
            if curr_v < low_bb and rsi_d < 60 and curr_adx < 45: 
                s_action = "COMPRA"
            elif curr_v > up_bb and rsi_d > 40 and curr_adx < 45: 
                s_action = "VENDI"

            # Aggiungiamo info al monitor debug
            icon = "🟢" if s_action else "⚪"
            debug_info = f"{label}: {curr_v:.4f} | BB: {low_bb:.4f}/{up_bb:.4f}"
            if s_action: debug_info += f" -> 🔥 {s_action}"
            debug_list.append(f"{icon} {debug_info}")

            if s_action:
                hist = st.session_state['signal_history']
                # Controllo Duplicati / Trade in corso
                is_running = not hist.empty and ((hist['Asset'] == label) & (hist['Stato'] == 'In Corso')).any()
                
                # Controllo Tempo (30 min)
                recent_signals = False
                if not hist.empty:
                    asset_hist = hist[hist['Asset'] == label]
                    if not asset_hist.empty:
                        last_sig = asset_hist.iloc[0]['DataOra']
                        # Semplice check temporale stringa se stesso giorno
                        if last_sig > (get_now_rome().replace(minute=get_now_rome().minute - 30)).strftime("%H:%M:%S"):
                           recent_signals = True
              
                if not is_running and not recent_signals:
                    p_unit, p_fmt, p_mult, a_type = get_asset_params(label)
                    investimento_puntata = current_balance * (current_risk / 100)
                
                    # APPLICAZIONE SPREAD 0,05%
                    # Se COMPRI, paghi di più. Se VENDI, incassi di meno.
                    if s_action == "COMPRA":
                        entry_with_spread = curr_v * (1 + SIMULATED_SPREAD)
                    else:
                        entry_with_spread = curr_v * (1 - SIMULATED_SPREAD)
                
                    # Calcolo SL e TP basati sul prezzo penalizzato dallo spread
                    percentuale_perdita_max = 0.10 
                    distanza_prezzo_sl = entry_with_spread * (percentuale_perdita_max / 10)
                
                    if s_action == "COMPRA":
                        sl_prezzo = entry_with_spread - distanza_prezzo_sl
                        tp_prezzo = entry_with_spread * 1.005 
                    else:
                        sl_prezzo = entry_with_spread + distanza_prezzo_sl
                        tp_prezzo = entry_with_spread * 0.995
        
                    # Calcolo del costo dello spread in Euro basato sulla puntata
                    costo_spread_euro = investimento_puntata * SIMULATED_SPREAD
                    
                    new_sig = {
                        'DataOra': get_now_rome().strftime("%H:%M:%S"),
                        'Asset': label, 
                        'Direzione': s_action, 
                        'Prezzo': p_fmt.format(entry_with_spread), 
                        'TP': p_fmt.format(tp_prezzo), 
                        'SL': p_fmt.format(sl_prezzo), 
                        'Stato': 'In Corso',
                        'Protezione': 'Trailing Step',
                        'Investimento €': f"{investimento_puntata:.2f}",
                        'Risultato €': "0.00",
                        'Costo Spread €': f"{costo_spread_euro:.3f}", # Salviamo il costo spread
                        'Stato_Prot': 'Iniziale'
                    }

                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    
                    st.session_state['last_alert'] = new_sig
                    save_history_permanently()
  
                    telegram_text = (f"🚀 *{s_action}* {label}\n"
                                     f"Entry: {new_sig['Prezzo']}\nTP: {new_sig['TP']}\nSL: {new_sig['SL']}")
                    send_telegram_msg(telegram_text)

            st.session_state['last_scan_status'] = f"✅ Scan OK: {get_now_rome().strftime('%H:%M:%S')}"

        except Exception as e:
            debug_list.append(f"❌ {label} Err: {str(e)}")
            continue
    
    # Salviamo il log per visualizzarlo in sidebar
    st.session_state['sentinel_logs'] = debug_list
                    
# Sostituisci la tua funzione get_win_rate con questa:
def display_performance_stats():
    if st.session_state['signal_history'].empty:
        return
    
    df = st.session_state['signal_history']
    conclusi = df[df['Stato'].str.contains('TARGET|STOP|DINAMICO', na=False)]
    
    if not conclusi.empty:
        vittorie = len(conclusi[conclusi['Stato'] == '✅ TARGET'])
        wr = (vittorie / len(conclusi)) * 100
        st.sidebar.write(f"📊 **Win Rate**: {wr:.1f}% ({vittorie}/{len(conclusi)})")

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
            height: 12px; margin-bottom: 25px; border: 1px solid #555; overflow: hidden;
        }
        .red-bar {
            height: 100%; background-color: #ff4b4b; width: 0%;
            animation: progressFill 60s linear infinite;
            box-shadow: 0 0 10px #ff4b4b;
        }
    </style>
    <div class="container-bar"><div class="red-bar"></div></div>
""", unsafe_allow_html=True)

with st.sidebar.expander("🔍 Live Sentinel Data", expanded=True):
    if 'sentinel_logs' in st.session_state and st.session_state['sentinel_logs']:
        for log in st.session_state['sentinel_logs']:
            st.caption(log)
    else:
        st.caption("In attesa del primo scan...")

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
#st.sidebar.metric("Drawdown Massimo", f"{dd:.2f}%", delta_color="inverse")

# --- LOGICA COLORE DRAWDOWN ---
# Verde se tra 0 e 10%, Rosso se oltre 20%, Grigio/Default tra 10 e 20
dd_color = "normal" 
if 0 <= abs(dd) <= 10:
    dd_color = "normal" # Streamlit usa il verde di default per i delta positivi/normali
elif abs(dd) > 20:
    dd_color = "inverse" # Lo rende rosso se considerato come valore negativo

st.sidebar.metric(
    "Drawdown Massimo", 
    f"{dd:.2f}%", 
    delta="OTTIMO" if abs(dd) <= 10 else "ATTENZIONE" if abs(dd) > 20 else "",
    delta_color=dd_color
)

display_performance_stats()

# --- MONITORAGGIO TRADE ATTIVI NELLA SIDEBAR (OTTIMIZZATO) ---
active_trades = st.session_state['signal_history'][st.session_state['signal_history']['Stato'] == 'In Corso']

#if not active_trades.empty:
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Monitor Real-Time")
    
for _, trade in active_trades.iterrows():
    try:
        t_ticker = asset_map.get(trade['Asset'], trade['Asset'])
        t_data = yf.download(t_ticker, period="1d", interval="1m", progress=False, timeout=5)
        
        if not t_data.empty:
            curr_p = float(t_data['Close'].iloc[-1])
            entry_p = float(str(trade['Prezzo']).replace(',', '.'))
            inv = float(str(trade['Investimento €']).replace(',', '.'))
            
            # Calcolo profitto/perdita
            p_diff = ((curr_p - entry_p) / entry_p) if trade['Direzione'] == 'COMPRA' else ((entry_p - curr_p) / entry_p)
            latente_perc = p_diff * 100
            
            # Parametri Barra (Limitiamo visivamente tra -1 e 1 per la scala)
            # Trasformiamo la perc in un valore da 0 a 100 per il CSS
            # -1% -> 0%, 0% -> 50%, +1% -> 100%
            pos_barra = max(0, min(100, (latente_perc + 1) * 50))
            color = "#006400" if latente_perc >= 0 else "#ff4b4b"
            
            st.sidebar.markdown(f"""
                <div style="margin-bottom: 15px; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 5px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85em;">
                        <b>{trade['Asset']}</b>
                        <span style="color:{color}; font-weight:bold;">{latente_perc:+.2f}%</span>
                    </div>
                    <div style="width: 100%; background: #333; height: 6px; border-radius: 3px; margin-top: 5px; position: relative;">
                        <div style="position: absolute; left: 50%; width: 2px; height: 8px; background: white; top: -1px; z-index: 1;"></div>
                        <div style="width: {pos_barra}%; background: {color}; height: 100%; border-radius: 3px; transition: width 0.5s;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except:
        continue

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

# --- TASTO ESPORTAZIONE DATI ---
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Backup Report")

if not st.session_state['signal_history'].empty:
    csv_data = st.session_state['signal_history'].to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 SCARICA CRONOLOGIA CSV",
        data=csv_data,
        file_name=f"Trading_Report_{get_now_rome().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.sidebar.info("Nessun dato da esportare")

#if st.sidebar.button("TEST ALERT"):
    #st.session_state['last_alert'] = {'Asset': 'TEST/EUR', 'Direzione': 'COMPRA', 'Prezzo': '1.0000', 'TP': '1.0100', 'SL': '0.9900', 'Protezione': 'Standard'}
    #if 'alert_start_time' in st.session_state: del st.session_state['alert_start_time']
    #st.rerun()

#st.sidebar.markdown("---")

# --- 6. POPUP ALERT (SINCRONIZZATO CON REFRESH 60s) ---
if st.session_state.get('last_alert'):
    # Suona solo la prima volta che appare l'alert
    if 'alert_notified' not in st.session_state:
        play_notification_sound()
        st.session_state['alert_notified'] = True

    alert = st.session_state['last_alert']
    hex_color = "#00ffcc" if alert['Direzione'] == 'COMPRA' else "#ff4b4b"

    # Box Alert Grafico
    st.markdown(f"""
        <div style="background-color: #000; border: 3px solid {hex_color}; padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center; box-shadow: 0 0 20px {hex_color}44;">
            <h2 style="color: white; margin: 0;">🚀 NUOVO SEGNALE RILEVATO: {alert['Asset']}</h2>
            <h1 style="color: {hex_color}; margin: 5px 0;">{alert['Direzione']} @ {alert['Prezzo']}</h1>
            <p style="color: #888; margin: 0;">TP: {alert['TP']} | SL: {alert['SL']}</p>
            <div style="margin-top: 10px; font-size: 0.8em; color: #555;">
                Questo alert scomparirà automaticamente al prossimo aggiornamento della sentinella.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Tasto per chiusura manuale immediata
    if st.button("✅ CHIUDI ALERT ORA", use_container_width=True):
        st.session_state['last_alert'] = None
        if 'alert_notified' in st.session_state: del st.session_state['alert_notified']
        st.rerun()
    
    st.divider()

# --- LOGICA DI PULIZIA AUTOMATICA ---
# Questa parte assicura che al prossimo giro di 'run_sentinel', l'alert venga rimosso
#if 'last_alert' in st.session_state and st.session_state['last_alert'] is not None:
        # Opzionale: puoi decidere di resettarlo qui o lasciarlo resettare alla fine dello script
        # Per la tua richiesta, lo resettiamo all'inizio di ogni scan in run_sentinel()

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

# --- 9. CRONOLOGIA SEGNALI (CORRETTA) ---
st.markdown("---")
st.subheader("📜 Cronologia Segnali")

if not st.session_state['signal_history'].empty:
    full_history = st.session_state['signal_history'].copy()
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        opzioni_stato = sorted(full_history['Stato'].unique().tolist())
        filtro_stato = st.multiselect("Filtra Esito:", options=opzioni_stato, default=[], placeholder="Tutti gli esiti")
    with col_f2:
        opzioni_asset = sorted(full_history['Asset'].unique().tolist())
        filtro_asset = st.multiselect("Filtra Valuta:", options=opzioni_asset, default=[], placeholder="Tutte le valute")

    # Applichiamo i filtri
    df_filtrato = full_history.copy()
    if filtro_stato:
        df_filtrato = df_filtrato[df_filtrato['Stato'].isin(filtro_stato)]
    if filtro_asset:
        df_filtrato = df_filtrato[df_filtrato['Asset'].isin(filtro_asset)]
    
    # ORDINE: Poiché usiamo concat(new, old), l'indice 0 è già il più recente.
    # Non serve .iloc[::-1]. Se vuoi essere sicuro al 100%, usa:
    display_df = df_filtrato.reset_index(drop=True)

    
    if not display_df.empty:
        st.dataframe(
            display_df.style.map(style_status, subset=['Stato', 'Risultato €']), # Aggiunto Risultato €
            use_container_width=True,
            hide_index=True,
            column_order=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'TP', 'SL', 'Stato', 'Investimento €', 'Risultato €', 'Costo Spread €', 'Stato_Prot']
        )

    else:
        st.warning("Nessun dato corrispondente ai filtri selezionati.")

# 4. SE LA CRONOLOGIA È VUOTA
else:
    st.info(f"📖 **In attesa di un segnale da registrare**")

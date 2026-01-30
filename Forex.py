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
asset_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDCHF": "USDCHF=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X",
            "EURGBP": "EURGBP=X", "GBPJPY": "GBPJPY=X", "EURJPY": "EURJPY=X"}

# Refresh automatico ogni 60 secondi
st_autorefresh(interval=60 * 1000, key="sentinel_refresh")

# Recupero dai Secrets
TELE_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELE_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]
   
# --- 2. FUNZIONI TECNICHE CORRETTE & OTTIMIZZATE ---
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

def get_now_rome():
    return datetime.now(rome_tz)

def style_protection(val):
    # Se il capitale è blindato in profitto, usiamo un verde brillante
    if 'Blindato' in str(val) or 'Garantito' in str(val):
        return 'background-color: #2ecc71; color: white; font-weight: bold;'
    # Se siamo al pareggio (Break-Even), usiamo un blu o arancio
    elif 'Pareggio' in str(val):
        return 'background-color: #3498db; color: white;'
    # Se è lo stop loss iniziale (-10%)
    elif 'Standard' in str(val):
        return 'color: #e74c3c; font-weight: bold;'
    return ''

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
    if val == 'Garantito': return 'color: #FFA500; font-weight: bold;' # Arancione per protezione attiva
    return ''

def calcola_pnl_protetto(prezzo_entrata, prezzo_attuale, direzione, investimento):
    # Calcolo base della variazione percentuale
    if direzione == "COMPRA":
        variazione = (prezzo_attuale - prezzo_entrata) / prezzo_entrata
    else:
        variazione = (prezzo_entrata - prezzo_attuale) / prezzo_entrata
    
    perc = variazione * 100
    
    # --- FILTRO ANTI-FOLLIA ---
    # Se la variazione è assurda (es. > 50% nel Forex in pochi minuti), 
    # ignoriamo il dato per evitare glitch grafici o chiusure errate.
    if abs(perc) > 50: 
        return 0.0, 0.0, True # Ritorna 'True' per indicare un glitch rilevato
        
    profitto_euro = investimento * variazione
    return perc, profitto_euro, False

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
        forex = ["EURUSD=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X", "NZDUSD=X", "EURCHF=X","EURJPY=X", "GBPJPY=X","EURGBP=X"]
        data = yf.download(forex, period="5d", interval="1d", progress=False, timeout=15)
        
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
            "USD 🇺🇸": (-returns.get("EURUSD=X",0) - returns.get("GBPUSD=X",0) + returns.get("USDJPY=X",0) - returns.get("AUDUSD=X",0) + returns.get("USDCAD=X",0) + returns.get("USDCHF=X",0) - returns.get("NZDUSD=X",0) + returns.get("USDCNY=X",0) + returns.get("USDRUB=X",0) + returns.get("USDCOP=X",0) + returns.get("USDARS=X",0) + returns.get("USDBRL=X",0)) / 12,
            "EUR 🇪🇺": (returns.get("EURUSD=X",0) + returns.get("EURJPY=X",0) + returns.get("EURGBP=X",0) + returns.get("EURCHF=X", 0) + returns.get("EURGBP=X", 0) + returns.get("EURJPY=X", 0)) / 6,
            "GBP 🇬🇧": (returns.get("GBPUSD=X",0) + returns.get("GBPJPY=X",0) - returns.get("EURGBP=X",0) + returns.get("GBPCHF=X", 0) + returns.get("GBPJPY=X", 0)) / 5,
            "JPY 🇯🇵": (-returns.get("USDJPY=X",0) - returns.get("EURJPY=X",0) - returns.get("GBPJPY=X",0)) / 3,
            "CHF 🇨🇭": (-returns.get("USDCHF=X",0) - returns.get("EURCHF=X",0) - returns.get("GBPCHF=X",0)) / 3,
            "AUD 🇦🇺": returns.get("AUDUSD=X", 0),
            "NZD 🇳🇿": returns.get("NZDUSD=X", 0),
            "CAD 🇨🇦": -returns.get("USDCAD=X", 0)
            #"CNY 🇨🇳": -returns.get("CNY=X", 0),
            #"RUB 🇷🇺": -returns.get("RUB=X", 0),
            #"COP 🇨🇴": -returns.get("COP=X", 0),
            #"ARS 🇦🇷": -returns.get("ARS=X", 0),
            #"BRL 🇧🇷": -returns.get("BRL=X", 0),
            #"MXN 🇲🇽": -returns.get("MXN=X", 0)
            #"BTC ₿": returns.get("BTC-USD", 0),
            #"ETH 💎": returns.get("ETH-USD", 0)
        }
        return pd.Series(strength).sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float)

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi_col = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi_col.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi_col.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DECRESCITA"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 CRESCITA"
    return "Neutrale"

def send_telegram_msg(msg):
    """Invia notifiche Telegram utilizzando i secrets configurati"""
    url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage"
    payload = {"chat_id": TELE_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=5)
        if r.status_code != 200:
            st.toast(f"Errore Telegram: {r.status_code}", icon="⚠️")
    except Exception as e:
        print(f"Errore invio Telegram: {e}")

def update_signal_outcomes():
    """Monitora i trade, gestisce Break-Even e Trailing Stop con protezione commissioni"""
    if st.session_state['signal_history'].empty: 
        return
    
    df = st.session_state['signal_history']
    updates_made = False
    COMMISSIONE_APPROX = 0.05 # Costo fisso simulato (es. spread/fee)
    
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            ticker = asset_map[row['Asset']]
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            if data.empty: continue
            
            # Prezzi attuali
            current_close = float(data['Close'].iloc[-1])
            current_high = float(data['High'].iloc[-1])
            current_low = float(data['Low'].iloc[-1])
            
            # Valori di ingresso
            entry_v = float(str(row['Prezzo']).replace(',', '.'))
            sl_v = float(str(row['SL']).replace(',', '.'))
            tp_v = float(str(row['TP']).replace(',', '.'))
            investimento = float(str(row['Investimento €']).replace(',', '.'))
            
            # Distanza Stop Loss iniziale (Unità di rischio)
            dist_init = abs(entry_v - sl_v) 
            
            # Calcolo profitto in termini di prezzo
            is_buy = row['Direzione'] == 'COMPRA'
            profitto_prezzo = (current_close - entry_v) if is_buy else (entry_v - current_close)
            
            # --- LOGICA DI PROTEZIONE DINAMICA ---
            target_lvl = 0
            nuovo_sl_val = None
            prot_label = row['Protezione']

            # 1. BREAK-EVEN (Scatta al +1% di ROI teorico, circa 0.1 * dist_init)
            if profitto_prezzo >= (dist_init * 0.1) and "Pareggio" not in row['Protezione']:
                target_lvl = 0.5
                nuovo_sl_val = entry_v 
                prot_label = "Pareggio (Break-Even)"

            # 2. TRAILING STEP PROGRESSIVI
            if profitto_prezzo >= (dist_init * 1.9): 
                target_lvl = 4
                nuovo_sl_val = entry_v + (dist_init * 1.5) if is_buy else entry_v - (dist_init * 1.5)
                prot_label = "Blindato +15%"
            elif profitto_prezzo >= (dist_init * 1.5): 
                target_lvl = 3
                nuovo_sl_val = entry_v + (dist_init * 1.0) if is_buy else entry_v - (dist_init * 1.0)
                prot_label = "Blindato +10%"
            elif profitto_prezzo >= (dist_init * 1.0): 
                target_lvl = 2
                nuovo_sl_val = entry_v + (dist_init * 0.5) if is_buy else entry_v - (dist_init * 0.5)
                prot_label = "Blindato +5%"

            # Verifica se possiamo avanzare di livello (solo in avanti)
            current_lvl_num = float(row['Stato_Prot'].split('_')[1]) if 'LIVELLO' in row['Stato_Prot'] else 0
            if target_lvl > current_lvl_num and nuovo_sl_val is not None:
                p_fmt = "{:.5f}" if "JPY" not in row['Asset'] else "{:.3f}"
                df.at[idx, 'SL'] = p_fmt.format(nuovo_sl_val)
                df.at[idx, 'Stato_Prot'] = f"LIVELLO_{target_lvl}"
                df.at[idx, 'Protezione'] = prot_label
                updates_made = True
                play_safe_sound()

            # --- VERIFICA CHIUSURE ---
            new_status = None
            prezzo_uscita = None

            if is_buy:
                if current_high >= tp_v: 
                    new_status, prezzo_uscita = '✅ TARGET', tp_v
                elif current_low <= sl_v: 
                    new_status, prezzo_uscita = ('🛡️ SL DINAMICO' if target_lvl > 0 else '❌ STOP LOSS'), sl_v
            else: # SELL
                if current_low <= tp_v: 
                    new_status, prezzo_uscita = '✅ TARGET', tp_v
                elif current_high >= sl_v: 
                    new_status, prezzo_uscita = ('🛡️ SL DINAMICO' if target_lvl > 0 else '❌ STOP LOSS'), sl_v

            if new_status:
                # Calcolo variazione percentuale reale all'uscita
                var_finale = (prezzo_uscita - entry_v) / entry_v if is_buy else (entry_v - prezzo_uscita) / entry_v
                
                # Calcolo Risultato con FILTRO PROTEZIONE (minimo 0 se protetto)
                lordo = investimento * var_finale
                netto = lordo - COMMISSIONE_APPROX
                
                # Se il trade era "Blindato" o in "Pareggio", non permettiamo un risultato negativo
                if target_lvl >= 0.5:
                    netto = max(0.0, netto)

                df.at[idx, 'Stato'] = new_status
                df.at[idx, 'Risultato €'] = f"{netto:+.2f}"
                updates_made = True
                play_close_sound()
                send_telegram_msg(f"🔔 **CHIUSURA {row['Asset']}**\nEsito: {new_status}\nNetto: {netto:+.2f}€")

        except Exception as e:
            print(f"Errore calcolo trade {idx}: {e}")
            continue 
        
    if updates_made:
        st.session_state['signal_history'] = df
        save_history_permanently()

def get_asset_params(pair):
    """Restituisce unità, formato, moltiplicatore e tipo asset"""
    if any(x in pair for x in ["BTC", "ETH"]):
        return 1.0, "{:.2f}", 1, "CRYPTO"
    elif "JPY" in pair:
        return 0.01, "{:.3f}", 100, "FOREX_JPY"
    else:
        return 0.0001, "{:.5f}", 10000, "FOREX_STD"

def run_sentinel():
    """Scanner multi-asset per l'individuazione di segnali Mean Reversion e Trend Strength"""
    current_balance = st.session_state.balance_val 
    current_risk = st.session_state.risk_val
    debug_list = []
    
    # 1. Ciclo di scansione su tutti gli asset definiti
    for label, ticker in asset_map.items():
        try:
            # Download dati: 2 giorni a 1m per i segnali, 1 anno a 1d per il trend macro
            df_rt_s = yf.download(ticker, period="2d", interval="1m", progress=False, timeout=10)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False, timeout=10)
            
            if df_rt_s.empty or df_d_s.empty: 
                debug_list.append(f"🔴 {label}: No Data")
                continue
            
            # Pulizia MultiIndex e normalizzazione colonne
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            if isinstance(df_d_s.columns, pd.MultiIndex): df_d_s.columns = df_d_s.columns.get_level_values(0)
            df_rt_s.columns = [c.lower() for c in df_rt_s.columns]
            df_d_s.columns = [c.lower() for c in df_d_s.columns]

            # 2. Calcolo Indicatori Tecnici (Sentinel AI Engine)
            # Bande di Bollinger (20, 2)
            bb_s = ta.bbands(df_rt_s['close'], length=20, std=2)
            if bb_s is None: continue

            c_low = [c for c in bb_s.columns if "BBL" in c.upper()][0]
            c_up = [c for c in bb_s.columns if "BBU" in c.upper()][0]
            
            curr_v = float(df_rt_s['close'].iloc[-1])
            low_bb = float(bb_s[c_low].iloc[-1])
            up_bb = float(bb_s[c_up].iloc[-1])
            
            # RSI Daily (per evitare di andare contro il trend primario)
            rsi_d = ta.rsi(df_d_s['close'], length=14).iloc[-1]
            
            # ADX (per filtrare i mercati troppo volatili/pericolosi)
            adx_df = ta.adx(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1] if adx_df is not None else 0

            # 3. Logica di Ingresso: Mean Reversion Filtrata
            s_action = None
            
            # CONDIZIONE COMPRA: Prezzo < Banda Bassa + RSI non saturo + ADX basso (no trend esplosivo)
            if curr_v < low_bb and rsi_d < 60 and curr_adx < 35: 
                s_action = "COMPRA"
            # CONDIZIONE VENDI: Prezzo > Banda Alta + RSI non saturo + ADX basso
            elif curr_v > up_bb and rsi_d > 40 and curr_adx < 35: 
                s_action = "VENDI"

            # Debug Monitor Update
            icon = "🟢" if s_action else "⚪"
            debug_info = f"{label}: {curr_v:.4f} | ADX: {curr_adx:.1f}"
            if s_action: debug_info += f" -> 🔥 {s_action}"
            debug_list.append(f"{icon} {debug_info}")

            # 4. Gestione Apertura Segnale
            if s_action:
                hist = st.session_state['signal_history']
                
                # Check 1: Evita duplicati (Asset già in corso)
                is_running = not hist.empty and ((hist['Asset'] == label) & (hist['Stato'] == 'In Corso')).any()
                
                # Check 2: Raffreddamento (Cooldown 30 min tra segnali sullo stesso asset)
                recent_signals = False
                if not hist.empty:
                    asset_hist = hist[hist['Asset'] == label]
                    if not asset_hist.empty:
                        last_sig_time = datetime.strptime(asset_hist.iloc[0]['DataOra'], "%H:%M:%S").time()
                        now_time = get_now_rome().time()
                        # Calcolo approssimativo minuti passati
                        minutes_diff = (now_time.hour * 60 + now_time.minute) - (last_sig_time.hour * 60 + last_sig_time.minute)
                        if 0 <= minutes_diff < 30:
                           recent_signals = True

                if not is_running and not recent_signals:
                    # Parametri specifici per l'asset (Pips, Precisione)
                    p_unit, p_fmt, p_mult, a_type = get_asset_params(label)
                    
                    # Calcolo Capitale Investito
                    investimento_totale = current_balance * (current_risk / 100)

                    # Definizione Livelli (0.1% di movimento = Unità base per lo Stop Loss)
                    distanza_base = curr_v * 0.0010 
                    
                    if s_action == "COMPRA":
                        sl = curr_v - distanza_base
                        tp = curr_v + (distanza_base * 2.0)
                    else:
                        sl = curr_v + distanza_base
                        tp = curr_v - (distanza_base * 2.0)
                    
                    # Creazione Dizionario Segnale
                    new_sig = {
                        'DataOra': get_now_rome().strftime("%H:%M:%S"),
                        'Asset': label, 
                        'Direzione': s_action, 
                        'Prezzo': p_fmt.format(curr_v), 
                        'TP': p_fmt.format(tp), 
                        'SL': p_fmt.format(sl), 
                        'Protezione': "Iniziale (Standard)",
                        'Stato_Prot': 'LIVELLO_0',
                        'Stato': 'In Corso',
                        'Investimento €': f"{investimento_totale:.2f}",
                        'Risultato €': "0.00"
                    }
                    
                    # Salvataggio e Notifica
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    save_history_permanently()
                    st.session_state['last_alert'] = new_sig
                    
                    telegram_text = (f"🚀 *NUOVO SEGNALE: {s_action}*\n"
                                     f"📈 Asset: {label}\n"
                                     f"💰 Entry: {new_sig['Prezzo']}\n"
                                     f"🎯 TP: {new_sig['TP']}\n"
                                     f"🛡️ SL: {new_sig['SL']}\n"
                                     f"💳 Risk: {new_sig['Investimento €']}€")
                    send_telegram_msg(telegram_text)

            st.session_state['last_scan_status'] = f"✅ Scan OK: {get_now_rome().strftime('%H:%M:%S')}"

        except Exception as e:
            debug_list.append(f"❌ {label} Error: {str(e)}")
            continue
    
    # Aggiornamento log sidebar
    st.session_state['sentinel_logs'] = debug_list
                    
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
    """Calcola l'andamento del saldo sommando i risultati reali registrati"""
    # 1. Partiamo dal saldo iniziale impostato nella sidebar
    initial_balance = st.session_state.get('balance_val', 1000)
    equity_curve = [initial_balance]
    
    if st.session_state['signal_history'].empty:
        return pd.Series(equity_curve)
    
    # 2. Ordiniamo dal più vecchio al più recente per costruire la curva
    # Nota: Assumiamo che i trade più vecchi siano in fondo, quindi invertiamo se necessario
    # Nel tuo script salvi i nuovi in cima (concat), quindi per la curva temporale dobbiamo invertire (`iloc[::-1]`)
    df_sorted = st.session_state['signal_history'].iloc[::-1]
    
    current_bal = initial_balance
    
    for _, row in df_sorted.iterrows():
        # Prendiamo il valore dalla colonna 'Risultato €'
        val_str = str(row['Risultato €'])
        
        # Puliamo la stringa (rimuoviamo simbolo € o spazi se presenti)
        val_clean = val_str.replace('€', '').replace(',', '.').strip()
        
        try:
            val_float = float(val_clean)
        except:
            val_float = 0.0
            
        # 3. Sommiamo SOLO se il trade è concluso (quindi ha un risultato diverso da 0 o vuoto)
        # Consideriamo validi tutti gli stati di chiusura
        if row['Stato'] in ['✅ TARGET', '❌ STOP LOSS', '🖐️ CHIUSURA MANUALE', '🛡️ SL DINAMICO']:
            current_bal += val_float
            
        equity_curve.append(current_bal)
        
    return pd.Series(equity_curve)

st.sidebar.header("🛠 Trading Desk (1m)")
balance = st.sidebar.number_input("**Conto (€)**", value=1000, key="balance_val")
risk_pc = st.sidebar.slider("**Investimento %**", 0.5, 5.0, 2.0, step=0.5, key="risk_val")

# --- 4. ESECUZIONE SENTINEL ---
# Assicuriamoci che lo scanner giri solo se lo stato è inizializzato
if 'signal_history' in st.session_state:
    run_sentinel()

# --- 5. SIDEBAR ---

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

# --- LOGICA DINAMICA ANALISI OPERATIVA ---
st.sidebar.subheader("📊 Analisi Operativa")

# Recuperiamo il DataFrame della cronologia
df_hist = st.session_state.get('signal_history', pd.DataFrame())

if not df_hist.empty:
    # 1. Conta i trade con stato 'In Corso' o 'APERTO'
    pendenti = len(df_hist[df_hist['Stato'].isin(['In Corso', 'APERTO'])])
    
    # 2. Conta i trade già conclusi (Target, Stop Loss o Chiusi manualmente)
    chiusi = len(df_hist[df_hist['Stato'].isin(['✅ TARGET', '❌ STOP LOSS', '🖐️ CHIUSO MAN.'])])
    
    # 3. Conta i trade vinti per il calcolo veloce (opzionale)
    vinti = len(df_hist[df_hist['Stato'] == '✅ TARGET'])
else:
    pendenti = 0
    chiusi = 0
    vinti = 0

# Visualizzazione Dinamica
st.sidebar.write(f"⏳ **Trade Pendenti:** {pendenti}")
st.sidebar.write(f"✅ **Trade Chiusi:** {chiusi}")

# Un piccolo tocco extra: mostriamo quanti ne abbiamo vinti sul totale dei chiusi
if chiusi > 0:
    st.sidebar.caption(f"🏆 Successi: {vinti} su {chiusi}")

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

# 1. Recupero trade attivi (Assicurati che lo Stato sia 'In Corso' come da tua immagine)
active_trades = st.session_state['signal_history'][st.session_state['signal_history']['Stato'] == 'In Corso']

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Monitor Real-Time")

if active_trades.empty:
    st.sidebar.info("💤 In attesa del primo trade")
else:
    for index, trade in active_trades.iterrows():
        try:
            # Download dati fresco
            t_ticker = asset_map.get(trade['Asset'], trade['Asset'])
            t_data = yf.download(t_ticker, period="1d", interval="1m", progress=False, timeout=5)
                
            if not t_data.empty:
                # --- CORREZIONE VARIABILI ---
                curr_p = float(t_data['Close'].iloc[-1])
                # Pulizia stringhe € se presenti
                entry_p = float(str(trade['Prezzo']).replace('€', '').replace(',', '.').strip())
                inv = float(str(trade['Investimento €']).replace('€', '').replace(',', '.').strip())
                    
                # Moltiplicatore pips (Fondamentale per evitare numeri abnormi come +422514%)
                pips_mult = get_asset_params(trade['Asset'])[2] 
                    
                # Calcolo differenza basato sulla direzione
                if trade['Direzione'] == "BUY" or trade['Direzione'] == "COMPRA":
                    diff_prezzo = curr_p - entry_p
                else:
                    diff_prezzo = entry_p - curr_p
                
                # ... (codice precedente: calcolo latente_perc e latente_euro)
                latente_perc = (diff_prezzo / entry_p) * 100 if trade['Direzione'] == "COMPRA" else -(diff_prezzo / entry_p) * 100
                latente_euro = (inv * latente_perc) / 100 
    
                # --- INIZIO FILTRO ANTI-FOLLIA ---
                # Se la variazione è superiore al 50% (impossibile nel Forex 1m senza glitch),
                # resettiamo i valori a 0 per evitare di inquinare la dashboard.
                is_glitch = False
                if abs(latente_perc) > 50:
                    latente_perc = 0.0
                    latente_euro = 0.0
                    is_glitch = True
                # --- FINE FILTRO ANTI-FOLLIA ---

                color = "#006400" if latente_euro >= 0 else "#FF4B4B"
                if is_glitch: color = "#FFA500" # Arancione per indicare "Dato Dubbio"
                    
                # --- UI MONITOR ---
                st.sidebar.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding-left: 10px; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 5px; margin-bottom: 5px;">
                        <b style="font-size: 0.85em;">{trade['Asset']} | {trade['Direzione']}</b><br>
                        <span style="color:{color}; font-size: 1.1em; font-weight: bold;">
                            {"⚠️ GLITCH DATI" if is_glitch else f"{latente_perc:+.2f}% ({latente_euro:+.2f}€)"}
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
                # ... (resto del codice per il tasto chiudi)
                
                # TASTO CHIUDI (Opzionale)
                if st.sidebar.button(f"✖ Chiudi {trade['Asset']}", key=f"close_{index}"):
                    st.session_state['signal_history'].at[index, 'Stato'] = 'CHIUSO MAN.'
                    # CORREZIONE: Usiamo la f-string :+.2f per forzare segno e decimali
                    st.session_state['signal_history'].at[index, 'Risultato €'] = f"{latente_euro:+.2f}"
                    st.rerun()
        except Exception as e:
            # Mostra l'errore tecnico reale solo per debug se vuoi, altrimenti lascia il messaggio di attesa
            st.sidebar.caption(f"⏳ Aggiornamento {trade['Asset']}...")

st.sidebar.markdown("---")
# ... (restante codice sidebar: sessioni, win rate, reset)
st.sidebar.subheader("🌍 Sessioni di Mercato")
for s_name, is_open in get_session_status().items():
    color = "🟢" if is_open else "🔴"
    status_text = "APERTO" if is_open else "CHIUSO"
    st.sidebar.markdown(f"**{s_name}** <small>: {status_text}</small> {color}",
unsafe_allow_html=True)
   
# --- TASTO ESPORTAZIONE DATI ---
#st.sidebar.markdown("---")
#st.sidebar.subheader("💾 Backup Report")

#if not st.session_state['signal_history'].empty:
    #csv_data = st.session_state['signal_history'].to_csv(index=False).encode('utf-8')
    #st.sidebar.download_button(
        #label="📥 SCARICA CRONOLOGIA CSV",
        #data=csv_data,
        #file_name=f"Trading_Report_{get_now_rome().strftime('%Y%m%d_%H%M')}.csv",
        #mime="text/csv",
        #use_container_width=True
    #)
#else:
    #st.sidebar.info("Nessun dato da esportare")

# --- TASTO TEST TELEGRAM ---
st.sidebar.markdown("---")
if st.sidebar.button("✈️ TEST NOTIFICA TELEGRAM"):
    test_msg = "🔔 **SENTINEL TEST**\nIl sistema di notifiche è operativo! 🚀"
    send_telegram_msg(test_msg)
    st.sidebar.success("Segnale di test inviato!")

# --- TASTO TEST DINAMICO ---
if st.sidebar.button("🔊 TEST ALERT COMPLETO"):
    # Calcolo dinamico basato sui tuoi cursori attuali
    current_bal = st.session_state.get('balance_val', 1000)
    current_r = st.session_state.get('risk_val', 2.0)
    inv_test = current_bal * (current_r / 100)
    
    test_data = {
        'DataOra': get_now_rome().strftime("%dd/%mm/%YYYY %H:%M:%S"),
        'Asset': 'TEST/EUR', 
        'Direzione': 'VENDI', 
        'Prezzo': '1.0950', 
        'TP': '1.0900', 
        'SL': '1.0980', 
        'Stato': 'In Corso',
        'Investimento €': f"{inv_test:.2f}", # Ora legge il 2% di 1000 = 20.00
        'Risultato €': "0.00",
        'Costo Spread €': f"{(inv_test):.2f}",
        'Stato_Prot': 'Iniziale',
        'Protezione': 'Trailing 3/6%'
    }
    
    st.session_state['signal_history'] = pd.concat(
        [pd.DataFrame([test_data]), st.session_state['signal_history']], 
        ignore_index=True
    )
    st.session_state['last_alert'] = test_data
    if 'alert_notified' in st.session_state: del st.session_state['alert_notified']
    st.rerun()

# Reset Sidebar
st.sidebar.markdown("---")
with st.sidebar.popover("🗑️ **Reset Cronologia**"):
    st.warning("Sei sicuro? Questa azione cancellerà tutti i segnali salvati.")

    if st.button("SÌ, CANCELLA ORA"):
        st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
        save_history_permanently() # Questo sovrascrive il file CSV con uno vuoto
        st.rerun()

st.sidebar.markdown("---")

#if st.sidebar.button("TEST ALERT"):
    #st.session_state['last_alert'] = {'Asset': 'TEST/EUR', 'Direzione': 'COMPRA', 'Prezzo': '1.0000', 'TP': '1.0100', 'SL': '0.9900', 'Protezione': 'Standard'}
    #if 'alert_start_time' in st.session_state: del st.session_state['alert_start_time']
    #st.rerun()

#st.sidebar.markdown("---")

# --- 6. POPUP ALERT (VERSIONE NATIVA - NON BLOCCA SIDEBAR) ---
if st.session_state.get('last_alert'):
    # Inizializzazione Timer
    if 'alert_start_time' not in st.session_state:
        st.session_state['alert_start_time'] = time_lib.time()
        play_notification_sound()

    elapsed = time_lib.time() - st.session_state['alert_start_time']
    countdown = max(0, int(30 - elapsed))
    
    # Auto-chiusura
    if elapsed > 30:
        st.session_state['last_alert'] = None
        if 'alert_start_time' in st.session_state: del st.session_state['alert_start_time']
        st.rerun()

    if st.session_state.get('last_alert'):
        alert = st.session_state['last_alert']
        color = "success" if alert['Direzione'] == 'COMPRA' else "error"
        hex_color = "#00ffcc" if alert['Direzione'] == 'COMPRA' else "#ff4b4b"

        # Creiamo un contenitore in cima alla pagina
        with st.container():
            st.markdown(f"""
                <div style="background-color: #000; border: 3px solid {hex_color}; padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align: center; box-shadow: 0 0 20px {hex_color}44;">
                    <h2 style="color: white; margin: 0;">🚀 NUOVO SEGNALE: {alert['Asset']}</h2>
                    <h1 style="color: {hex_color}; margin: 5px 0;">{alert['Direzione']} @ {alert['Prezzo']}</h1>
                    <p style="color: #888; margin: 0;">TP: {alert['TP']} | SL: {alert['SL']} | Auto-chiusura in {countdown}s</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Tasto CHIUDI nativo di Streamlit
            if st.button("✅ HO VISTO, CHIUDI ALERT", key="close_manual", use_container_width=True):
                st.session_state['last_alert'] = None
                if 'alert_start_time' in st.session_state: del st.session_state['alert_start_time']
                st.rerun()
        
        st.divider() # Separa l'alert dal resto del grafico

# --- 7. BODY PRINCIPALE ---
# Banner logic
banner_path = "banner1.png"
if os.path.exists(banner_path):
    st.image(banner_path, use_container_width=True)
else:
    st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1><p style="color: white; opacity: 0.8; margin:0;">Sentinel AI Engine • Forex & Crypto Analysis</p></div>', unsafe_allow_html=True)

st.info(f"🛰️ **Sentinel AI Attiva**: Monitoraggio in corso su {len(asset_map)} asset Forex in tempo reale (1m).")
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

# --- 9. CRONOLOGIA SEGNALI (CON COLORI DINAMICI) ---
st.markdown("---")
st.subheader("📜 Cronologia Segnali")

if not st.session_state['signal_history'].empty:
    display_df = st.session_state['signal_history'].copy()
    display_df = display_df.sort_values(by='DataOra', ascending=False)

    try:
        # Applichiamo gli stili a colonne diverse
        styled_df = display_df.style.map(
            style_status, subset=['Stato']
        ).map(
            style_protection, subset=['Protezione']
        )

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_order=[
                'DataOra', 'Asset', 'Direzione', 'Prezzo', 
                'TP', 'SL', 'Stato', 'Protezione', 
                'Investimento €', 'Risultato €'
            ]
        )
    except Exception as e:
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 4. Pulsante esportazione (Sempre dentro l'IF, ma fuori dal TRY/EXCEPT)
    st.write("") 
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Esporta Cronologia (CSV)",
        data=csv_data,
        file_name=f"trading_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# 5. Se la cronologia è vuota (allineato all'IF iniziale)
else:
    st.info("Nessun segnale registrato.")

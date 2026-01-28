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
from datetime import datetime, timedelta

# --- CONFIGURAZIONE TRADING ---
SIMULATED_SPREAD = 0.0005  # Esempio: 5 pips di spread

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
asset_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X",
            "EURGBP": "EURGBP=X", "GBPJPY": "GBPJPY=X", "EURJPY": "EURJPY=X", "USDCNY": "USDCNY=X", "USDCOP": "USDCOP=X", "USDARS": "USDARS=X", "USDRUB": "USDRUB=X", "USDBRL": "USDBRL=X"}

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

def is_market_open(asset_name):
    today = get_now_rome().weekday()
    # Se è Sabato (5) o Domenica (6), il Forex è chiuso
    if today >= 5:
        return False
        
    return True

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
    base = "font-weight: bold;"
    if val == 'VINTO': return f'{base} color: #00ffcc;'
    if val == 'PERSO': return f'{base} color: #ff4b4b;'
    if val == 'APERTO': return f'{base} color: #ffaa00;'
    if val == 'CHIUSO MAN.': return f'{base} color: #aaaaaa;' # Grigio per chiusura manuale
    if val == '✅ TARGET': return 'background-color: rgba(0, 255, 204, 0.2); color: #00ffcc;'
    if val == '❌ STOP LOSS': return 'background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b;'
    if val == '🛡️ SL DINAMICO': return 'background-color: rgba(255, 165, 0, 0.2); color: #ffa500;'
    
    try:
        # Rimuove il simbolo € e forza il float per il controllo colore
        clean_val = str(val).replace('€', '').replace('+', '').strip()
        num = float(clean_val)
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
        forex = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURCHF=X","EURJPY=X", "GBPJPY=X", "GBPCHF=X","EURGBP=X", "EURGBP=X", "GBPJPY=X", "EURJPY=X", "CNY=X", "COP=X", "ARS=X", "RUB=X", "BRL=X",]
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
            "CAD 🇨🇦": -returns.get("USDCAD=X", 0),
            "CNY 🇨🇳": -returns.get("CNY=X", 0),
            "RUB 🇷🇺": -returns.get("RUB=X", 0),
            "COP 🇨🇴": -returns.get("COP=X", 0),
            "ARS 🇦🇷": -returns.get("ARS=X", 0),
            "BRL 🇧🇷": -returns.get("BRL=X", 0),
            "MXN 🇲🇽": -returns.get("MXN=X", 0)
            #"BTC ₿": returns.get("BTC-USD", 0),
            #"ETH 💎": returns.get("ETH-USD", 0)
        }
        return pd.Series(strength).sort_values(ascending=False)
    except Exception:
        return pd.Series(dtype=float)

def get_asset_params(pair):
    # --- CRYPTO ---
    if "BTC" in pair or "ETH" in pair:
        return 1.0, "{:.2f}", 1, "CRYPTO"
    
    # --- FOREX ESOTICI (Valori Alti) ---
    elif any(x in pair for x in ["COP", "ARS"]):
        # Peso Colombiano e Argentino (es. 3950.50)
        return 1.0, "{:.2f}", 1, "FOREX_LATAM"
    
    # --- FOREX JPY & RUB ---
    elif any(x in pair for x in ["JPY", "RUB"]):
        # Yen e Rublo (es. 150.250 o 90.150)
        return 0.01, "{:.3f}", 100, "FOREX_3DEC"
    
    # --- FOREX STANDARD & CINA ---
    elif "CNY" in pair or "BRL" in pair:
        # Yuan e Real (es. 7.2345 o 4.9567)
        return 0.0001, "{:.4f}", 10000, "FOREX_4DEC"
    
    # --- DEFAULT (EURUSD, ecc) ---
    else:
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
            ticker = asset_map.get(row['Asset'])
            if not ticker: continue
            
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            if data.empty: continue

            # Pulizia colonne
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.columns = [str(c).lower() for c in data.columns]

            current_price = float(data['close'].iloc[-1])
            
            # Conversione sicura da stringa a float per i calcoli
            entry_v = float(str(row['Prezzo']).replace(',', '.'))
            investimento = float(str(row['Investimento €']).replace('€','').replace(',', '.'))
            tp_v = float(str(row['TP']).replace(',', '.'))
            sl_v = float(str(row['SL']).replace(',', '.'))
            current_sl = float(str(row['SL']).replace(',', '.'))
            direzione = row['Direzione']
            
            # --- CALCOLO GAIN ---
            if direzione == 'COMPRA':
                percent_gain = ((current_price - entry_v) / entry_v) * 100
            else:
                percent_gain = ((entry_v - current_price) / entry_v) * 100

            # --- LOGICA TRAILING STOP ---
            new_sl = current_sl
            status_prot = row.get('Stato_Prot', 'Iniziale')

            # Asset Forex
            if "BTC" not in row['Asset'] and "ETH" not in row['Asset']:
                if percent_gain >= 3.0 and 'Iniziale' in status_prot:
                    new_sl = entry_v
                    status_prot = 'BE (0%)'
                    play_safe_sound()
                    send_telegram_msg(f"🛡️ **TARGET DINAMICO**\n{row['Asset']}: Stop Loss a Pareggio!")
                
                elif percent_gain >= 6.0 and 'BE' in status_prot:
                    new_sl = entry_v * 1.05 if direzione == 'COMPRA' else entry_v * 0.95
                    status_prot = 'Safe (+5%)'
                    play_safe_sound()
                    send_telegram_msg(f"💰 **PROFITTO BLINDATO**\n{row['Asset']}: Stop Loss garantisce +5%!")
            
            # Aggiornamento fisico SL se cambiato
            if new_sl != current_sl:
                _, p_fmt, _, _ = get_asset_params(row['Asset'])
                df.at[idx, 'SL'] = p_fmt.format(new_sl)
                df.at[idx, 'Stato_Prot'] = status_prot
                updates_made = True

            # --- CONTROLLO CHIUSURA ---
            target_hit = (direzione == 'COMPRA' and current_price >= tp_v) or (direzione == 'VENDI' and current_price <= tp_v)
            stop_hit = (direzione == 'COMPRA' and current_price <= sl_v) or (direzione == 'VENDI' and current_price >= sl_v)

            if target_hit or stop_hit:
                esito = '✅ TARGET' if target_hit else '❌ STOP LOSS'
                
                # Calcolo profitto netto (Target = x2, Stop = -x1)
                profitto_netto = (investimento * 2) if target_hit else -investimento
                
                # Aggiorniamo il DataFrame
                df.at[idx, 'Stato'] = esito
                df.at[idx, 'Risultato €'] = f"{profitto_netto:+.2f}" # Salviamo come stringa formattata
                updates_made = True
                play_close_sound()
                
                # Notifica Telegram
                icona = "🟢" if s_action == "COMPRA" else "🔴"
                telegram_text = (
                    f"{icona} *{s_action}* {label}\n"
                    f"Entry: {new_sig['Prezzo']}\n"
                    f"TP: {new_sig['TP']}\n"
                    f"SL: {new_sig['SL']}\n"
                    f"-------------------\n"
                    f"€€€: € {new_sig['Investimento €']}"
                )
                send_telegram_msg(telegram_text)

        except Exception as e:
            print(f"Errore aggiornamento {row['Asset']}: {e}")
            continue
    
    if updates_made:
        st.session_state['signal_history'] = df
        save_history_permanently()

def run_sentinel():
    # --- 1. AUTO-PULIZIA MEMORIA ---
    if 'signal_history' not in st.session_state:
        st.session_state['signal_history'] = pd.DataFrame()
        
    if not st.session_state['signal_history'].empty:
        if len(st.session_state['signal_history']) > 50:
            st.session_state['signal_history'] = st.session_state['signal_history'].head(50)
            # save_history_permanently() # Decommenta se hai questa funzione

    # Setup Variabili
    debug_list = []
    assets = list(asset_map.items())
    hist = st.session_state['signal_history']
    
    # --- 2. CICLO DI SCANSIONE (Unico per efficienza) ---
    for label, ticker in assets:
        try:
            # A. SCARICO DATI
            df_rt_s = yf.download(ticker, period="2d", interval="1m", progress=False)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df_rt_s.empty or df_d_s.empty:
                debug_list.append(f"🔴 {label}: No Data")
                continue
            
            # Pulizia Colonne
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            if isinstance(df_d_s.columns, pd.MultiIndex): df_d_s.columns = df_d_s.columns.get_level_values(0)
            
            curr_v = float(df_rt_s['Close'].iloc[-1]) # Prezzo Attuale
            
            # --- B. AGGIORNAMENTO TRADE APERTI (Check TP/SL) ---
            # Controlliamo se abbiamo trade aperti per QUESTO asset
            if not hist.empty:
                indices = hist[(hist['Asset'] == label) & (hist['Stato'] == 'APERTO')].index
                for idx in indices:
                    trade = hist.loc[idx]
                    tp = float(str(trade['TP']).replace(',', '.'))
                    sl = float(str(trade['SL']).replace(',', '.'))
                    
                    outcome = None
                    risultato = 0.0
                    
                    if trade['Direzione'] == 'BUY':
                        if curr_v >= tp: outcome, risultato = 'VINTO', 40.0
                        elif curr_v <= sl: outcome, risultato = 'PERSO', -20.0
                    elif trade['Direzione'] == 'SELL':
                        if curr_v <= tp: outcome, risultato = 'VINTO', 40.0
                        elif curr_v >= sl: outcome, risultato = 'PERSO', -20.0
                    
                    if outcome:
                        st.session_state['signal_history'].at[idx, 'Stato'] = outcome
                        st.session_state['signal_history'].at[idx, 'Risultato €'] = risultato
                        st.toast(f"🔔 {label} Trade Chiuso: {outcome} ({risultato}€)")

            # --- C. CALCOLO INDICATORI (Per Nuovi Segnali) ---
            # Bande di Bollinger
            bb_s = ta.bbands(df_rt_s['Close'], length=20, std=2)
            if bb_s is None: continue
            
            lower_col = [c for c in bb_s.columns if "BBL" in c][0]
            upper_col = [c for c in bb_s.columns if "BBU" in c][0]
            low_bb = bb_s[lower_col].iloc[-1]
            up_bb = bb_s[upper_col].iloc[-1]
            
            # RSI e ADX
            rsi_d = ta.rsi(df_d_s['Close'], length=14).iloc[-1]
            adx_df = ta.adx(df_rt_s['High'], df_rt_s['Low'], df_rt_s['Close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1] if adx_df is not None else 0

            # --- D. LOGICA INGRESSO ---
            s_action = None
            if curr_v < low_bb and rsi_d < 60 and curr_adx < 45: 
                s_action = "BUY"
            elif curr_v > up_bb and rsi_d > 40 and curr_adx < 45: 
                s_action = "SELL"
            
            # Debug Monitor Info
            icon = "🟢" if s_action else "⚪"
            debug_list.append(f"{icon} {label}: {curr_v:.4f} | RSI: {rsi_d:.1f}")

            # --- E. ESECUZIONE SEGNALE ---
            if s_action:
                # 1. Recupero Parametri
                # Assumo che get_asset_params ritorni: (unit, decimali, multiplier, type)
                params = get_asset_params(label)
                p_decimals = params[1]
                p_mult = params[2]

                # 2. Controllo Anti-Spam (30 min)
                recent_signals = False
                asset_hist = hist[hist['Asset'] == label]
                if not asset_hist.empty:
                    last_time_str = asset_hist.iloc[0]['DataOra']
                    try:
                        # Gestisce formati diversi per sicurezza
                        fmt = "%d/%m/%Y %H:%M:%S"
                        last_dt = datetime.strptime(last_time_str, fmt)
                        if (datetime.now() - last_dt).total_seconds() / 60 < 30:
                            recent_signals = True
                    except:
                        pass # Se errore data, assume nessun segnale recente

                if not recent_signals:
                    # 3. Calcoli Prezzi
                    entry_price = curr_v
                    
                    # Calcolo TP/SL (Risk Reward 1:2)
                    # Distanza base: 0.2% del prezzo (adattabile)
                    distanza = entry_price * 0.002 
                    
                    if s_action == "BUY":
                        sl_price = entry_price - distanza       # Rischio 1
                        tp_price = entry_price + (distanza * 2) # Reward 2
                    else: # SELL
                        sl_price = entry_price + distanza
                        tp_price = entry_price - (distanza * 2)

                    # Calcolo Spread (Formula corretta)
                    investimento = 20.0
                    costo_spread = (SIMULATED_SPREAD * p_mult) * (investimento / 10.0)

                    # 4. Creazione Dizionario Segnale
                    new_sig = {
                        'DataOra': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'Asset': label,
                        'Direzione': s_action,
                        'Prezzo': f"{entry_price:.{p_decimals}f}",
                        'TP': f"{tp_price:.{p_decimals}f}",
                        'SL': f"{sl_price:.{p_decimals}f}",
                        'Stato': 'APERTO',
                        'Investimento €': investimento,
                        'Risultato €': 0.00,
                        'Costo Spread €': costo_spread,
                        'Stato_Prot': 'Iniziale'
                    }

                    # 5. Salvataggio e Notifica
                    st.session_state['signal_history'] = pd.concat([
                        pd.DataFrame([new_sig]), 
                        st.session_state['signal_history']
                    ], ignore_index=True)
                    
                    st.session_state['sent_signals'].add(label)
                    
                    # Notifica UI
                    st.toast(f"🚀 {label}: Ordine {s_action} aperto!", icon="🔥")
                    
                    # Notifica Telegram (Opzionale)
                    # msg = f"🔥 {s_action} {label}\nPrice: {new_sig['Prezzo']}\nTP: {new_sig['TP']}"
                    # send_telegram_msg(msg)
                
                else:
                    debug_list.append(f"⏳ {label}: Ignorato (Recente)")

        except Exception as e:
            debug_list.append(f"❌ {label} Err: {str(e)}")
            continue

    # Chiusura Funzione
    st.session_state['sentinel_logs'] = debug_list
    st.session_state['last_scan_status'] = f"✅ Scan OK: {datetime.now().strftime('%H:%M:%S')}"

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
    initial_balance = balance 
    risk_pc = st.session_state.get('risk_val', 2.0)
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
            height: 12px; margin-bottom: 25px; border: 1px solid #995; overflow: hidden;
        }
        .red-bar {
            height: 100%; background-color: #ff4b4b; width: 0%;
            animation: progressFill 60s linear infinite;
            box-shadow: 0 0 10px #ff4b4b;
        }
    </style>
    <div class="container-bar"><div class="red-bar"></div></div>
""", unsafe_allow_html=True)

# Status Sentinel
logs = st.session_state.get('sentinel_logs', [])
status = st.session_state.get('last_scan_status', "Init...")
if "❌" in str(logs): st.sidebar.warning("Alcuni dati mancanti")
else: st.sidebar.success(status)

with st.sidebar.expander("🔍 Log Dettagliato"):
    for l in logs: st.caption(l)

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

# Grafico Equity (Piccolo e pulito)
#fig_equity = go.Figure()
#fig_equity.add_trace(go.Scatter(y=equity_series, mode='lines', fill='tozeroy', line=dict(color='#00ffcc')))
#fig_equity.update_layout(height=100, margin=dict(l=0,r=0,t=0,b=0), xaxis_visible=False, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
#st.sidebar.plotly_chart(fig_equity, use_container_width=True, config={'displayModeBar': False})

# Dettagli operazione selezionata (se presente)

# 1. Recupero trade attivi (Assicurati che lo Stato sia 'In Corso' come da tua immagine)
active_trades = st.session_state['signal_history'][st.session_state['signal_history']['Stato'] == 'In Corso']

if not active_trades.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Monitor Real-Time")
    
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
                
                # Calcolo profitto e percentuale corretti
                latente_euro = diff_prezzo * pips_mult * (inv / 10)
                latente_perc = (diff_prezzo / entry_p) * 100
                
                color = "#00FFCC" if latente_euro >= 0 else "#FF4B4B"
                
                # --- UI MONITOR ---
                st.sidebar.markdown(f"""
                    <div style="border-left: 4px solid {color}; padding-left: 10px; background: rgba(255,255,255,0.05); padding: 8px; border-radius: 5px; margin-bottom: 5px;">
                        <b style="font-size: 0.85em;">{trade['Asset']} | {trade['Direzione']}</b><br>
                        <span style="color:{color}; font-size: 1.1em; font-weight: bold;">
                            {latente_perc:+.2f}% ({latente_euro:+.2f}€)
                        </span>
                    </div>
                """, unsafe_allow_html=True)
                
                # TASTO CHIUDI (Opzionale)
                if st.sidebar.button(f"✖ Chiudi {trade['Asset']}", key=f"close_{index}"):
                    st.session_state['signal_history'].at[index, 'Stato'] = 'CHIUSO MAN.'
                    st.session_state['signal_history'].at[index, 'Risultato €'] = round(latente_euro, 2)
                    st.rerun()
        except Exception as e:
            # Mostra l'errore tecnico reale solo per debug se vuoi, altrimenti lascia il messaggio di attesa
            st.sidebar.caption(f"⏳ Aggiornamento {trade['Asset']}...")

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
        'DataOra': get_now_rome().strftime("%H:%M:%S"),
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

# --- 6. POPUP ALERT (OTTIMIZZATO) ---
if st.session_state.get('last_alert'):
    alert = st.session_state['last_alert']
    
    # Suona solo la prima volta
    if 'alert_notified' not in st.session_state:
        play_notification_sound()
        st.session_state['alert_notified'] = True
        # Registriamo quando è apparso l'alert
        st.session_state['alert_time'] = time_lib.time()

    hex_color = "#00ffcc" if alert['Direzione'] == 'COMPRA' else "#ff4b4b"

    st.markdown(f"""
        <div style="background-color: #000; border: 3px solid {hex_color}; padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 0 20px {hex_color}44;">
            <h2 style="color: white; margin: 0;">🚀 NUOVO SEGNALE: {alert['Asset']}</h2>
            <h1 style="color: {hex_color}; margin: 5px 0;">{alert['Direzione']} @ {alert['Prezzo']}</h1>
            <p style="color: #888;">TP: {alert['TP']} | SL: {alert['SL']} | Protezione: {alert.get('Protezione', 'Standard')}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tasto di chiusura
    if st.button("✅ CHIUDI", use_container_width=True):
        st.session_state['last_alert'] = None
        if 'alert_notified' in st.session_state: del st.session_state['alert_notified']
        st.rerun()

    # Autorefresh specifico per il popup (opzionale: lo chiude dopo 5 minuti se non cliccato)
    if time_lib.time() - st.session_state.get('alert_time', 0) > 180: # 3 minuti
        st.session_state['last_alert'] = None
        if 'alert_notified' in st.session_state: del st.session_state['alert_notified']
    
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

# --- 8. CURRENCY STRENGTH (ORDINATO 7x2) ---
st.markdown("---")
st.subheader("⚡ Currency Strength Meter")
s_data = get_currency_strength()

if not s_data.empty:
    items = list(s_data.items())
    # Divisione in due blocchi da 7
    riga1 = items[:7]
    riga2 = items[7:14]

    for riga in [riga1, riga2]:
        cols = st.columns(7)
        for i, (curr, val) in enumerate(riga):
            # Colori dinamici basati sulla forza
            if val > 0.20:
                bg, border = "rgba(0, 168, 107, 0.15)", "#006400" # Molto forte
            elif val < -0.20:
                bg, border = "rgba(220, 20, 60, 0.15)", "#ff4b4b"  # Molto debole
            else:
                bg, border = "rgba(178, 178, 178, 0.05)", "#444"   # Neutra

            with cols[i]:
                st.markdown(
                    f"""
                    <div style='text-align:center; background:{bg}; padding:8px; border-radius:8px; 
                                border:1px solid {border}; min-height:85px; margin-bottom:10px;'>
                        <div style='font-size:0.8em; color:#000000; margin-bottom:4px;'>RANK {items.index((curr,val))+1}</div>
                        <b style='color:black; font-size:0.9em;'>{curr}</b><br>
                        <span style='color:{border}; font-size:1.1em; font-weight:bold;'>{val:+.2f}%</span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
else:
    st.info("⏳ Analisi macro-volatilità in corso...")

# --- 9. CRONOLOGIA SEGNALI (FIX PROFITTO NETTO) ---
st.markdown("---")
st.subheader("📜 Cronologia Segnali")

if not st.session_state['signal_history'].empty:
    # 1. Preparazione dati (Copia UNICA e pulizia numerica immediata)
    df_base = st.session_state['signal_history'].copy()
    
    # Puliamo le colonne monetarie per i calcoli matematici
    cols_da_pulire = ['Investimento €', 'Risultato €', 'Costo Spread €']
    for col in cols_da_pulire:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(
                df_base[col].astype(str).str.replace('€', '').str.replace(',', '.').str.strip(), 
                errors='coerce'
            ).fillna(0.0)

    # 2. Calcolo Statistiche (Dashboard)
    df_conclusi = df_base[df_base['Stato'].isin(['VINTO', 'PERSO'])]
    tot_conclusi = len(df_conclusi)
    vinti = len(df_conclusi[df_conclusi['Stato'] == 'VINTO'])
    
    win_rate = (vinti / tot_conclusi * 100) if tot_conclusi > 0 else 0
    profitto_netto = df_base['Risultato €'].sum()
    rendimento_medio = df_base['Risultato €'].mean() if tot_conclusi > 0 else 0

    # 3. Visualizzazione Dashboard Metriche
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
    with m2:
        # Il delta mostra il profitto totale in verde/rosso
        st.metric("💰 Profitto Netto", f"€ {profitto_netto:.2f}", delta=f"{profitto_netto:.2f} €")
    with m3:
        st.metric("📊 Media x Trade", f"€ {rendimento_medio:.2f}")

    # --- GRAFICO EQUITY CURVE ---
    if not df_conclusi.empty:
        # Ordiniamo cronologicamente (dal più vecchio al più recente)
        df_chart = df_conclusi.iloc[::-1].copy()
        df_chart['Equity'] = df_chart['Risultato €'].cumsum()
        
        st.line_chart(df_chart['Equity'], use_container_width=True)
        st.caption("📈 Andamento del Profitto Cumulativo (€)")
   
    st.markdown("---")

    # 4. Gestione Tabella e Filtri
    df_visualizzazione = df_base.copy() # Usiamo i dati già puliti
    cols_necessarie = ['DataOra', 'Asset', 'Direzione', 'Prezzo', 'TP', 'SL', 'Stato', 'Investimento €', 'Risultato €', 'Costo Spread €', 'Stato_Prot']
    
    for col in cols_necessarie:
        if col not in df_visualizzazione.columns:
            df_visualizzazione[col] = "-"

    # Interfaccia Filtri
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        opzioni_stato = sorted([str(x) for x in df_visualizzazione['Stato'].unique()])
        filtro_stato = st.multiselect("Filtra Esito:", options=opzioni_stato)
    with col_f2:
        opzioni_asset = sorted([str(x) for x in df_visualizzazione['Asset'].unique()])
        filtro_asset = st.multiselect("Filtra Valuta:", options=opzioni_asset)

    if filtro_stato:
        df_visualizzazione = df_visualizzazione[df_visualizzazione['Stato'].isin(filtro_stato)]
    if filtro_asset:
        df_visualizzazione = df_visualizzazione[df_visualizzazione['Asset'].isin(filtro_asset)]
    
    # 5. Rendering Tabella con Styler
    if not df_visualizzazione.empty:
        format_dict = {
            'Investimento €': '€ {:.2f}',
            'Risultato €': '€ {:+.2f}',
            'Costo Spread €': '€ {:.2f}'
        }

        try:
            # Qui applichiamo il grassetto (font-weight: bold) e i colori
            styled_df = (df_visualizzazione.style
                .format(format_dict)
                .map(style_status, subset=['Stato', 'Risultato €']))
        except:
            styled_df = (df_visualizzazione.style
                .format(format_dict)
                .applymap(style_status, subset=['Stato', 'Risultato €']))

        st.dataframe(styled_df, use_container_width=True, hide_index=True, column_order=cols_necessarie)
        
        st.download_button(
            label=f"📥 Esporta {len(df_visualizzazione)} righe",
            data=df_visualizzazione.to_csv(index=False).encode('utf-8'),
            file_name="cronologia_trading.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("Nessun segnale trovato con i filtri selezionati.")
else:
    st.info("📖 In attesa del primo segnale della sentinella...")

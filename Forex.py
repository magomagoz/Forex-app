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

# --- 2. FUNZIONI DI UTILITÀ ---
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

def get_asset_params(pair):
    if "BTC" in pair or "ETH" in pair:
        return 1.0, "{:.2f}", 1, "CRYPTO"
    elif any(x in pair for x in ["COP", "ARS"]):
        # Peso Colombiano e Argentino (es. 3950.50)
        return 1.0, "{:.2f}", 1, "FOREX_LATAM"
    elif any(x in pair for x in ["JPY", "RUB"]):
        # Yen e Rublo (es. 150.250 o 90.150)
        return 0.01, "{:.3f}", 100, "FOREX_3DEC"
    elif "CNY" in pair or "BRL" in pair:
        # Yuan e Real (es. 7.2345 o 4.9567)
        return 0.0001, "{:.4f}", 10000, "FOREX_4DEC"
    else:
        return 0.0001, "{:.5f}", 10000, "FOREX_STD"

def get_realtime_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, timeout=5)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def detect_divergence(df):
    if len(df) < 20: return "Analisi..."
    price, rsi_col = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi_col.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi_col.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DECRESCITA"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 CRESCITA"
    return "Neutrale"

def get_session_status():
    now = get_now_rome().time()
    return {
        "Tokyo 🇯🇵": (time(1,0), time(9,0)), 
        "Londra 🇬🇧": (time(9,0), time(17,30)), 
        "New York 🇺🇸": (time(14,0), time(22,0))
    }

def is_market_open(asset_name):
    today = get_now_rome().weekday()
    # Se è Sabato (5) o Domenica (6), il Forex è chiuso
    if today >= 5:
        return False
        
    return True

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
    if "BTC" in pair or "ETH" in pair:
        return 1.0, "{:.2f}", 1, "CRYPTO"
    elif any(x in pair for x in ["COP", "ARS"]):
        # Peso Colombiano e Argentino (es. 3950.50)
        return 1.0, "{:.2f}", 1, "FOREX_LATAM"
    elif any(x in pair for x in ["JPY", "RUB"]):
        # Yen e Rublo (es. 150.250 o 90.150)
        return 0.01, "{:.3f}", 100, "FOREX_3DEC"
    elif "CNY" in pair or "BRL" in pair:
        # Yuan e Real (es. 7.2345 o 4.9567)
        return 0.0001, "{:.4f}", 10000, "FOREX_4DEC"
    else:
        return 0.0001, "{:.5f}", 10000, "FOREX_STD"

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


# --- 3. CORE LOGIC: SENTINEL ENGINE (UNIFICATO) ---
def run_sentinel():
    # Inizializzazione sessione
    if 'signal_history' not in st.session_state:
        st.session_state['signal_history'] = pd.DataFrame()
    
    if 'sent_signals' not in st.session_state:
        st.session_state['sent_signals'] = set()

    hist = st.session_state['signal_history']
    
    # 1. Cleanup Memoria
    if not hist.empty and len(hist) > 50:
        st.session_state['signal_history'] = hist.head(50)

    debug_list = []
    assets = list(asset_map.items())
    
    # Parametri globali utente
    balance = st.session_state.get('balance_val', 1000)
    risk_pc = st.session_state.get('risk_val', 2.0)
    
    # --- CICLO DI SCANSIONE SINGOLO ---
    for label, ticker in assets:
        try:
            # A. Scarico Dati
            df_rt = yf.download(ticker, period="2d", interval="1m", progress=False)
            df_d = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df_rt.empty or df_d.empty:
                debug_list.append(f"🔴 {label}: No Data")
                continue
                
            # Pulizia Colonne
            if isinstance(df_rt.columns, pd.MultiIndex): df_rt.columns = df_rt.columns.get_level_values(0)
            if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
            
            curr_v = float(df_rt['Close'].iloc[-1])
            
            # --- B. CONTROLLO TRADE APERTI (TP/SL) ---
            if not hist.empty:
                # Trova indici dei trade aperti per questo asset
                open_indices = hist[(hist['Asset'] == label) & (hist['Stato'].isin(['APERTO', 'In Corso']))].index
                
                for idx in open_indices:
                    trade = hist.loc[idx]
                    
                    # Parsing sicuro dei valori
                    try:
                        tp_val = float(str(trade['TP']).replace(',', '.'))
                        sl_val = float(str(trade['SL']).replace(',', '.'))
                        inv_val = float(str(trade['Investimento €']).replace('€','').replace(',','.'))
                        entry_val = float(str(trade['Prezzo']).replace(',', '.'))
                        direzione = trade['Direzione']
                        p_mult = get_asset_params(label)[2]
                    except:
                        continue # Salta se dati corrotti
                    
                    outcome = None
                    profitto = 0.0
                    
                    # Logica TP/SL
                    if direzione in ['BUY', 'COMPRA']:
                        if curr_v >= tp_val: outcome = 'VINTO'
                        elif curr_v <= sl_val: outcome = 'PERSO'
                        pips_diff = curr_v - entry_val
                    else: # SELL
                        if curr_v <= tp_val: outcome = 'VINTO'
                        elif curr_v >= sl_val: outcome = 'PERSO'
                        pips_diff = entry_val - curr_v
                    
                    if outcome:
                        # Calcolo Profitto Matematico: Pips * ValorePip
                        # Valore Pip stimato su inv 20€ leva alta: approx 1€ per 10 pips standard
                        # Formula semplificata per coerenza:
                        raw_gain = pips_diff * p_mult 
                        profitto = raw_gain * (inv_val / 10.0) # Scalato sull'investimento
                        
                        st.session_state['signal_history'].at[idx, 'Stato'] = outcome
                        st.session_state['signal_history'].at[idx, 'Risultato €'] = round(profitto, 2)
                        st.toast(f"🔔 {label}: {outcome} ({profitto:.2f}€)", icon="💰" if profitto>0 else "💀")
                        play_sound('cash' if profitto > 0 else 'notification')
                        save_history_permanently()

            # --- C. CALCOLO NUOVI SEGNALI ---
            bb = ta.bbands(df_rt['Close'], length=20, std=2)
            rsi = ta.rsi(df_d['Close'], length=14).iloc[-1]
            adx = ta.adx(df_rt['High'], df_rt['Low'], df_rt['Close'], length=14)
            
            if bb is None or adx is None: continue
            
            lower_b = bb[bb.columns[0]].iloc[-1] # BBL
            upper_b = bb[bb.columns[2]].iloc[-1] # BBU
            curr_adx = adx['ADX_14'].iloc[-1]
            
            s_action = None
            if curr_v < lower_b and rsi < 60 and curr_adx < 45: s_action = "COMPRA"
            elif curr_v > upper_b and rsi > 40 and curr_adx < 45: s_action = "VENDI"
            
            # Debug UI
            icon = "🟢" if s_action else "⚪"
            debug_list.append(f"{icon} {label}: {curr_v:.4f} | RSI: {rsi:.0f}")
            
            # --- D. ESECUZIONE SEGNALE ---
            if s_action:
                # 1. Recupero Parametri
                _, p_fmt, p_mult, _ = get_asset_params(label)
                
                # 2. Controllo Anti-Spam (30 min)
                recent = False
                asset_hist = hist[hist['Asset'] == label]
                if not asset_hist.empty:
                    last_t = asset_hist.iloc[0]['DataOra']
                    try:
                        last_dt = datetime.strptime(str(last_t), "%d/%m/%Y %H:%M:%S")
                        if (datetime.now() - last_dt).total_seconds() < 1800: recent = True
                    except: pass
                
                if not recent:
                    # 3. Setup Trade
                    entry_p = curr_v
                    dist = entry_p * 0.002 # 0.2% distanza
                    
                    if s_action == "COMPRA":
                        sl = entry_p - dist
                        tp = entry_p + (dist * 2)
                    else:
                        sl = entry_p + dist
                        tp = entry_p - (dist * 2)
                    
                    inv_calc = balance * (risk_pc / 100)
                    costo_spread = (SIMULATED_SPREAD * p_mult) * (inv_calc / 20.0) # Stima costo
                    
                    new_sig = {
                        'DataOra': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        'Asset': label,
                        'Direzione': s_action,
                        'Prezzo': p_fmt.format(entry_p),
                        'TP': p_fmt.format(tp),
                        'SL': p_fmt.format(sl),
                        'Stato': 'In Corso',
                        'Investimento €': f"{inv_calc:.2f}",
                        'Risultato €': 0.00,
                        'Costo Spread €': costo_spread,
                        'Stato_Prot': 'Iniziale',
                        'Protezione': 'Trailing 3/6%'
                    }
                    
                    # 4. Salvataggio
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    st.session_state['last_alert'] = new_sig
                    save_history_permanently()
                    
                    # 5. Notifiche
                    play_sound('notification')
                    msg = f"🚀 *SEGNALE {label}*\n{s_action} @ {new_sig['Prezzo']}\nTP: {new_sig['TP']} | SL: {new_sig['SL']}\nInv: €{new_sig['Investimento €']}"
                    send_telegram_msg(msg)
                    st.toast(f"Ordine {label} Aperto!", icon="🚀")

        except Exception as e:
            debug_list.append(f"❌ {label} Err: {str(e)}")
            continue
            
    st.session_state['sentinel_logs'] = debug_list
    st.session_state['last_scan_status'] = f"✅ Scan OK: {get_now_rome().strftime('%H:%M:%S')}"

def display_performance_stats():
    if st.session_state['signal_history'].empty:
        return
    
    df = st.session_state['signal_history']
    conclusi = df[df['Stato'].str.contains('TARGET|STOP|DINAMICO', na=False)]
    
    if not conclusi.empty:
        vittorie = len(conclusi[conclusi['Stato'] == '✅ TARGET'])
        wr = (vittorie / len(conclusi)) * 100
        st.sidebar.write(f"📊 **Win Rate**: {wr:.1f}% ({vittorie}/{len(conclusi)})")

# --- 4. INIZIALIZZAZIONE STATO (Session State) ---
if 'signal_history' not in st.session_state: 
    st.session_state['signal_history'] = load_history_from_csv()
if 'sentinel_logs' not in st.session_state:
    st.session_state['sentinel_logs'] = []
if 'last_alert' not in st.session_state:
    st.session_state['last_alert'] = None
if 'last_scan_status' not in st.session_state:
    st.session_state['last_scan_status'] = "In attesa..."

# --- 5. ESECUZIONE AGGIORNAMENTO DATI (PRIMA DELLA GUI) ---
# Importante: Aggiorniamo i risultati TP/SL prima di disegnare la sidebar
#update_signal_outcomes()

# Esecuzione Sentinel
run_sentinel()

# --- 5. INTERFACCIA GRAFICA (SIDEBAR) ---
st.sidebar.header("🛠 Trading Desk")

# Status Sentinel
logs = st.session_state.get('sentinel_logs', [])
status = st.session_state.get('last_scan_status', "Init...")
if "❌" in str(logs): st.sidebar.warning("Alcuni dati mancanti")
else: st.sidebar.success(status)

with st.sidebar.expander("🔍 Log Dettagliato"):
    for l in logs: st.caption(l)

# Input
selected_label = st.sidebar.selectbox("Asset Grafico", list(asset_map.keys()))
balance = st.sidebar.number_input("Saldo Conto (€)", value=1000, key="balance_val")
risk_pc = st.sidebar.slider("Rischio %", 0.5, 5.0, 2.0, step=0.5, key="risk_val")
inv_sim = balance * (risk_pc / 100)
st.sidebar.caption(f"Investimento per trade: € {inv_sim:.2f}")

st.sidebar.markdown("---")

# MONITOR LIVE (Corretto)
active = st.session_state['signal_history']
if not active.empty:
    active = active[active['Stato'].isin(['In Corso', 'APERTO'])]

if not active.empty:
    st.sidebar.subheader("⚡ Monitor Ordini")
    for idx, row in active.iterrows():
        try:
            # Dati live
            ticker = asset_map.get(row['Asset'])
            live_df = yf.download(ticker, period="1d", interval="1m", progress=False)
            
            if not live_df.empty:
                curr_p = float(live_df['Close'].iloc[-1])
                entry_p = float(str(row['Prezzo']).replace(',', '.'))
                inv_p = float(str(row['Investimento €']).replace('€','').replace(',','.'))
                
                # Calcolo Pips
                _, _, p_mult, _ = get_asset_params(row['Asset'])
                if row['Direzione'] in ['COMPRA', 'BUY']:
                    diff = curr_p - entry_p
                else:
                    diff = entry_p - curr_p
                
                profit_eur = diff * p_mult * (inv_p / 10.0)
                profit_perc = (diff / entry_p) * 100
                
                color = "#00FFCC" if profit_eur >= 0 else "#FF4B4B"
                
                st.sidebar.markdown(f"""
                <div style="border-left: 3px solid {color}; padding-left: 8px; margin-bottom: 5px; background:rgba(255,255,255,0.05)">
                    <b>{row['Asset']}</b> ({row['Direzione']})<br>
                    <span style="color:{color}; font-weight:bold;">{profit_eur:+.2f}€ ({profit_perc:+.2f}%)</span>
                </div>
                """, unsafe_allow_html=True)
                
                if st.sidebar.button(f"Chiudi {row['Asset']}", key=f"cls_{idx}"):
                    st.session_state['signal_history'].at[idx, 'Stato'] = 'CHIUSO MAN.'
                    st.session_state['signal_history'].at[idx, 'Risultato €'] = round(profit_eur, 2)
                    save_history_permanently()
                    st.rerun()
        except:
            st.sidebar.caption(f"Caricamento {row['Asset']}...")

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

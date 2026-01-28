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

def is_market_open(asset_name):
    today = get_now_rome().weekday()
    # Se è Sabato (5) o Domenica (6), il Forex è chiuso
    if today >= 5:
        return False
        
    return True

def play_sound(sound_type='notification'):
    urls = {
        'notification': "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
        'cash': "https://assets.mixkit.co/active_storage/sfx/2017/2017-preview.mp3",
        'safe': "https://assets.mixkit.co/active_storage/sfx/2021/2021-preview.mp3"
    }
    src = urls.get(sound_type, urls['notification'])
    st.markdown(f"""<audio autoplay><source src="{src}" type="audio/mpeg"></audio>""", unsafe_allow_html=True)

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
    """Restituisce: (unit, format_string, multiplier, type)"""
    if "BTC" in pair or "ETH" in pair: return 1.0, "{:.2f}", 1, "CRYPTO"
    elif any(x in pair for x in ["COP", "ARS"]): return 1.0, "{:.2f}", 1, "FOREX_LATAM"
    elif any(x in pair for x in ["JPY", "RUB"]): return 0.01, "{:.3f}", 100, "FOREX_3DEC"
    elif "CNY" in pair or "BRL" in pair: return 0.0001, "{:.4f}", 10000, "FOREX_4DEC"
    else: return 0.0001, "{:.5f}", 10000, "FOREX_STD"

def get_realtime_data(ticker):
    try:
        df = yf.download(ticker, period="5d", interval="5m", progress=False, timeout=5)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def detect_divergence(df):
    if len(df) < 20: return "..."
    price = df['close'] if 'close' in df else df['Close']
    rsi_col = df['rsi'] if 'rsi' in df else df['RSI']
    
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p = price.iloc[-20:-1].max()
    prev_max_r = rsi_col.iloc[-20:-1].max()
    
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 Bearish Div."
    return "Neutrale"

def get_session_status():
    now = get_now_rome().time()
    return {
        "Tokyo 🇯🇵": (time(1,0), time(9,0)), 
        "Londra 🇬🇧": (time(9,0), time(17,30)), 
        "New York 🇺🇸": (time(14,0), time(22,0))
    }

def get_currency_strength():
    # Funzione semplificata per evitare blocchi yfinance su troppi ticker
    return pd.Series(dtype=float) 

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

# --- 4. INIZIALIZZAZIONE & STATE ---
if 'signal_history' not in st.session_state: 
    st.session_state['signal_history'] = load_history_from_csv()
if 'sentinel_logs' not in st.session_state: st.session_state['sentinel_logs'] = []
if 'last_alert' not in st.session_state: st.session_state['last_alert'] = None

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

# --- 6. POPUP ALERT ---
if st.session_state['last_alert']:
    alert = st.session_state['last_alert']
    color = "#00ffcc" if alert['Direzione'] in ['COMPRA', 'BUY'] else "#ff4b4b"
    st.markdown(f"""
    <div style="border: 2px solid {color}; padding: 15px; border-radius: 10px; text-align: center; background: #111;">
        <h2 style="margin:0; color:{color}">🔥 SEGNALE: {alert['Asset']}</h2>
        <h3 style="margin:5px;">{alert['Direzione']} @ {alert['Prezzo']}</h3>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Chiudi Alert"):
        st.session_state['last_alert'] = None
        st.rerun()

# --- 7. MAIN DASHBOARD ---
st.title("📊 Forex Momentum AI")
st.caption(f"Sessione Attiva: {selected_label} | Aggiornato: {get_now_rome().strftime('%H:%M:%S')}")

# Grafico
ticker = asset_map[selected_label]
data = get_realtime_data(ticker)
if data is not None:
    # Calcolo Indicatori
    data['MA20'] = data['Close'].rolling(20).mean()
    data['Upper'] = data['MA20'] + 2*data['Close'].rolling(20).std()
    data['Lower'] = data['MA20'] - 2*data['Close'].rolling(20).std()
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper'], line=dict(color='rgba(0,255,255,0.3)'), name='BB Up'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower'], line=dict(color='rgba(0,255,255,0.3)'), fill='tonexty', name='BB Low'))
    fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

# Tabella Cronologia
st.subheader("📜 Storico Operazioni")
df = st.session_state['signal_history']
if not df.empty:
    # Pulizia Dati
    df_view = df.copy()
    cols = ['Investimento €', 'Risultato €', 'Costo Spread €']
    for c in cols:
        if c in df_view.columns:
            df_view[c] = pd.to_numeric(df_view[c].astype(str).str.replace('€','').str.replace(',','.'), errors='coerce').fillna(0)

    # Dashboard Metriche
    finished = df_view[df_view['Stato'].isin(['VINTO', 'PERSO', 'CHIUSO MAN.'])]
    if not finished.empty:
        net_pl = finished['Risultato €'].sum()
        win = len(finished[finished['Risultato €'] > 0])
        wr = (win / len(finished)) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Net Profit", f"{net_pl:.2f}€", delta_color="normal")
        c2.metric("Win Rate", f"{wr:.1f}%")
        c3.metric("Trades Chiusi", len(finished))

    # Styling Tabella
    st.dataframe(
        df_view[['DataOra', 'Asset', 'Direzione', 'Prezzo', 'TP', 'SL', 'Stato', 'Investimento €', 'Risultato €']].style
        .map(style_status, subset=['Stato'])
        .format({'Investimento €': '€ {:.2f}', 'Risultato €': '€ {:+.2f}'}),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Nessuna operazione registrata.")

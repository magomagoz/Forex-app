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
    """Carica la cronologia dal file CSV all'avvio"""
    if os.path.exists("permanent_signals_db.csv"):
        try:
            df = pd.read_csv("permanent_signals_db.csv")
            # Assicurati che le colonne siano corrette
            expected_cols = ['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato']
            for col in expected_cols:
                if col not in df.columns: df[col] = ""
            return df
        except:
            return pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
    return pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])

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

def play_close_sound():
    # Un suono più breve e "cash register" per le chiusure
    audio_html = """
        <audio autoplay><source src="https://assets.mixkit.co/active_storage/sfx/2017/2017-preview.mp3" type="audio/mpeg"></audio>
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
    
    updates_made = False
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            ticker = asset_map[row['Asset']]
            # Scarichiamo solo l'ultima candela per velocità
            data = yf.download(ticker, period="1d", interval="1m", progress=False)
            
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
                
                # Conversione sicura in float
                current_high = float(data['High'].iloc[-1])
                current_low = float(data['Low'].iloc[-1])
                
                # IMPORTANTE: Convertiamo i valori salvati (che sono stringhe) in float
                sl_v = float(str(row['SL']).replace(',', '.')) 
                tp_v = float(str(row['TP']).replace(',', '.'))
                
                new_status = None
                if row['Direzione'] == 'COMPRA':
                    if current_high >= tp_v: new_status = '✅ TARGET'
                    elif current_low <= sl_v: new_status = '❌ STOP LOSS'
                elif row['Direzione'] == 'VENDI':
                    if current_low <= tp_v: new_status = '✅ TARGET'
                    elif current_high >= sl_v: new_status = '❌ STOP LOSS'
                
                if new_status:
                    df.at[idx, 'Stato'] = new_status
                    updates_made = True
                    play_close_sound()
                    msg = f"🔔 **OPERAZIONE CHIUSA**\nAsset: {row['Asset']}\nEsito: {new_status}\nProfit/Loss: Controlla Dashboard"
                    send_telegram_msg(msg)
        except Exception as e:
            print(f"Errore update {row['Asset']}: {e}")
            continue 
        
    if updates_made:
        st.session_state['signal_history'] = df
        save_history_permanently()

def run_sentinel():
    """Scansiona tutti gli asset definiti in asset_map"""
    assets = list(asset_map.items())
    for label, ticker in assets:
        try:
            # 1. DOWNLOAD DATI
            df_rt_s = yf.download(ticker, period="2d", interval="1m", progress=False)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False)
            
            if df_rt_s.empty or df_d_s.empty: continue
            
            # Normalizzazione colonne (Fix per errori MultiIndex)
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
            atr_d = ta.atr(df_d_s['high'], df_d_s['low'], df_d_s['close'], length=14).iloc[-1]
            adx_df = ta.adx(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1]

            # 3. LOGICA SEGNALE (OTTIMIZZATA)
            s_action = None
            
            # DEBUG: Stampa i valori reali per capire perché non entrano segnali
            # print(f"Analisi {label}: Prezzo {curr_v:.5f} | BB_Low {low_bb:.5f} | RSI_D {rsi_d:.1f} | ADX {curr_adx:.1f}")

            # Limiti leggermente più ampi: RSI Daily (40-60) e ADX fino a 45
            if curr_v < low_bb and rsi_d < 60 and curr_adx < 45: 
                s_action = "COMPRA"
            elif curr_v > up_bb and rsi_d > 40 and curr_adx < 45: 
                s_action = "VENDI"

            # ... calcolo s_action ...
            
            # CORREZIONE QUI: Tutto questo blocco deve essere indentato sotto 'if s_action:'
            if s_action:
                hist = st.session_state['signal_history']
                
                # Controllo se esiste già un trade aperto per questo asset
                is_running = not hist.empty and ((hist['Asset'] == label) & (hist['Stato'] == 'In Corso')).any()
                
                if not is_running:
                    # Qui prendi i parametri. ATTENZIONE: balance e risk_pc sono variabili globali
                    # Assicurati che siano definite prima di chiamare run_sentinel
                    p_unit, p_fmt, p_mult, _ = get_asset_params(ticker) # Aggiunto a_type per evitare errore unpack
                    
                    sl = curr_v - (1.5 * atr_d) if s_action == "COMPRA" else curr_v + (1.5 * atr_d)
                    tp = curr_v + (3 * atr_d) if s_action == "COMPRA" else curr_v - (3 * atr_d)
                            
                    risk_val = balance * (risk_pc / 100)
                    dist_p = abs(curr_v - sl) * p_mult
                    
                    # Protezione divisione per zero
                    if dist_p == 0: dist_p = 0.0001
                    sz = risk_val / (dist_p * 10) # *10 è standard forex approx, verifica per crypto
                            
                    new_sig = {
                        'DataOra': get_now_rome().strftime("%H:%M:%S"),
                        'Asset': label, 
                        'Direzione': s_action, 
                        'Prezzo': p_fmt.format(curr_v), 
                        'TP': p_fmt.format(tp), 
                        'SL': p_fmt.format(sl), 
                        #'Size': f"{sz:.2f}", 
                        'Stato': 'In Corso',
                        'Investimento €': f"{risk_val:.2f}"
                    }
                        
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    save_history_permanently()
                    st.session_state['last_alert'] = new_sig
                    send_telegram_msg(f"🚀 *{s_action}* {label}\nPrezzo: {new_sig['Prezzo']}")
                    st.rerun()

            st.session_state['last_scan_status'] = f"🟢 {get_now_rome().strftime('%H:%M:%S')} - {label}: OK"
            
            # --- POSIZIONE DELLO SLEEP ---
            # Questo mette in pausa il bot per mezzo secondo prima di passare all'asset successivo
            time_lib.sleep(0.5) 
            
        except Exception as e:
            now = get_now_rome().strftime("%H:%M:%S")
            st.session_state['last_scan_status'] = f"🔴 {now} - {label}: Errore"
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

# --- 4. SIDEBAR ---
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
balance = st.sidebar.number_input("**Conto (€)**", value=1000)
risk_pc = st.sidebar.slider("**Investimento %**", 0.5, 5.0, 1.0, step=0.5)

# --- CALCOLO INVESTIMENTO SIMULATO ---
investimento_simulato = balance * (risk_pc / 100)
saldo_residuo = balance - investimento_simulato

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Gestione Capitale")
col_cap1, col_cap2 = st.sidebar.columns(2)
col_cap1.metric("Conto", f"€ {balance:.2f}")
col_cap2.metric("Investimento", f"€ {investimento_simulato:.2f}")

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

# --- 5. POPUP ALERT (FIXED RENDERING) ---
if st.session_state['last_alert']:
    play_notification_sound()
    alert = st.session_state['last_alert']
    main_color = "#00ffcc" if alert['Direzione'] == 'COMPRA' else "#ff4b4b"

    # 1. CSS separato per evitare conflitti di parsing
    st.markdown(f"""
        <style>
            .full-screen-overlay {{
                position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(0, 0, 0, 0.85); z-index: 9990; backdrop-filter: blur(5px);
            }}
            .popup-card {{
                position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                width: 90%; max-width: 500px; background: #111;
                border: 3px solid {main_color}; border-radius: 20px;
                padding: 40px; text-align: center; z-index: 9995;
                box-shadow: 0 0 60px {main_color}44; color: white;
            }}
            div.stButton > button {{
                position: fixed !important; bottom: 15% !important; left: 50% !important;
                transform: translateX(-50%) !important; z-index: 10000 !important;
                width: 280px !important; background-color: {main_color} !important;
                color: black !important; font-weight: bold !important;
                border-radius: 10px !important; border: none !important;
            }}
        </style>
    """, unsafe_allow_html=True)

    # 2. HTML pulito (usiamo stringhe semplici senza troppe variabili annidate)
    html_content = f"""
        <div class="full-screen-overlay"></div>
        <div class="popup-card">
            <div style="letter-spacing:3px; color:{main_color}; font-weight:bold;">AI SIGNAL DETECTED</div>
            <div style="font-size: 4em; font-weight: 800; margin: 15px 0;">{alert['Asset']}</div>
            <div style="background:{main_color}; color:black; padding:10px; border-radius:10px; font-weight:900; font-size:1.5em;">
                🚀 {alert['Direzione']}
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:30px; border-top:1px solid #333; padding-top:20px;">
                <div style="text-align:left;">
                    <div style="color:#aaa; font-size:0.8em;">ENTRY PRICE</div>
                    <div style="font-size:1.5em;">{alert['Prezzo']}</div>
                </div>

                <div style="text-align:center;">
                    <div style="color:#aaa; font-size:0.8em;">TARGET PRICE</div>
                    <div style="font-size:1.5em; color:{main_color};">{alert['TP']}</div>
                </div>

                <div style="text-align:right;">
                    <div style="color:#aaa; font-size:0.8em;">STOP LOSS</div>
                    <div style="font-size:1.5em;">{alert['SL']}</div>
                </div>
            
            </div>
        </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

    # 3. Il tasto fisico
    if st.button("❌ CHIUDI MONITOR", key="close_alert_btn"):
        st.session_state['last_alert'] = None
        st.rerun()
    
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
    
    # --- FILTRI RAPIDI ---
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filtro = st.selectbox("🔍 Filtra per stato:", ["Tutti", "In Corso", "✅ TARGET", "❌ STOP LOSS"])
    
    if filtro != "Tutti":
        display_df = display_df[display_df['Stato'] == filtro]
    
    def style_status(val):
        if '✅' in val: return 'background-color: rgba(0, 255, 204, 0.1); color: #00ffcc; font-weight: bold'
        if '❌' in val: return 'background-color: rgba(255, 75, 75, 0.1); color: #ff4b4b; font-weight: bold'
        return 'color: #ffcc00; font-weight: bold'

    #st.dataframe(display_df, hide_index=True)
    
    st.dataframe(
        display_df.style.map(style_status, subset=['Stato']), 
        use_container_width=True,
        hide_index=True
    )
    
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Salva Cronologia", csv, "cronologia_forex.csv", "text/csv")
else:
    st.write("Nessun segnale rilevato finora. In attesa di opportunità...")

st.markdown("---")

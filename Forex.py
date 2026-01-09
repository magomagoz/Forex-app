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

# --- 1. CONFIGURAZIONE & REFRESH ---
st.set_page_config(page_title="Forex Momentum Pro AI", layout="wide", page_icon="📈")
st.markdown("""
    <style>
        .block-container {padding-top: 1rem !important;}
        [data-testid="stSidebar"] > div: first-child {padding-top: 0rem :important;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

#Definizione Fuso Orario Roma
rome_tz = pytz.timezone('Europe/Rome')

#Refresh automatico ogni 60 secondi
st_autorefresh(interval=60 * 1000, key="sentinel_refresh")

if 'signal_history' not in st.session_state: 
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
if 'last_alert' not in st.session_state:
    st.session_state['last_alert'] = None

# --- 2. FUNZIONI TECNICHE ---
def get_now_rome():
    return datetime.now(rome_tz)

def play_notification_sound():
    audio_html = """
        <audio autoplay>
            <source src="https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3" type="audio/mpeg">
        </audio>
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
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except: return None

def get_currency_strength():
    try:
        forex = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X", "EURCHF=X","EURJPY=X", "GBPJPY=X", "USDCAD=X", "GBPCHF=X","EURGBP=X"]
        crypto = ["BTC-USD", "ETH-USD"]
        data = yf.download(forex + crypto, period="2d", interval="1d", progress=False, timeout=15)
        if data is None or data.empty: return pd.Series(dtype=float)
        close_data = data['Close'] if 'Close' in data else data
        returns = close_data.pct_change().iloc[-1] * 100
        strength = {
            "USD 🇺🇸": (-returns.get("EURUSD=X",0) - returns.get("GBPUSD=X",0) + returns.get("USDJPY=X",0) - returns.get("AUDUSD=X",0) + returns.get("USDCAD=X",0) + returns.get("USDCHF=X",0) - returns.get("NZDUSD=X",0)) / 7,
            "EUR 🇪🇺": (returns.get("EURUSD=X",0) + returns.get("EURJPY=X",0) + returns.get("EURGBP=X",0)) / 3,
            "GBP 🇬🇧": (returns.get("GBPUSD=X",0) + returns.get("GBPJPY=X",0) - returns.get("EURGBP=X",0)) / 3,
            "JPY 🇯🇵": (-returns.get("USDJPY=X",0) - returns.get("EURJPY=X",0) - returns.get("GBPJPY=X",0)) / 3,
            "CHF 🇨🇭": (-returns.get("USDCHF=X",0) - returns.get("EURCHF=X",0) - returns.get("GBPCHF=X",0)) / 3,
            "AUD 🇦🇺": returns.get("AUDUSD=X", 0),
            "CAD 🇨🇦": -returns.get("USDCAD=X", 0),
            "BTC ₿": returns.get("BTC-USD", 0),
            "ETH 💎": returns.get("ETH-USD", 0)
        }
        return pd.Series(strength).sort_values(ascending=False)
    except: return pd.Series(dtype=float)

def get_asset_params(pair):
    if "-" in pair: return 1.0, "{:.2f}", 1, "CRYPTO"
    if "JPY" in pair: return 0.01, "{:.2f}", 1000, "FOREX" 
    return 0.0001, "{:.5f}", 10, "FOREX"

def detect_divergence(df):
    if len(df) < 20: return "Analisi in corso"
    price, rsi_col = df['close'], df['rsi']
    curr_p, curr_r = float(price.iloc[-1]), float(rsi_col.iloc[-1])
    prev_max_p, prev_max_r = price.iloc[-20:-1].max(), rsi_col.iloc[-20:-1].max()
    prev_min_p, prev_min_r = price.iloc[-20:-1].min(), rsi_col.iloc[-20:-1].min()
    if curr_p > prev_max_p and curr_r < prev_max_r: return "📉 DECRESCITA"
    elif curr_p < prev_min_p and curr_r > prev_min_r: return "📈 CRESCITA"
    return "Neutrale"

# --- 3. MOTORI DI BACKGROUND ---
def update_signal_outcomes():
    if st.session_state['signal_history'].empty: return
    df = st.session_state['signal_history']
    for idx, row in df[df['Stato'] == 'In Corso'].iterrows():
        try:
            data = yf.download(asset_map[row['Asset']], period="5d", interval="1m", progress=False)
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
            # Monitoraggio rapido su base 1 minuto
            df_rt_s = yf.download(ticker, period="5d", interval="1m", progress=False)
            df_d_s = yf.download(ticker, period="1y", interval="1d", progress=False)
            if df_rt_s.empty or df_d_s.empty: continue
            if isinstance(df_rt_s.columns, pd.MultiIndex): df_rt_s.columns = df_rt_s.columns.get_level_values(0)
            if isinstance(df_d_s.columns, pd.MultiIndex): df_d_s.columns = df_d_s.columns.get_level_values(0)
            df_rt_s.columns = [c.lower() for c in df_rt_s.columns]
            df_d_s.columns = [c.lower() for c in df_d_s.columns]

            # Indicatori: Bande, RSI e ADX
            bb_s = ta.bbands(df_rt_s['close'], length=20, std=2)
            rsi_s = ta.rsi(df_d_s['close'], length=14).iloc[-1]
            atr_s = ta.atr(df_d_s['high'], df_d_s['low'], df_d_s['close'], length=14).iloc[-1]
            adx_df = ta.adx(df_rt_s['high'], df_rt_s['low'], df_rt_s['close'], length=14)
            curr_adx = adx_df['ADX_14'].iloc[-1]
          
            # Nomi colonne BB
            c_l = [c for c in bb_s.columns if "BBL" in c.upper()][0]
            c_m = [c for c in bb_s.columns if "BBM" in c.upper()][0]
            c_u = [c for c in bb_s.columns if "BBU" in c.upper()][0]

            c_v = float(df_rt_s['close'].iloc[-1])
            l_bb = float(bb_s[c_l].iloc[-1]) # Usa il nome colonna trovato
            m_bb = float(bb_s[c_m].iloc[-1])
            u_bb = float(bb_s[c_u].iloc[-1])
            
            # Logica Segnale: aggiunto filtro ADX < 30 (evita trend esplosivi)
            s_action = None
            if c_v < l_bb and rsi_s < 45 and curr_adx < 30: s_action = "COMPRA"
            elif c_v > u_bb and rsi_s > 55 and curr_adx < 30: s_action = "VENDI"
            
            if s_action:
                hist = st.session_state['signal_history']
                # Evita duplicati nello stesso minuto
                if hist.empty or not ((hist['Asset'] == label) & (hist['Direzione'] == s_action)).head(1).any():
                    p_unit, p_fmt, p_mult = get_asset_params(ticker)[:3]
                    atr_s = ta.atr(df_d_s['high'], df_d_s['low'], df_d_s['close'], length=14).iloc[-1]
                    sl = c_v - (1.5 * atr_s) if s_action == "COMPRA" else c_v + (1.5 * atr_s)
                    tp = c_v + (3 * atr_s) if s_action == "COMPRA" else c_v - (3 * atr_s)
                    risk_val = balance * (risk_pc / 100)
                    dist_p = abs(c_v - sl) * p_mult
                    sz = risk_val / (dist_p * 10) if dist_p > 0 else 0
                    
                    new_sig = {'DataOra': get_now_rome().strftime("%d/%m/%Y %H:%M:%S"), 'Asset': label, 'Direzione': s_action, 'Prezzo': p_fmt.format(c_v), 'SL': p_fmt.format(sl), 'TP': p_fmt.format(tp), 'Size': f"{sz:.2f}", 'Stato': 'In Corso'}
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_sig]), hist], ignore_index=True)
                    st.session_state['last_alert'] = new_sig
                    st.rerun()

            # Prepariamo il messaggio di log predefinito
            scan_detail = "OK"
            if curr_adx > 30:
                scan_detail = f"Trend Forte (ADX:{curr_adx:.0f})"
            elif rsi_s > 45 and rsi_s < 55:
                scan_detail = f"RSI Neutro ({rsi_s:.0f})"

            if s_action:
                # Se c'è un'azione, il log mostrerà l'azione
                st.session_state['last_scan_status'] = f"🚀 {label}: SEGNALE {s_action}!"
            else:
                # Altrimenti mostra il motivo del "no signal"
                st.session_state['last_scan_status'] = f"📡 {label}: {scan_detail}"

        except Exception as e:
            st.session_state['last_scan_status'] = f"⚠️ Errore su {label}"
            continue
                
# --- 4. SIDEBAR CON TIMER E SESSIONI ---
st.sidebar.header("🛠 Trading Desk (1m)")
if "start_time" not in st.session_state: st.session_state.start_time = time_lib.time()
countdown = 60 - int(time_lib.time() - st.session_state.start_time) % 60
st.sidebar.markdown(f"⏳ **Prossimo Scan: {countdown}s**")

st.sidebar.subheader("📡 Sentinel Status")
status = st.session_state.get('last_scan_status', 'In attesa...')

# Usiamo SOLO il box colorato, rimuovi st.sidebar.code(status)
if "SEGNALE" in status:
    st.sidebar.success(status)
elif "Trend Forte" in status or "⚠️" in status:
    st.sidebar.warning(status)
else:
    st.sidebar.info(status)
         
asset_map = {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X", "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD"}
selected_label = st.sidebar.selectbox("**Asset**", list(asset_map.keys()))
pair = asset_map[selected_label]
balance = st.sidebar.number_input("**Conto (€)**", value=1000)
risk_pc = st.sidebar.slider("**Rischio %**", 0.5, 5.0, 1.0)

st.sidebar.subheader("🌍 Sessioni di Mercato")
for s_name, is_open in get_session_status().items():
    color = "🟢" if is_open else "🔴"
    status_text = "OPEN" if is_open else "CLOSED"
    st.sidebar.markdown(f"{color} **{s_name}**: {status_text}")

if st.sidebar.button("🗑️ Reset Cronologia"):
    st.session_state['signal_history'] = pd.DataFrame(columns=['DataOra', 'Asset', 'Direzione', 'Prezzo', 'SL', 'TP', 'Size', 'Stato'])
    st.rerun()

# --- SPOSTA QUI (subito dopo la sidebar) ---
update_signal_outcomes()
run_sentinel()

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

# --- 6. HEADER E GRAFICO AVANZATO (FIXED) ---
st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1><p style="color: white; opacity: 0.8; margin:0;">Sentinel AI Engine • Forex & Crypto Analysis</p></div>', unsafe_allow_html=True)

p_unit, price_fmt, p_mult, a_type = get_asset_params(pair)
df_rt = get_realtime_data(pair) 
df_d = yf.download(pair, period="1y", interval="1d", progress=False) # Daily per analisi

if df_rt is not None and not df_rt.empty:
    # Calcolo indicatori
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    df_rt['rsi'] = ta.rsi(df_rt['close'], length=14)
    
    c_u = [c for c in bb.columns if "BBU" in c.upper()][0]
    c_m = [c for c in bb.columns if "BBM" in c.upper()][0]
    c_l = [c for c in bb.columns if "BBL" in c.upper()][0]
    
    p_df = df_rt.tail(60)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # --- RIGA 1: PREZZO E BANDE ---
    # Candele
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['open'], high=p_df['high'], 
                                 low=p_df['low'], close=p_df['close'], name='Prezzo'), row=1, col=1)
    # Bande Bollinger
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_u], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), name='Upper BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_m], line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name='Middle BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_l], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), fill='tonexty', name='Lower BB'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['rsi'], line=dict(color='#ffcc00', width=2), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#00ff00", row=2, col=1)

    # --- AGGIUNTA LINEE VERTICALI OGNI 30 MINUTI ---
    # Prendiamo l'intervallo temporale del dataframe visualizzato (p_df)
    start_time = p_df.index.min()
    end_time = p_df.index.max()
    
    # Generiamo i marker ogni 5 minuti
    # '5T' sta per 5 minuti in Pandas
    v_lines = pd.date_range(start=start_time, end=end_time, freq='5T')
    
    for line_time in v_lines:
        fig.add_vline(
            x=line_time, 
            line_width=1, 
            #line_dash="dash", 
            line_color="rgba(0, 0, 0, 0.2)", # Nero
            row="all", # Applica a entrambi i sottografici (Prezzo e RSI)
            col=1
        )

    # Layout finale (questo lo hai già, assicurati che fig.update_layout segua il ciclo for)
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, 
                      margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.02))

    # --- AGGIUNTA FRECCE SEGNALI ---
    if not st.session_state['signal_history'].empty:
        asset_signals = st.session_state['signal_history'][st.session_state['signal_history']['Asset'] == selected_label]
        for _, sig in asset_signals.iterrows():
            try:
                sig_time = pd.to_datetime(sig['DataOra'], dayfirst=True).floor('1min')
                # Cerchiamo il match più vicino nel grafico
                nearest_idx = p_df.index.get_indexer([sig_time], method='nearest')[0]
                actual_time = p_df.index[nearest_idx]
                
                y_pos = p_df.loc[actual_time, 'low'] if sig['Direzione'] == 'COMPRA' else p_df.loc[actual_time, 'high']
                color = "#00ffcc" if sig['Direzione'] == 'COMPRA' else "#ff4b4b"
                fig.add_annotation(
                    x=actual_time, y=y_pos, text=sig['Direzione'], showarrow=True, arrowhead=2,
                    arrowcolor=color, ax=0, ay=40 if sig['Direzione'] == 'COMPRA' else -40,
                    font=dict(color=color), bgcolor="rgba(0,0,0,0.8)", row=1, col=1
                )
            except: continue

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0,r=0,t=30,b=40))
    
    # SOLO ORA RENDERIZZIAMO IL GRAFICO
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 7. CURRENCY STRENGTH ---
    st.markdown("---")
    st.subheader("⚡ Currency Strength Meter")
    s_data = get_currency_strength()
    if not s_data.empty:
        # Creiamo esattamente il numero di colonne pari ai dati ricevuti
        cols = st.columns(len(s_data))
        for i, (curr, val) in enumerate(s_data.items()):
            bg = "#006400" if val > 0.15 else "#8B0000" if val < -0.15 else "#333333"
            txt_c = "#00FFCC" if val > 0.15 else "#FF4B4B" if val < -0.15 else "#FFFFFF"
            # Usiamo cols[i] in modo sicuro
            cols[i].markdown(f"<div style='text-align:center; background:{bg}; padding:6px; border-radius:8px; border:1px solid {txt_c}; min-height:80px;'><b style='color:white; font-size:0.8em;'>{curr}</b><br><span style='color:{txt_c};'>{val:.2f}%</span></div>", unsafe_allow_html=True)

# --- 8. ANALISI AI (CORRETTA) ---
if df_rt is not None and not df_rt.empty and df_d is not None and not df_d.empty:
    # Ricalcoliamo o recuperiamo i nomi delle colonne per sicurezza
    # in caso lo script saltasse la parte grafica
    try:
        c_u_ai = [c for c in df_rt.columns if "BBU" in c.upper()][0]
        c_l_ai = [c for c in df_rt.columns if "BBL" in c.upper()][0]
        
        if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
        df_d.columns = [c.lower() for c in df_d.columns]
        df_d['rsi'] = ta.rsi(df_d['close'], length=14) 
        df_d['atr'] = ta.atr(df_d['high'], df_d['low'], df_d['close'], length=14)
        
        rsi_val = float(df_d['rsi'].iloc[-1])
        last_atr = float(df_d['atr'].iloc[-1])
        curr_p = float(df_rt['close'].iloc[-1])
        
        # Calcolo Score
        score = 50
        if curr_p < df_rt[c_l_ai].iloc[-1]: score += 20
        elif curr_p > df_rt[c_u_ai].iloc[-1]: score -= 20
        
        if not is_low_liquidity():
            action = "COMPRA" if (score >= 65 and rsi_val < 60) else "VENDI" if (score <= 35 and rsi_val > 40) else None
            if action:
                hist = st.session_state['signal_history']
                if hist.empty or not ((hist['Asset'] == selected_label) & (hist['Direzione'] == action)).head(1).any():
                    sl = curr_p - (1.5 * last_atr) if action == "COMPRA" else curr_p + (1.5 * last_atr)
                    tp = curr_p + (3 * last_atr) if action == "COMPRA" else curr_p - (3 * last_atr)
                    dist_p = abs(curr_p - sl) * p_mult
                    size = (balance * (risk_pc / 100)) / (dist_p * 10) if dist_p > 0 else 0
                    new_a = {'DataOra': get_now_rome().strftime("%d/%m/%Y %H:%M:%S"), 'Asset': selected_label, 'Direzione': action, 'Prezzo': price_fmt.format(curr_p), 'SL': price_fmt.format(sl), 'TP': price_fmt.format(tp), 'Size': f"{size:.2f}", 'Stato': 'In Corso'}
                    st.session_state['signal_history'] = pd.concat([pd.DataFrame([new_a]), hist], ignore_index=True)
                    st.session_state['last_alert'] = new_a
                    st.rerun()

        # Visualizzazione Metriche
        adx_df_ai = ta.adx(df_rt['high'], df_rt['low'], df_rt['close'], length=14)
        curr_adx_ai = adx_df_ai['ADX_14'].iloc[-1]

        st.markdown("---")
        st.subheader("🕵️ Sentinel Market Analysis")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("RSI Daily", f"{rsi_val:.1f}", detect_divergence(df_d))
        col_b.metric("Sentinel Score", f"{score}/100")
        adx_emoji = "🔴" if curr_adx_ai > 30 else "🟡" if curr_adx_ai > 20 else "🟢"
        col_c.metric("Forza Trend (ADX)", f"{curr_adx_ai:.1f}", adx_emoji)
    except Exception as e:
        st.error(f"Errore nell'analisi AI: {e}")

# --- 9. CRONOLOGIA SEGNALI (CORRETTA) ---
if not st.session_state['signal_history'].empty:
    def style_status(val):
        if not isinstance(val, str): return ''
        color = '#00ffcc' if '✅' in val else '#ff4b4b' if '❌' in val else '#ffcc00'
        return f'color: {color}; font-weight: bold'
    
    # Usiamo .map invece di .applymap per compatibilità con Pandas 2.x+
    styled_df = st.session_state['signal_history'].style.map(style_status, subset=['Stato'])
    st.dataframe(styled_df, use_container_width=True)

st.info(f"🛰️ **Sentinel AI**: Monitoraggio attivo su {len(asset_map)} asset.")

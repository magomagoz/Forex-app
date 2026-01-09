# --- 6. HEADER E GRAFICO AVANZATO (Con RSI) ---
st.markdown('<div style="background: linear-gradient(90deg, #0f0c29, #302b63, #24243e); padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #00ffcc;"><h1 style="color: #00ffcc; margin: 0;">📊 FOREX MOMENTUM PRO AI</h1><p style="color: white; opacity: 0.8; margin:0;">Sentinel AI Engine • Forex & Crypto Analysis</p></div>', unsafe_allow_html=True)

p_unit, price_fmt, p_mult, a_type = get_asset_params(pair)
# Nota: qui assume che tu abbia fatto la modifica "5m" suggerita prima.
# Se non l'hai fatta, df_rt sarà a 1m, se l'hai fatta sarà a 5m. Funziona in entrambi i casi.
df_rt = get_realtime_data(pair) 
df_d = yf.download(pair, period="1y", interval="5d", progress=False)

if df_rt is not None and not df_rt.empty:
    # Calcolo indicatori per il grafico
    bb = ta.bbands(df_rt['close'], length=20, std=2)
    df_rt = pd.concat([df_rt, bb], axis=1)
    df_rt['rsi'] = ta.rsi(df_rt['close'], length=14) # Calcolo RSI per il grafico
    
    # Nomi colonne bande
    c_up = [c for c in df_rt.columns if "BBU" in c.upper()][0]
    c_mid = [c for c in df_rt.columns if "BBM" in c.upper()][0]
    c_low = [c for c in df_rt.columns if "BBL" in c.upper()][0]
    
    st.subheader(f"📈 Chart 5m: {selected_label}")
    
    # Prepariamo gli ultimi 60 periodi per la visualizzazione
    p_df = df_rt.tail(60)
    
    # Creazione sottografici (2 righe: Prezzo sopra, RSI sotto)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])

    # --- RIGA 1: PREZZO E BANDE ---
    # Candele
    fig.add_trace(go.Candlestick(x=p_df.index, open=p_df['open'], high=p_df['high'], 
                                 low=p_df['low'], close=p_df['close'], name='Prezzo'), row=1, col=1)
    # Bande Bollinger
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_up], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), name='Upper BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_mid], line=dict(color='rgba(255, 255, 255, 0.3)', width=1), name='Middle BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df[c_low], line=dict(color='rgba(173, 216, 230, 0.4)', width=1), fill='tonexty', name='Lower BB'), row=1, col=1)

    # --- RIGA 2: RSI ---
    fig.add_trace(go.Scatter(x=p_df.index, y=p_df['rsi'], line=dict(color='#ffcc00', width=2), name='RSI'), row=2, col=1)
    # --- MIGLIORAMENTO VISIVO RSI ---
    #fig.add_trace(go.Scatter(x=p_df.index, y=p_df['rsi'], line=dict(color='#ffcc00', width=2), name='RSI'), row=2, col=1)
    # Zone di Ipercomprato/Ipervenduto evidenziate
    #fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0, row=2, col=1)
    #fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0, row=2, col=1)
    # fine blocco miglioramento RSI


    # Linee RSI (70 e 30)
    fig.add_hline(y=75, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=25, line_dash="dot", line_color="#00ff00", row=2, col=1)
    fig.add_hrect(y0=25, y1=75, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)

    
    # Visualizzazione Win Rate in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏆 Performance Oggi")
    wr = get_win_rate()
    if wr:
        st.sidebar.info(wr)

    
    # Layout finale
    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, 
                      margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=1.02))

    st.plotly_chart(fig, use_container_width=True)
    
    curr_p = float(df_rt['close'].iloc[-1])
    # Mostriamo anche il valore attuale dell'RSI accanto al prezzo
    curr_rsi = float(df_rt['rsi'].iloc[-1])
    
    c_met1, c_met2 = st.columns(2)
    c_met1.metric(f"Prezzo {selected_label}", price_fmt.format(curr_p))
    c_met2.metric(f"RSI (5m)", f"{curr_rsi:.1f}", delta="IPERCOMPRATO" if curr_rsi > 70 else "IPERVENDUTO" if curr_rsi < 30 else "NEUTRO")

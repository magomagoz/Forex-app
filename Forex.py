import yfinance as yf
import pandas as pd

def get_clean_data(ticker, period="2y", interval="1d"):
    try:
        # Scarica i dati
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            st.error(f"Nessun dato trovato per {ticker}. Verifica la connessione o il simbolo.")
            return None

        # Fix per il Multi-index di yfinance (nuova versione)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Pulizia: rimuove eventuali righe con valori nulli iniziali
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Errore tecnico durante il download: {e}")
        return None

# Utilizzo nel tuo programma Streamlit
data = get_clean_data("EURUSD=X")

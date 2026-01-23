from iqoptionapi.stable_api import IQ_Option
import time

print("Tentativo di connessione...")
# Inserisci qui i tuoi dati reali per il test rapido
api = IQ_Option("TUA_EMAIL", "TUA_PASSWORD")
check, reason = api.connect()

if check:
    print("✅ Connesso con successo!")
    api.change_balance("PRACTICE")
    saldo = api.get_balance()
    print(f"💰 Saldo Conto Practice: {saldo}€")
    
    # Prova a leggere l'ultimo prezzo di EURUSD
    print("Lettura prezzo EURUSD...")
    api.subscribe_strike_list("EURUSD", 1)
    time.sleep(2)
    print(f"Prezzo attuale: {api.get_realtime_candle('EURUSD', 1)}")
    
    api.disconnect()
else:
    print(f"❌ Errore: {reason}")

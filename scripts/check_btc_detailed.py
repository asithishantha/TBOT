
import pandas as pd
import talib as ta
from pathlib import Path

def check_btc_more():
    for tf in ['1d', '4h', '1h']:
        csv_file = Path(f"data/raw/BTCUSDT_{tf}.csv")
        if not csv_file.exists():
            print(f"{tf} CSV not found")
            continue
        
        df = pd.read_csv(csv_file)
        # Find timestamp column
        date_col = None
        for col in ['date', 'timestamp', 'time', 'datetime']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
            df = df.dropna(subset=[date_col])
            df.sort_values(date_col, inplace=True)
            df.set_index(date_col, inplace=True)
        
        close = df['close'].iloc[-1]
        ema_200 = ta.EMA(df['close'], timeperiod=200).iloc[-1]
        ema_50 = ta.EMA(df['close'], timeperiod=50).iloc[-1]
        
        print(f"--- {tf.upper()} ---")
        print(f"Close: {close}")
        print(f"EMA 200: {ema_200}")
        print(f"EMA 50: {ema_50}")
        print(f"Above EMA 200: {close > ema_200}")
        print(f"Above EMA 50: {close > ema_50}")

check_btc_more()

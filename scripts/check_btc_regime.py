
import pandas as pd
import talib as ta
from pathlib import Path

def calculate_btc_ema():
    csv_file = Path("data/raw/BTCUSDT_1d.csv")
    if not csv_file.exists():
        print("CSV not found")
        return
    
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
    
    ema_200 = ta.EMA(df['close'], timeperiod=200)
    
    latest_close = df['close'].iloc[-1]
    latest_ema = ema_200.iloc[-1]
    
    slope_lookback = 20
    ema_slope = (ema_200.iloc[-1] - ema_200.iloc[-slope_lookback]) / ema_200.iloc[-slope_lookback]
    
    print(f"Latest Close: {latest_close}")
    print(f"Latest 1D 200 EMA: {latest_ema}")
    print(f"EMA Slope (20d): {ema_slope}")
    
    is_bullish = (latest_close > latest_ema) and (ema_slope > 0.0005)
    is_bearish = (latest_close < latest_ema) and (ema_slope < -0.0005)
    
    print(f"Is Bullish (Macro): {is_bullish}")
    print(f"Is Bearish (Macro): {is_bearish}")
    
    # Check 4H too
    csv_file_4h = Path("data/raw/BTCUSDT_4h.csv")
    if csv_file_4h.exists():
        df_4h = pd.read_csv(csv_file_4h)
        # Find timestamp column
        date_col = None
        for col in ['date', 'timestamp', 'time', 'datetime']:
            if col in df_4h.columns:
                date_col = col
                break
        if date_col:
            df_4h[date_col] = pd.to_datetime(df_4h[date_col], utc=True, errors='coerce')
            df_4h = df_4h.dropna(subset=[date_col])
            df_4h.sort_values(date_col, inplace=True)
            df_4h.set_index(date_col, inplace=True)
            
            ema_200_4h = ta.EMA(df_4h['close'], timeperiod=200).iloc[-1]
            print(f"Latest 4H 200 EMA: {ema_200_4h}")
            print(f"Price vs 4H 200: {'Above' if df_4h['close'].iloc[-1] > ema_200_4h else 'Below'}")

calculate_btc_ema()

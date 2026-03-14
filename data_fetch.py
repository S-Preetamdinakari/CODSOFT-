import yfinance as yf
import pandas as pd

CRYPTO_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "BNB": "BNB-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "DOT": "DOT-USD",
    "LTC": "LTC-USD",
    "TRX": "TRX-USD"
}

def fetch_crypto_data(symbol, period="3y"):
    ticker = yf.Ticker(CRYPTO_MAP[symbol])
    df = ticker.history(period=period)

    # 🔥 RESET INDEX & REMOVE TIMEZONE (CRITICAL FIX)
    df.reset_index(inplace=True)
    # Handle timezone-aware and timezone-naive datetimes properly
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        df["Date"] = pd.to_datetime(df["Date"])

    return df

def fetch_live_price(symbol):
    """Fetch the current live price for a cryptocurrency"""
    try:
        ticker = yf.Ticker(CRYPTO_MAP[symbol])
        data = ticker.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
        else:
            return None
    except Exception as e:
        print(f"Error fetching live price: {e}")
        return None

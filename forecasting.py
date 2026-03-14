import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


# ---------------- ARIMA FORECAST ----------------
def arima_forecast(df, steps=30):
    try:
        # Ensure we have enough data
        if len(df) < 50:
            raise ValueError("Not enough data points for ARIMA")
        
        data = df["Close"].dropna().astype(float)
        
        # Use a simpler ARIMA order if default fails
        try:
            model = ARIMA(data, order=(1, 1, 1))
            model_fit = model.fit()
        except:
            # Fallback to simpler model
            model = ARIMA(data, order=(1, 1, 0))
            model_fit = model.fit()
        
        forecast_values = model_fit.forecast(steps=steps)
        
        # Create a date range for the forecast
        last_date = df['Date'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # Create a new Series with the date index
        forecast = pd.Series(forecast_values.values, index=forecast_dates)
        
        return forecast
    except Exception as e:
        print(f"ARIMA Error: {e}")
        raise


# ---------------- PROPHET FORECAST ----------------
def prophet_forecast(df, days=30):
    try:
        # Ensure we have enough data
        if len(df) < 50:
            raise ValueError("Not enough data points for Prophet")
        
        p_df = df[["Date", "Close"]].copy()
        p_df = p_df.rename(columns={"Date": "ds", "Close": "y"})
        p_df['ds'] = pd.to_datetime(p_df['ds'])
        p_df = p_df.sort_values('ds').reset_index(drop=True)
        p_df = p_df.dropna()
        
        if len(p_df) < 50:
            raise ValueError("Not enough valid data points for Prophet")

        model = Prophet(yearly_seasonality=False, interval_width=0.95)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(p_df)
        
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        return forecast
    except Exception as e:
        print(f"Prophet Error: {e}")
        raise

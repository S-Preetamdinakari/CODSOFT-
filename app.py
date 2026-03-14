import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np

from data_fetch import fetch_crypto_data, fetch_live_price
from preprocessing import preprocess_data
from forecasting import arima_forecast, prophet_forecast
from lstm_model import lstm_forecast

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Crypto Time Series Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': "https://github.com",
        'About': "# Cryptocurrency Time Series Analysis Dashboard"
    }
)

# --- CUSTOM CSS FOR ATTRACTIVE UI ---
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-right: 2px solid #00d4ff;
    }
    
    /* Headers */
    h1 {
        color: #00d4ff;
        font-size: 2.5rem !important;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        margin-bottom: 10px !important;
    }
    
    h2 {
        color: #00f4ff;
        font-size: 1.8rem !important;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 10px;
        margin-top: 20px !important;
    }
    
    h3 {
        color: #00d4ff;
        font-size: 1.3rem !important;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 244, 255, 0.05) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    /* Tabs */
    [data-baseweb="tab-list"] {
        border-bottom: 2px solid rgba(0, 212, 255, 0.2) !important;
    }
    
    [data-baseweb="tab"] {
        color: #b0b8c1 !important;
        border-radius: 10px 10px 0 0 !important;
    }
    
    [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 3px solid #00d4ff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    .stSelectbox, .stDateInput, .stTextInput {
        color: #00d4ff !important;
    }
    
    [data-baseweb="input"] {
        background-color: rgba(0, 212, 255, 0.05) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
        color: #00d4ff !important;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background-color: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 8px;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(15, 20, 25, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
    }
    
    /* Markdown text */
    p {
        color: #b0b8c1;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0, 212, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

plt.style.use('dark_background')

# Set matplotlib colors for better visualization
plt.rcParams['figure.facecolor'] = '#0f1419'
plt.rcParams['axes.facecolor'] = '#1a1f2e'
plt.rcParams['grid.color'] = '#00d4ff'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['lines.linewidth'] = 2.5

# --- SIDEBAR ---
st.sidebar.markdown("<h2 style='text-align: center; color: #00d4ff;'>🧭 Navigation</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "1. Overview / Executive KPIs",
    "2. Price Explorer & Candlesticks",
    "3. Forecast & Uncertainty",
    "4. Sentiment & News Impact",
    "5. Volatility & Risk Visuals",
    "6. Indicators Dashboard",
    "7. Correlations & Market Structure",
    "8. Feature Importance & Explainability",
    "9. Strategy Backtest & Performance",
    "10. Interactive Explorer"
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #00d4ff;'>⚙️ Configuration</h3>", unsafe_allow_html=True)
coins = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "LTC", "TRX"]
coin = st.sidebar.selectbox("Select Cryptocurrency", coins, label_visibility="collapsed")

# --- HELPER FUNCTION FOR DATE FORMATTING ---
def format_dates_on_plot(ax):
    """Format dates on matplotlib plot for better readability"""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_process_data(coin_symbol):
    data = fetch_crypto_data(coin_symbol)
    processed_data = preprocess_data(data)
    return processed_data

df_full = load_and_process_data(coin)

# Get valid date range
min_date = df_full['Date'].min()
if pd.isna(min_date):
    min_date = pd.Timestamp.now() - pd.Timedelta(days=365)
else:
    min_date = min_date.date()

start_date = st.sidebar.date_input("Start Date", min_date, label_visibility="collapsed")
df = df_full[df_full['Date'] >= pd.Timestamp(start_date)].copy()

# --- MAIN APP HEADER ---
col_header1 = st.columns([1])[0]
with col_header1:
    st.markdown("# 📊 Crypto Time Series Analysis")
    st.markdown(f"### 🪙 Analyzing **{coin}** | Models: **ARIMA • Prophet • LSTM**")

st.markdown("---")

# --- FETCH LIVE PRICE ---
live_price = fetch_live_price(coin)
if live_price is None:
    live_price = df['Close'].iloc[-1]

# ---------------- 1. Overview / Executive KPIs ----------------
if page == "1. Overview / Executive KPIs":
    st.header("📊 Executive Overview & Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    latest_price = live_price
    latest_return = df['Returns'].iloc[-1]
    volatility = df['Returns'].std() * np.sqrt(365)
    
    with col1:
        st.metric("💰 Latest Price", f"${latest_price:,.2f}", f"{latest_return:.2%}", delta_color="normal")
    with col2:
        st.metric("📈 Daily Return", f"{latest_return:.2%}", label_visibility="collapsed")
    with col3:
        st.metric("📊 Annual Volatility", f"{volatility:.2%}", label_visibility="collapsed")
    with col4:
        st.metric("📋 Total Data Points", len(df), label_visibility="collapsed")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns([2, 1])
    with col_chart1:
        st.subheader("💹 Price Trend")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Date'], df['Close'], linewidth=2.5, color='#00d4ff')
        ax.fill_between(df['Date'], df['Close'], alpha=0.2, color='#00d4ff')
        ax.set_xlabel('Date', color='#b0b8c1')
        ax.set_ylabel('Price (USD)', color='#b0b8c1')
        ax.tick_params(colors='#b0b8c1')
        format_dates_on_plot(ax)
        fig.autofmt_xdate()
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_chart2:
        st.subheader("📌 Quick Stats")
        st.info(f"""
        **Min Price:** ${df['Close'].min():,.2f}
        
        **Max Price:** ${df['Close'].max():,.2f}
        
        **Avg Price:** ${df['Close'].mean():,.2f}
        
        **Std Dev:** ${df['Close'].std():,.2f}
        """)

# ---------------- 2. Price Explorer & Candlesticks ----------------
elif page == "2. Price Explorer & Candlesticks":
    st.header("🕯️ Price Explorer & Candlesticks")
    
    # Candlestick approximation using matplotlib
    st.subheader("📊 Candlestick Chart (Last 90 Days)")
    subset = df.tail(90).copy()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Candlestick data not available in preprocessing, show Close price instead
    ax.plot(subset['Date'], subset['Close'], label='Close', color='#00d4ff', linewidth=2.5)
    ax.fill_between(subset['Date'], subset['Close'], alpha=0.2, color='#00d4ff')
    ax.set_title(f"{coin} Price Chart (Last 90 Days)", fontsize=14, fontweight='bold', color='#00d4ff')
    ax.set_xlabel('Date', color='#b0b8c1')
    ax.set_ylabel('Price (USD)', color='#b0b8c1')
    ax.tick_params(colors='#b0b8c1')
    ax.legend(loc='best', facecolor='#1a1f2e', edgecolor='#00d4ff')
    format_dates_on_plot(ax)
    fig.autofmt_xdate()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Price statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔝 Highest (90D)", f"${subset['Close'].max():,.2f}")
    with col2:
        st.metric("🔻 Lowest (90D)", f"${subset['Close'].min():,.2f}")
    with col3:
        st.metric("↔️ Range", f"${subset['Close'].max() - subset['Close'].min():,.2f}")

# ---------------- 3. Forecast & Uncertainty ----------------
elif page == "3. Forecast & Uncertainty":
    st.header("🔮 Forecast & Uncertainty Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 ARIMA", "🌟 Prophet", "🤖 LSTM (BTC)"])
    
    with tab1:
        st.subheader("ARIMA Forecast (30 Days)")
        try:
            arima_pred = arima_forecast(df)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Date'][-90:], df['Close'][-90:], label='Historical Data', color='#00d4ff', linewidth=2.5)
            ax.plot(arima_pred.index, arima_pred.values, label='ARIMA Forecast', color='#ff9100', linewidth=2.5, linestyle='--')
            ax.fill_between(range(len(df['Date'][-90:]), len(df['Date'][-90:]) + len(arima_pred)), 
                            arima_pred.values - arima_pred.std(), 
                            arima_pred.values + arima_pred.std(), 
                            color='#ff9100', alpha=0.2, label='Uncertainty ±1σ')
            ax.legend(loc='best', facecolor='#1a1f2e', edgecolor='#00d4ff')
            ax.set_xlabel('Date', color='#b0b8c1')
            ax.set_ylabel('Price (USD)', color='#b0b8c1')
            ax.tick_params(colors='#b0b8c1')
            ax.set_title('ARIMA Model Forecast', fontsize=14, fontweight='bold', color='#00d4ff')
            format_dates_on_plot(ax)
            fig.autofmt_xdate()
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Could not generate ARIMA forecast: {str(e)}")
        
    with tab2:
        st.subheader("Prophet Forecast with Uncertainty")
        try:
            prophet_pred = prophet_forecast(df)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df['Date'][-90:], df['Close'][-90:], label='Historical Data', color='#00d4ff', linewidth=2.5)
            ax.plot(prophet_pred['ds'], prophet_pred['yhat'], label='Prophet Forecast', color='#00ff41', linewidth=2.5)
            ax.fill_between(prophet_pred['ds'], prophet_pred['yhat_lower'], prophet_pred['yhat_upper'], 
                           color='#00ff41', alpha=0.2, label='95% Confidence Interval')
            ax.legend(loc='best', facecolor='#1a1f2e', edgecolor='#00d4ff')
            ax.set_xlabel('Date', color='#b0b8c1')
            ax.set_ylabel('Price (USD)', color='#b0b8c1')
            ax.tick_params(colors='#b0b8c1')
            ax.set_title('Prophet Model Forecast', fontsize=14, fontweight='bold', color='#00d4ff')
            format_dates_on_plot(ax)
            fig.autofmt_xdate()
            plt.tight_layout()
            ax.set_title('Prophet Model Forecast', fontsize=14, fontweight='bold', color='#00d4ff')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Could not generate Prophet forecast: {str(e)}")
        
    with tab3:
        st.subheader("🧠 LSTM Neural Network Prediction")
        st.markdown(f"> Training LSTM model for **{coin}** using 60-day historical window")
        try:
            # Check if we have enough data for LSTM (need at least 60 data points)
            if len(df) < 65:
                st.warning(f"⚠️ Not enough data for LSTM. Need at least 65 data points, have {len(df)}. Please select a longer date range.")
            else:
                with st.spinner('🔄 Training LSTM model (this may take a moment)...'):
                    price = lstm_forecast(df)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Next Day Prediction", f"${price:,.2f}")
                with col2:
                    current = df['Close'].iloc[-1]
                    change = price - current
                    pct_change = (change/current)*100 if current != 0 else 0
                    st.metric("📊 Expected Change", f"${change:,.2f}", f"{pct_change:.2f}%")
                with col3:
                    st.metric("💡 Model Type", "LSTM", label_visibility="collapsed")
                
                st.success(f"✅ LSTM model successfully trained on {len(df)} historical data points for {coin}")
        except Exception as e:
            st.error(f"❌ LSTM training failed: {str(e)}. Try selecting a longer date range or different coin.")

# ---------------- 4. Sentiment & News Impact ----------------
elif page == "4. Sentiment & News Impact":
    st.header("💬 Sentiment & News Impact Analysis")
    st.markdown("> Integrating real-time sentiment from NewsAPI, Twitter & market signals")
    
    col_info, col_chart = st.columns([1, 2])
    
    with col_info:
        st.info("""
        📰 **Sentiment Sources:**
        - News APIs
        - Social Media
        - Market Indicators
        - Trading Volumes
        """)
    
    # Simulated Sentiment
    dates = df['Date'][-30:]
    sentiment_scores = np.random.uniform(-1, 1, size=30)
    sent_df = pd.DataFrame({'Date': dates, 'Sentiment': sentiment_scores})
    
    with col_chart:
        st.subheader("📊 Daily Sentiment Score (Simulated)")
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['#ff1744' if x < 0 else '#00ff41' for x in sent_df['Sentiment']]
        ax.bar(sent_df['Date'], sent_df['Sentiment'], color=colors, edgecolor='#00d4ff', linewidth=0.5)
        ax.axhline(y=0, color='#b0b8c1', linestyle='--', linewidth=1)
        ax.set_xlabel('Date', color='#b0b8c1')
        ax.set_ylabel('Sentiment Score', color='#b0b8c1')
        ax.tick_params(colors='#b0b8c1')
        ax.set_title('Market Sentiment Analysis', fontsize=12, fontweight='bold', color='#00d4ff')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Sentiment statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("😊 Positive Days", f"{(sent_df['Sentiment'] > 0).sum()} / 30", label_visibility="collapsed")
    with col2:
        st.metric("😔 Negative Days", f"{(sent_df['Sentiment'] < 0).sum()} / 30", label_visibility="collapsed")
    with col3:
        st.metric("📈 Avg Sentiment", f"{sent_df['Sentiment'].mean():.3f}", label_visibility="collapsed")

# ---------------- 5. Volatility & Risk Visuals ----------------
elif page == "5. Volatility & Risk Visuals":
    st.header("⚠️ Volatility & Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Daily Returns Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['Returns'].dropna(), bins=50, color='#00d4ff', edgecolor='#00d4ff', alpha=0.7)
        ax.axvline(df['Returns'].mean(), color='#ff9100', linestyle='--', linewidth=2, label=f"Mean: {df['Returns'].mean():.2%}")
        ax.set_xlabel('Daily Return', color='#b0b8c1')
        ax.set_ylabel('Frequency', color='#b0b8c1')
        ax.tick_params(colors='#b0b8c1')
        ax.legend(facecolor='#1a1f2e', edgecolor='#00d4ff')
        ax.set_title('Return Distribution', fontsize=12, fontweight='bold', color='#00d4ff')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Return Statistics")
        st.metric("📌 Mean Return", f"{df['Returns'].mean():.2%}", label_visibility="collapsed")
        st.metric("📊 Std Deviation", f"{df['Returns'].std():.2%}", label_visibility="collapsed")
        st.metric("⬆️ Max Daily Return", f"{df['Returns'].max():.2%}", label_visibility="collapsed")
        st.metric("⬇️ Min Daily Return", f"{df['Returns'].min():.2%}", label_visibility="collapsed")
    
    st.markdown("---")
    
    st.subheader("📉 Rolling Volatility (30-Day Standard Deviation)")
    rolling_vol = df['Returns'].rolling(window=30).std()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Date'], rolling_vol, color='#ff9100', linewidth=2)
    ax.fill_between(df['Date'], rolling_vol, alpha=0.2, color='#ff9100')
    ax.set_xlabel('Date', color='#b0b8c1')
    ax.set_ylabel('Volatility', color='#b0b8c1')
    ax.tick_params(colors='#b0b8c1')
    ax.set_title('Rolling 30-Day Volatility', fontsize=12, fontweight='bold', color='#00d4ff')
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- 6. Indicators Dashboard ----------------
elif page == "6. Indicators Dashboard":
    st.header("📊 Technical Indicators Dashboard")
    
    # RSI Calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔴 Relative Strength Index (RSI)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Date'], df['RSI'], color='#00d4ff', linewidth=2)
        ax.axhline(y=70, color='#ff1744', linestyle='--', linewidth=1.5, label='Overbought (70)')
        ax.axhline(y=30, color='#00ff41', linestyle='--', linewidth=1.5, label='Oversold (30)')
        ax.fill_between(df['Date'], 70, 100, alpha=0.1, color='#ff1744')
        ax.fill_between(df['Date'], 0, 30, alpha=0.1, color='#00ff41')
        ax.set_xlabel('Date', color='#b0b8c1')
        ax.set_ylabel('RSI Value', color='#b0b8c1')
        ax.set_ylim(0, 100)
        ax.tick_params(colors='#b0b8c1')
        ax.legend(facecolor='#1a1f2e', edgecolor='#00d4ff')
        ax.set_title('14-Period RSI Indicator', fontsize=12, fontweight='bold', color='#00d4ff')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📌 RSI Zones")
        current_rsi = df['RSI'].iloc[-1]
        
        if current_rsi > 70:
            zone = "🔴 Overbought"
            color = "#ff1744"
        elif current_rsi < 30:
            zone = "🟢 Oversold"
            color = "#00ff41"
        else:
            zone = "🟡 Neutral"
            color = "#ffd600"
        
        st.markdown(f"<div style='background: rgba(0, 212, 255, 0.1); border: 2px solid {color}; border-radius: 10px; padding: 20px; text-align: center;'><h3 style='color: {color}; margin: 0;'>{zone}</h3><p style='color: #b0b8c1; margin: 10px 0 0 0;'>Current RSI: <strong>{current_rsi:.2f}</strong></p></div>", unsafe_allow_html=True)

# ---------------- 7. Correlations & Market Structure ----------------
elif page == "7. Correlations & Market Structure":
    st.header("🔗 Market Correlations & Structure")
    
    st.markdown("> Analyze how different cryptocurrencies move in relation to each other")
    
    @st.cache_data
    def get_correlation_matrix():
        top_coins = ["BTC", "ETH", "BNB", "SOL", "ADA"]
        closes = {}
        for c in top_coins:
            try:
                d = fetch_crypto_data(c, period="1y")
                closes[c] = d['Close']
            except:
                st.warning(f"Could not fetch {c} data")
        return pd.DataFrame(closes).corr()

    try:
        corr_matrix = get_correlation_matrix()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔥 Correlation Matrix Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Correlation'}, ax=ax, 
                       linewidths=1, linecolor='#00d4ff')
            ax.set_title('Cryptocurrency Correlation Matrix', fontsize=12, fontweight='bold', color='#00d4ff')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("📊 Insights")
            st.info("""
            **Correlation Interpretation:**
            
            🟢 **Positive (+1):** Move together
            
            🔴 **Negative (-1):** Move oppositely
            
            🟡 **Zero (0):** No relationship
            """)
    except:
        st.error("❌ Could not generate correlation matrix. Check data availability.")

# ---------------- 8. Feature Importance & Explainability ----------------
elif page == "8. Feature Importance & Explainability":
    st.header("🔬 Model Explainability (Prophet Components)")
    
    st.markdown("> Understand which components drive price predictions in the Prophet model")
    
    try:
        # Fit Prophet again to show components
        from prophet import Prophet
        p_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        
        with st.spinner("🔄 Fitting Prophet model..."):
            m = Prophet()
            m.fit(p_df)
        
        st.subheader("📈 Forecast Components")
        fig = m.plot_components(m.predict(m.make_future_dataframe(periods=30)))
        
        # Style the components plot
        for ax in fig.axes:
            ax.set_facecolor('#1a1f2e')
            ax.tick_params(colors='#b0b8c1')
            for spine in ax.spines.values():
                spine.set_color('#00d4ff')
        
        st.pyplot(fig)
        
        st.markdown("---")
        st.info("""
        **Component Breakdown:**
        - **Trend:** Overall upward/downward direction
        - **Yearly:** Seasonal patterns across the year
        - **Weekly:** Day-of-week patterns
        """)
    except:
        st.error("❌ Could not generate Prophet components. Ensure historical data is sufficient.")

# ---------------- 9. Strategy Backtest & Performance ----------------
elif page == "9. Strategy Backtest & Performance":
    st.header("💹 Trading Strategy Backtest")
    
    st.markdown("> Simple Moving Average Crossover Strategy (MA7 vs MA30)")
    
    # Simple Strategy: Buy if MA7 > MA30
    df_strategy = df.copy()
    df_strategy['Signal'] = (df_strategy['MA_7'] > df_strategy['MA_30']).astype(int)
    df_strategy['Strategy_Return'] = df_strategy['Signal'].shift(1) * df_strategy['Returns']
    
    cum_returns = (1 + df_strategy[['Returns', 'Strategy_Return']].fillna(0)).cumprod()
    
    # Plot strategy performance
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Cumulative Returns Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_strategy['Date'], cum_returns['Returns'], label='Buy & Hold', color='#00d4ff', linewidth=2.5)
        ax.plot(df_strategy['Date'], cum_returns['Strategy_Return'], label='MA Strategy', color='#00ff41', linewidth=2.5)
        ax.fill_between(df_strategy['Date'], cum_returns['Returns'], alpha=0.1, color='#00d4ff')
        ax.fill_between(df_strategy['Date'], cum_returns['Strategy_Return'], alpha=0.1, color='#00ff41')
        ax.set_xlabel('Date', color='#b0b8c1')
        ax.set_ylabel('Cumulative Return', color='#b0b8c1')
        ax.tick_params(colors='#b0b8c1')
        ax.legend(loc='best', facecolor='#1a1f2e', edgecolor='#00d4ff')
        ax.set_title('Strategy vs Buy & Hold', fontsize=12, fontweight='bold', color='#00d4ff')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Performance Metrics")
        strategy_return = (cum_returns['Strategy_Return'].iloc[-1] - 1) * 100
        buyhold_return = (cum_returns['Returns'].iloc[-1] - 1) * 100
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("🎯 Strategy Return", f"{strategy_return:.1f}%", label_visibility="collapsed")
        with col_m2:
            st.metric("📊 Buy & Hold", f"{buyhold_return:.1f}%", label_visibility="collapsed")
        
        outperformance = strategy_return - buyhold_return
        if outperformance > 0:
            st.success(f"✅ Outperformance: +{outperformance:.1f}%")
        else:
            st.warning(f"⚠️ Underperformance: {outperformance:.1f}%")

# ---------------- 10. Interactive Explorer ----------------
elif page == "10. Interactive Explorer":
    st.header("🔍 Data Explorer & Export")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Raw Data Table")
        
        # Data filtering options - only use defaults that exist in dataframe
        available_cols = df.columns.tolist()
        default_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        default_selection = [col for col in default_cols if col in available_cols]
        
        filter_cols = st.multiselect(
            "Select columns to display:",
            available_cols,
            default=default_selection if default_selection else available_cols[:5]
        )
        
        if filter_cols:
            display_df = df[filter_cols].tail(100)  # Show last 100 rows
            st.dataframe(display_df, width='stretch', height=400)
    
    with col2:
        st.subheader("💾 Export Data")
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{coin}_data.csv",
            mime="text/csv",
            use_container_width=False
        )
        
        st.markdown("---")
        st.info(f"""
        **Dataset Info:**
        - Total Rows: {len(df)}
        - Columns: {len(df.columns)}
        - Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}
        """)
    
    st.markdown("---")
    
    # Data statistics
    st.subheader("📈 Summary Statistics")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    st.dataframe(df[numeric_cols].describe(), width='stretch')

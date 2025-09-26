# app.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime as dt

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("About This Project")
st.sidebar.info(
    """
    **Stock Market Trend Prediction App**

    This app predicts future stock prices using a **pre-trained LSTM deep learning model**.
    You can input any stock ticker (e.g., AAPL, TSLA, POWERGRID.NS) and predict the next few days' closing prices.
    """
)

st.sidebar.title("About Me")
st.sidebar.info(
    """
    **Mirza Yasir Abdullah Baig**

    - [LinkedIn](https://www.linkedin.com/in/mirza-yasir-abdullah-baig/)
    - [GitHub](https://github.com/mirzayasirabdullahbaig07)
    - [Kaggle](https://www.kaggle.com/mirzayasirabdullah07)
    """
)

# ------------------------------
# Main Page
# ------------------------------
st.title("📈 Stock Market Trend Prediction - LSTM Model")
st.write("Predict future stock prices for any ticker using a pre-trained LSTM deep learning model.")

# ------------------------------
# User input: stock ticker
# ------------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, POWERGRID.NS):", "AAPL")
days_to_predict = st.number_input("Enter number of days to predict:", min_value=1, max_value=30, value=5)

# Load LSTM model
model = load_model("stock_dl_model.h5")

# ------------------------------
# Download stock data
# ------------------------------
start = dt.datetime(2000, 1, 1)
end = dt.datetime.today()

try:
    data = yf.download(ticker, start, end)
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

if data.empty:
    st.error("No data found for this ticker.")
    st.stop()

data = data.sort_index()
st.subheader(f"Showing last 10 rows for {ticker}")
st.write(data.tail(10))

# ------------------------------
# Download button
# ------------------------------
csv = data.to_csv(index=True)
st.download_button(
    label="📥 Download Stock Data as CSV",
    data=csv,
    file_name=f"{ticker}_data.csv",
    mime="text/csv"
)

# ------------------------------
# Prepare numeric 'Close' data
# ------------------------------
if 'Close' in data.columns:
    close_data = data['Close'].values.reshape(-1,1)
elif 'Adj Close' in data.columns:
    close_data = data['Adj Close'].values.reshape(-1,1)
else:
    st.error("Could not find a numeric 'Close' column in the data.")
    st.stop()

st.subheader("Historical Closing Price Chart")
st.line_chart(close_data)

# ------------------------------
# LSTM Prediction
# ------------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

time_step = 100
if len(scaled_data) < time_step:
    st.error(f"Not enough data to make predictions. Need at least {time_step} days.")
else:
    x_input = scaled_data[-time_step:].reshape(1, time_step, 1)
    predicted_prices = []

    for _ in range(days_to_predict):
        pred_price = model.predict(x_input, verbose=0)
        predicted_prices.append(pred_price[0][0])
        pred_price_3d = pred_price.reshape(1,1,1)
        x_input = np.append(x_input[:,1:,:], pred_price_3d, axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1,1))

    st.subheader(f"Predicted Closing Prices for next {days_to_predict} day(s)")
    pred_df = pd.DataFrame({
        'Day': range(1, days_to_predict+1),
        'Predicted Close': predicted_prices.flatten()
    })
    st.write(pred_df)

    # Plot predictions alongside historical data
    st.subheader("Historical + Predicted Closing Price Chart")
    combined_chart = np.concatenate([close_data, predicted_prices])
    st.line_chart(combined_chart)

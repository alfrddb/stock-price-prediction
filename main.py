import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Predictor")
st.markdown("Forecast future stock prices using a Long Short-Term Memory (**LSTM**) deep learning model built with PyTorch.")

# --- SIDEBAR: USER PARAMETERS ---
st.sidebar.header("Model Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
seq_length = st.sidebar.slider("Sequence Length (Days)", min_value=10, max_value=90, value=60)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=50, value=20)

# --- DATA LOADING ---
@st.cache_data
def load_data(tk, start):
    df = yf.download(tk, start)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data_load_state = st.text("Loading data from Yahoo Finance...")
df = load_data(ticker, start_date)
data_load_state.text("")

if df.empty or 'Close' not in df.columns:
    st.error("Error retrieving data. Please check the stock ticker (e.g., AAPL, TSLA, MSFT).")
else:
    st.subheader(f"Historical Trend: {ticker.upper()}")
    st.line_chart(df['Close'])

    # --- DATA PREPARATION FOR LSTM ---
    close_prices = df['Close'].values.reshape(-1, 1)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(close_prices)

    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            x = data[i:(i + seq_len)]
            y = data[i + seq_len]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(scaled_data, seq_length)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # --- LSTM MODEL ARCHITECTURE ---
    class LSTMModel(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=50, num_layers=1, output_dim=1):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # --- MODEL TRAINING ---
    if st.button("Start LSTM Model Training"):
        with st.spinner("Training in progress... Please wait."):
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                output = model(X_train_t)
                loss = criterion(output, y_train_t)
                loss.backward()
                optimizer.step()
        
        st.success("Training completed successfully!")

        # --- EVALUATION AND PREDICTION ---
        model.eval()
        with torch.no_grad():
            train_predict = model(X_train_t).numpy()
            test_predict = model(X_test_t).numpy()

        # Inverse transformation
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        # --- RESULTS PLOT ---
        st.subheader("Actual vs Predicted Prices Comparison")
        
        test_dates = df.index[seq_length + len(train_predict) : seq_length + len(train_predict) + len(test_predict)]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], label="Historical Actual Price", color="black", alpha=0.5)
        ax.plot(test_dates, test_predict, label="LSTM Prediction (Test)", color="orange", linewidth=2)
        ax.set_title(f"{ticker.upper()} - Closing Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

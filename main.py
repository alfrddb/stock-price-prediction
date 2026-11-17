import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="LSTM Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Predictor")
st.markdown("Forecast future stock prices using a Long Short-Term Memory (**LSTM**) deep learning model built with PyTorch.")

st.sidebar.header("Model Parameters")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
seq_length = st.sidebar.slider("Sequence Length (Days)", min_value=10, max_value=90, value=60)
epochs = st.sidebar.slider("Training Epochs", min_value=5, max_value=50, value=20)

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

    close_prices = df['Close'].values.reshape(-1, 1)
    
    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            x = data[i:(i + seq_len)]
            y = data[i + seq_len]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(close_prices, seq_length)

    if len(X) < 50:
        st.warning("Not enough historical data for the selected start date and sequence length. Please choose an earlier start date or a shorter sequence length.")
    else:
        train_size = int(len(X) * 0.8)
        
        train_raw_end = train_size + seq_length
        train_prices = close_prices[:train_raw_end]
        
        scaler = StandardScaler()
        scaler.fit(train_prices)
        scaled_data = scaler.transform(close_prices)

        X_scaled, y_scaled = create_sequences(scaled_data, seq_length)
        
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

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

            model.eval()
            with torch.no_grad():
                train_predict = model(X_train_t).numpy()
                test_predict = model(X_test_t).numpy()

            train_predict = scaler.inverse_transform(train_predict)
            test_predict = scaler.inverse_transform(test_predict)

            st.subheader("Actual vs Predicted Prices Comparison")
            
            test_dates_index = df.index[seq_length + train_size : seq_length + train_size + len(test_predict)]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df.index, df['Close'], label="Historical Actual Price", color="black", alpha=0.5)
            ax.plot(test_dates_index, test_predict, label="LSTM Prediction (Test)", color="orange", linewidth=2)
            ax.set_title(f"{ticker.upper()} - Closing Price Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)

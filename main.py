import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# Streamlit page configuration
st.set_page_config(page_title="LSTM Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction with LSTM & PyTorch")
st.markdown("This web app uses a Deep Learning model (LSTM) to analyze and predict closing stock prices based on historical data from Yahoo Finance.")

# --- Sidebar for user controls ---
st.sidebar.header("Model Parameters")
ticker = st.sidebar.text_input("Enter Ticker (e.g., AMZN, AAPL, TSLA)", value="AMZN")
start_date = st.sidebar.date_input("Start Date for History", value=pd.to_datetime("2020-01-01"))
seq_length = st.sidebar.slider("Sequence Length (Days)", min_value=10, max_value=60, value=30)
num_epochs = st.sidebar.slider("Training Epochs", min_value=50, max_value=300, value=100, step=25)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading ---
@st.cache_data
def load_data(tk, start):
    df = yf.download(tk, start)
    return df

data_load_state = st.text("Downloading data from Yahoo Finance...")
df = load_data(ticker, start_date)
data_load_state.text("Data loaded successfully!")

if df.empty:
    st.error("Error: No data found for the entered ticker or invalid date range.")
else:
    # Show dataframe preview
    with st.expander("View raw downloaded data"):
        st.write(df.tail())

    # Historical closing price chart
    st.subheader(f"Historical Closing Prices ({ticker})")
    fig, ax = plt.subplots(figsize=(12, 5))
    df['Close'].plot(ax=ax, title=f"Closing Price - {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # --- Preprocessing ---
    scaler = StandardScaler()
    close_data = df[['Close']].copy()
    close_data['Close_Scaled'] = scaler.fit_transform(close_data[['Close']])

    values = close_data['Close_Scaled'].values

    data = []
    for i in range(len(values) - seq_length):
        data.append(values[i:i+seq_length])
    data = np.array(data)

    if len(data) == 0:
        st.warning("The dataset is too short for the chosen sequence length. Please reduce the 'Sequence Length'.")
    else:
        train_size = int(0.8 * len(data))

        X_train = torch.from_numpy(data[:train_size, :-1]).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(data[:train_size, -1]).type(torch.Tensor).to(device)

        X_test = torch.from_numpy(data[train_size:, :-1]).type(torch.Tensor).to(device)
        y_test = torch.from_numpy(data[train_size:, -1]).type(torch.Tensor).to(device)

        # --- LSTM Model Definition ---
        class PredictionModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super(PredictionModel, self).__init__()
                self.num_layers = num_layers
                self.hidden_dim = hidden_dim
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
                out, _ = self.lstm(x, (h0.detach(), c0.detach()))
                out = self.fc(out[:, -1, :])
                return out

        model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # --- Model Training ---
        if st.button("Start LSTM Model Training"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(num_epochs):
                model.train()
                y_train_pred = model(X_train)
                loss = criterion(y_train_pred, y_train)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update progress bar
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.6f}")

            st.success("Training completed successfully!")

            # --- Evaluation & Predictions ---
            model.eval()
            y_train_pred = model(X_train)
            y_test_pred = model(X_test)

            # Inverse transform to return to real prices
            y_train_pred_inv = scaler.inverse_transform(y_train_pred.detach().cpu().numpy())
            y_train_inv = scaler.inverse_transform(y_train.detach().cpu().numpy())
            y_test_pred_inv = scaler.inverse_transform(y_test_pred.detach().cpu().numpy())
            y_test_inv = scaler.inverse_transform(y_test.detach().cpu().numpy())

            train_rmse = root_mean_squared_error(y_train_inv[:, 0], y_train_pred_inv[:, 0])
            test_rmse = root_mean_squared_error(y_test_inv[:, 0], y_test_pred_inv[:, 0])

            col1, col2 = st.columns(2)
            col1.metric("Train RMSE", f"{train_rmse:.2f}")
            col2.metric("Test RMSE", f"{test_rmse:.2f}")

            # Results Chart
            st.subheader("Predictions vs Actual Data (Test Set)")
            fig_res, ax_res = plt.subplots(figsize=(12, 6))
            
            test_dates = df.index[train_size + seq_length:]
            
            ax_res.plot(test_dates, y_test_inv[:, 0], label="Actual", color="blue")
            ax_res.plot(test_dates, y_test_pred_inv[:, 0], label="Predicted (LSTM)", color="orange", linestyle="--")
            ax_res.set_xlabel("Date")
            ax_res.set_ylabel("Price")
            ax_res.legend()
            st.pyplot(fig_res)

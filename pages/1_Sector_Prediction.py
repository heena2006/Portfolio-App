import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, root_mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta,date

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

if "top_sectors" in st.session_state:

    st.subheader("Previous Sector Prediction")

    st.dataframe(
        st.session_state["top_sectors"]
    )
# Title
st.title("📈 Sector/Stock-wise Prediction")

# Inputs
sectors_input = st.text_input("Enter Sector Symbols (comma-separated)", "^CNXAUTO,^CNXIT,^NSEBANK,^CNXFMCG,^CNXMETAL,^CNXMEDIA,^CNXPSUBANK,^CNXREALTY,^CNXCONSUM,NIFTY_FIN_SERVICE.NS,^CNXPHARMA")
#start_date = st.date_input("From Date", pd.to_datetime("2019-01-01"))
#end_date = st.date_input("To Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input(
    "**Select Investment Date**",
    value=date.today()
)

# Take extra buffer days
temp_start = end_date - timedelta(days=580)

# Download temporary data
temp_data = yf.download(
    sectors_input,
    start=temp_start,
    end=end_date
)['Close'].bfill()

# Keep last 252 trading days only
data = temp_data.tail(378)

start_date = data.index[0].date()

#st.write("Auto Start Date:", start_date)
#st.write("End Date:", end_date)
total_days, num_stocks = data.shape
st.write("Total Days:", total_days)
#st.write("Number of Sectors:", num_stocks)

run_button = st.button("Run Prediction")

if run_button:

    st.session_state["start_date"] = start_date

    st.session_state["end_date"] = end_date

    stocks = [s.strip() for s in sectors_input.split(",")]

    summary_data = []

    for stock in stocks:
        st.write(f"Processing: {stock}")

        stock_data = yf.download(stock, start=start_date, end=end_date)

        if stock_data.empty:
            #st.warning(f"No data for {stock}")
            continue

        closing_prices = stock_data["Close"]

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(closing_prices.values.reshape(-1, 1))

        # Dataset creation
        def create_dataset(dataset, window_size=5):
            x, y = [], []
            for i in range(len(dataset) - window_size - 1):
                x.append(dataset[i:(i + window_size), 0])
                y.append(dataset[i + window_size, 0])
            return np.array(x), np.array(y)

        window_size = 5
        x, y = create_dataset(scaled_data, window_size)

        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))

        # Model
        model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(1, window_size), activation='tanh', recurrent_activation='sigmoid'),
        Dropout(0.2),
        LSTM(16, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'),
        Dropout(0.2),
        Dense(1)
         ])

        model.compile(optimizer="adam", loss="mean_squared_error")

        # Train the model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto', restore_best_weights=True)
        model.fit(x, y, batch_size=32, epochs=10, verbose=0, callbacks=[early_stopping])


        #model.fit(x, y, epochs=20, batch_size=32, verbose=0)

        # Predictions
        predictions = model.predict(x)

        predictions = scaler.inverse_transform(predictions)
        y_actual = scaler.inverse_transform(y.reshape(-1, 1))

        # Metrics
        r2 = r2_score(y_actual, predictions)
        mape = mean_absolute_percentage_error(y_actual, predictions)
        rmse = root_mean_squared_error(y_actual, predictions)
        accuracy = 100 * (1 - mape)

        # Future prediction (4 months)
        future_predictions_list = []
        last_data = scaled_data[-window_size:]

        for _ in range(6): #6
            current_input = last_data.reshape(1, 1, window_size)
            pred = model.predict(current_input)
            future_predictions_list.append(pred[0, 0])
            last_data = np.append(last_data[1:], pred, axis=0)

        future_predictions = scaler.inverse_transform(np.array(future_predictions_list).reshape(-1, 1))
        future_returns = (future_predictions[-1][0]-y_actual[-1][0])/y_actual[-1][0] *100

        summary_data.append([
            stock,
            predictions[-1][0],
            y_actual[-1][0],
            future_predictions[-1][0],
            future_returns,
            mape,
            rmse,
            r2,
            accuracy
        ])

       # Plot
        #fig, ax = plt.subplots()
        #ax.plot(y_actual, label="Actual")
       #ax.plot(predictions, label="Predicted")
        #ax.set_title(stock)
        #ax.legend()
        #st.pyplot(fig)

    # Display results
    df = pd.DataFrame(summary_data, columns=[
        "Sector",
        "Predicted Value",
        "Actual Value",
        "Future Prediction",
        "Future Return"
    ])
    # Sort by Future Return (Descending)
    df = df.sort_values(by="Future Return", ascending=False)
    st.subheader("📊 Summary Results For Next 6 months")
    #st.dataframe(df)
    top5 = df.head(5)
    st.dataframe(top5)
    st.session_state["top_sectors"] = top5["Sector"].tolist()
    #st.session_state["top_sectors"] = top5

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "sector_results.csv", "text/csv")

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
from utils.sector_mapping import sector_stocks

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

st.title("📊 Stock Prediction")

if "top_sectors" not in st.session_state:

    st.warning("Run Sector Prediction first.")

else:
    start_date = st.session_state["start_date"]

    end_date = st.session_state["end_date"]

    selected_sectors = st.session_state["top_sectors"]
    #st.write(selected_sectors)
    #st.subheader("Previous Stock Prediction")

    #st.dataframe(
     #   st.session_state["top_stock"]
    #)
# Title

    stock_summary = []
    all_stocks = []
    for sector in selected_sectors:
        sector = sector.strip()
        stock_list = sector_stocks.get(sector,[])
        st.subheader(f"Stocks in {sector}")
        if not stock_list:

            st.warning(f"No stock mapping found for {sector}")
            continue
        stock_df = pd.DataFrame(
        stock_list,
        columns=["Stocks"]
         )

        #st.dataframe(stock_df)

        #all_stocks.extend(stock_list)

        for stock in stock_list:

            st.write(f"Processing: {stock}")

            stock_data = yf.download(stock, start=start_date, end=end_date)

            if stock_data.empty:
                st.warning(f"No data for {stock}")
                continue

            closing_prices = stock_data["Close"]
            total_days, num_stocks = closing_prices.shape
            #st.write("Total Days:", total_days)
            #st.write("Number of Stocks:", num_stocks)

        # Scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(closing_prices.values.reshape(-1, 1))

        # Dataset creation
            def create_dataset(dataset, window_size=5):  #5
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


        #model.fit(x, y, epochs=10, batch_size=32, verbose=0)

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
            future_returns = (future_predictions[-1][-1]-y_actual[-1][-1])/y_actual[-1][-1] *100

            stock_summary.append([
            sector,
            stock,
            predictions[-1][-1],
            y_actual[-1][-1],
            future_predictions[-1][-1],
            future_returns
            ])
    #st.session_state["selected_stocks"] = all_stocks
       # Plot
        #fig, ax = plt.subplots()
        #ax.plot(y_actual, label="Actual")
       #ax.plot(predictions, label="Predicted")
        #ax.set_title(stock)
        #ax.legend()
        #st.pyplot(fig)

    # Display results
df = pd.DataFrame(stock_summary, columns=[
        "Sector",
        "Stock",
        "Predicted Value",
        "Actual Value",
        "Future Prediction",
        "Future Return"
         ])
    # Sort by Future Return (Descending)
df = df.sort_values(by="Future Return", ascending=False)
st.subheader("📊 Summary Results For Next 6 months")
    #st.dataframe(df)
#top2 = df.head(2)
top2 = df.groupby("Sector").head(2)
st.subheader(
            "📊 Top 2 Stocks From Each Sector"
        )

st.dataframe(top2)
# --------------------------------
# Save Stocks
# --------------------------------
st.session_state["top_stock_df"] = top2[["Sector", "Stock"]]
st.session_state["top_stock"] = (top2["Stock"].tolist())

#st.dataframe(df)
#st.session_state["top_stock"] = top2["Stock"].tolist()

    # Download option
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "stock_results.csv", "text/csv")

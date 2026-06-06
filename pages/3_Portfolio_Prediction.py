import streamlit as st
from datetime import date
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

np.random.seed(42)
tf.random.set_seed(42)

st.set_page_config(
    page_title="Portfolio Optimization",
    layout="wide"
)
st.title("📊 Proposed Model")
# -----------------------------
# User Inputs
# -----------------------------
#start_date = st.date_input("**Start Date**")
#end_date = st.date_input("**End Date**")
#st.sidebar.header("Model Inputs")
#start_date = st.sidebar.date_input("**Start Date**")
#end_date = st.sidebar.date_input("**End Date**")
st.title("📊 Portfolio Prediction")

if "top_sectors" not in st.session_state:

    st.warning("Run Sector Prediction first.")

else:
    start_date = st.session_state["start_date"]

    end_date = st.session_state["end_date"]

    selected_sectors = st.session_state["top_sectors"]
    #st.write(selected_sectors)

    #stock_list = st.session_state["top_stock"]
    stock_list = [s.strip() for s in st.session_state["top_stock"]]
    #st.write(stock_list)
    st.subheader("Sector Prediction")
    st.dataframe(
        st.session_state["top_sectors"]
    )
    st.subheader("Stock Prediction")
    st.dataframe(
        st.session_state["top_stock_df"]
    )

    # modify stock
    default_stocks = ",".join(st.session_state["top_stock"])

    stock_input = st.text_input("Portfolio Stocks",value=default_stocks)

    stocks = [s.strip()    for s in stock_input.split(",")]
    default_start = st.session_state.get("start_date", pd.to_datetime("2019-01-01"))

    default_end = st.session_state.get("end_date",date.today())
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end
        )
if st.button("🚀 Optimize Portfolio"):

    stocks = [
        s.strip()
        for s in stock_input.split(",")
        if s.strip()
    ]

    st.session_state["portfolio_stocks"] = stocks
    st.session_state["portfolio_start"] = start_date
    st.session_state["portfolio_end"] = end_date
# Title
    #for stock in stock_list:
     #   st.write(f"Processing: {stock}")
    #   stock = [s.strip() for s in stock_list]

        #st.write("Downloading data...")
    data = yf.download(stocks, start=start_date, end=end_date)['Close'].bfill()
    #data = data.dropna(axis=1, how='all')
    #if data.empty:
     #   st.warning(f"No data for {stock}")
      #  continue

    valid_stocks = data.columns.tolist()

    #st.write("Valid Stocks:", valid_stocks)

    #stocks = [s.strip() for s in stock_input.split(",")]

    #st.write("Downloading data...")
    #data = yf.download(stocks, start=start_date, end=end_date)['Close'].bfill()

    log_return = np.log(data / data.shift(1)).dropna()
    returns_array = log_return.values
    total_days, num_stocks = returns_array.shape
    #st.write("Total Days:", returns_array.shape)
    st.write("Total Days:", total_days)
    st.write("Number of Stocks:", num_stocks)

    train_window = 252
    test_window = 126
    lookback = 30
    results = []

    #----------------------
    #NIFTY 100
    #----------------------
    nifty = yf.download("^CNX100", start=start_date, end=end_date)['Close'].bfill()
    nifty_returns = np.log(nifty / nifty.shift(1)).dropna()

    # -----------------------------
    # LSTM Model
    # -----------------------------
    def create_lstm_model(input_shape, num_stocks):
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=input_shape,
                 activation='tanh', recurrent_activation='sigmoid'),
            Dropout(0.2),
            LSTM(16, return_sequences=False,
                 activation='tanh', recurrent_activation='sigmoid'),
            Dropout(0.2),
            Dense(num_stocks)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # -----------------------------
    # Portfolio Metrics
    # -----------------------------
    def portfolio_metrics(weights, returns, days):
        sum_returns = np.sum(returns,axis=0)
        avg_returns = sum_returns/days #np.sum(returns,axis=0)
        #avg_returns = np.mean(returns, axis=0)
        absolute_returns = (((1+avg_returns))**round(days)-1)
        annual_returns = (((1+absolute_returns))**round(252/days)-1)
        portfolio_return = np.sum(weights * annual_returns)

        covariance = np.cov(returns, rowvar=False)
        variance = np.dot(weights.T, np.dot(covariance, weights))
        sum_covariance = np.sum(variance)
        portfolio_risk = np.sqrt(sum_covariance*(252))
        #portfolio_risk = np.sqrt(variance * 252)

        return portfolio_return, portfolio_risk

    # -----------------------------
    # ACO
    # -----------------------------
    def ant_colony_optimization(returns):
        min_weight, max_weight = 0.03, 0.25
        pheromone = np.ones(returns.shape[1]) / returns.shape[1]
        best_solution = np.zeros(returns.shape[1])
        best_performance = -np.inf
        epsilon = 1e-8

        num_iterations, num_ants =50, 60
        evaporation_rate, alpha, beta = 0.2, 2, 3
        rf_rate = 0.0685

        for interation in range(num_iterations):
            for ant in range(num_ants):
                heuristic = np.random.uniform(min_weight, max_weight, returns.shape[1])
                probabilities = pheromone ** alpha * heuristic ** beta
                raw_weights = probabilities   # Normalize probabilities
                sum_weights = np.sum(raw_weights)  # Ensure they sum to 1
                weights = np.array(raw_weights)/sum_weights
                #weights = probabilities / np.sum(probabilities)
                weights = np.clip(weights, min_weight, max_weight)
                weights /= np.sum(weights)

                port_ret, port_risk = portfolio_metrics(weights, returns, len(returns))
                if port_risk < epsilon:
                    continue
                else:
                    port_risk = 0.30

                performance = (port_ret - rf_rate) / port_risk

                if port_risk <= 0.30:
                    if performance > best_performance:
                     best_performance = performance
                     best_solution = weights
          # Avoid division by zero or negative infinity
            if best_performance > -np.inf and best_solution is not None:
                pheromone = (1 - evaporation_rate) * pheromone
                pheromone += evaporation_rate / best_performance
                best_solution = np.clip(best_solution, min_weight, max_weight)
                best_solution /= np.sum(best_solution)

        return best_solution, best_performance

    # -----------------------------
    # Walk-forward
    # -----------------------------
    step = train_window
    start = 0

    progress = st.progress(0)

    while (start + train_window + test_window) <= total_days:

        train_data = returns_array[start:start+train_window]
        test_data = returns_array[start+train_window:start+train_window+test_window]

        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_data)

        X_train, y_train = [], []

        for i in range(len(scaled_train) - lookback):
            X_train.append(scaled_train[i:i+lookback])
            y_train.append(scaled_train[i+lookback])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        model = create_lstm_model((lookback, num_stocks), num_stocks)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30,verbose=0, mode='auto', restore_best_weights=True)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0,callbacks=[early_stopping])

        last_input = scaled_train[-lookback:].copy()
        predictions_scaled = []

        for t in range(test_window):
            pred = model.predict(last_input.reshape(1, lookback, num_stocks),
                                 verbose=0).reshape(num_stocks,)
            predictions_scaled.append(pred)

            if t < lookback:
                true_scaled = scaler.transform(test_data[t].reshape(1,-1))
                last_input = np.vstack([last_input[1:], true_scaled])
            else:
                last_input = np.vstack([last_input[1:], pred[np.newaxis]])
                #last_input = np.vstack([last_input[1:], pred])

        predictions_scaled = np.array(predictions_scaled)
        Pred_price = scaler.inverse_transform(predictions_scaled)
        optimal_weights_mv, optimal_score_mv = ant_colony_optimization(train_data)
        optimal_weights, optimal_score = ant_colony_optimization(Pred_price)
        # Equal Weight Portfolio
        equal_weights = np.ones(num_stocks) / num_stocks
        eq_test_ret, eq_test_risk = portfolio_metrics(equal_weights, test_data, test_window)
        eq_sharpe = (eq_test_ret - 0.0685) / eq_test_risk

        #NIFTY 50 DATA
        nifty_test = nifty_returns.values[start+train_window :start+train_window+test_window]

        nifty_ret = np.mean(nifty_test) * 252
        nifty_risk = np.std(nifty_test) * np.sqrt(252)
        nifty_sharpe = (nifty_ret - 0.0685) / nifty_risk

        train_ret, train_risk = portfolio_metrics(optimal_weights, train_data, train_window)
        test_ret, test_risk = portfolio_metrics(optimal_weights, test_data, test_window)
        pred_ret, pred_risk   = portfolio_metrics(optimal_weights, Pred_price,test_window)
        test_ret_mv, test_risk_mv = portfolio_metrics(optimal_weights_mv, test_data,test_window)
        mv_sharpe = (test_ret_mv - 0.0685) / test_risk_mv

        #results.append([start, train_ret, train_risk, test_ret, test_risk,pred_ret, pred_risk,optimal_weights ])
        results.append([start,test_ret, test_risk,(test_ret-0.0685)/test_risk,test_ret_mv, test_risk_mv,mv_sharpe,eq_test_ret,eq_sharpe,nifty_ret,nifty_sharpe,optimal_weights])

        start += step
        progress.progress(min(start/total_days, 1.0))

    results_df = pd.DataFrame(results,
                              columns=["Days","PM_Return","PM_Risk","PM_Sharpe","MV_Return","MV_Risk","MV_Sharpe","EQ_Return",
             "EQ_Sharpe","NIFTY_Return",
             "NIFTY_Sharpe","Optimal Weight"])

    st.success("Model Completed!")

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
    "Portfolio Results",
    "Comparison",
    "Graph"
    ])

    with tab1:
        st.subheader("Portfolio Results")
        #st.write("Cumulative Return:", (1 + results_df["PM_Return"]).cumprod() - 1)
        st.write("Cumulative Return:", round(results_df["PM_Return"].sum(),2))
        st.write("Average Sharpe Ratio:", round(results_df["PM_Sharpe"].mean(),2))
        #st.dataframe(results_df)

        st.subheader("Optimal Stock Allocation")
        weight_df = pd.DataFrame({
            "Stock": valid_stocks,
            "Weight (%)": (optimal_weights *100).round(2)
        })
        st.dataframe(weight_df)

    # -----------------------------------
    # TAB 2 — Portfolio Comparision
    # -----------------------------------
    with tab2:
        st.subheader("📊 Portfolio Comparison")
        st.write("Average Sharpe Ratios")
        st.write("Proposed model:", round(results_df["PM_Sharpe"].mean(),4))
        st.write("MV model:", round(results_df["MV_Sharpe"].mean(),4))
        st.write("Equal Weight:", round(results_df["EQ_Sharpe"].mean(),4))
        st.write("NIFTY 100:", round(results_df["NIFTY_Sharpe"].mean(),4))
        st.subheader("Portfolio Result Fold Wise")
        st.dataframe(results_df)
    #comparison = results_df.set_index("Fold")[[
    #"PM_Return",
    #"EqualWeight_Return",
    #"NIFTY_Return"]]

  #  st.bar_chart(comparison)

# -----------------------------------
# TAB 2 — Portfolio Comparision Graph
# -----------------------------------
    with tab3:
        st.subheader("📊 Portfolio Comparisonn Graph Fold wise")
        import plotly.graph_objects as go

# Ensure Fold column exists
        results_df["Fold"] = range(1, len(results_df) + 1)

        fig = go.Figure()

        # Bar 1 – Proposed Model
        fig.add_trace(go.Bar(
                x=results_df["Fold"],
                y=results_df["PM_Return"],
                name="Proposed Model"
            ))
            # Bar 1 – Proposed Model
        fig.add_trace(go.Bar(
                x=results_df["Fold"],
                y=results_df["MV_Return"],
                name="MV Model"
            ))

            # Bar 2 – Equal Weight
        fig.add_trace(go.Bar(
            x=results_df["Fold"],
                y=results_df["EQ_Return"],
                name="Equal Weight"
            ))

        # Bar 3 – NIFTY 100
        fig.add_trace(go.Bar(
                x=results_df["Fold"],
                y=results_df["NIFTY_Return"],
                name="NIFTY 100"
            ))

        fig.update_layout(
                barmode='group',   # 🔥 THIS IS IMPORTANT
                title="Fold-wise Return Comparison",
                xaxis_title="Fold",
                yaxis_title="Return",
                xaxis=dict(tickmode='linear')
            )

        st.plotly_chart(fig, use_container_width=True)



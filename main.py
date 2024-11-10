# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from statsmodels.tsa.arima.model import ARIMA
#from fbprophet import Prophet
#from statsforecast import StatsForecast
#from statsforecast.models import AutoARIMA

# Set page configuration
st.set_page_config(layout="wide")

# Function to calculate regression metrics
def calculate_metrics(y_true, y_pred, n, p):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, rmse, mae, r2, adj_r2, mape

# Function to forecast using ARIMA
def forecast_arima(data, target_column, p, d, q, future_steps, conf_interval):
    model = ARIMA(data[target_column], order=(p, d, q))
    model_fit = model.fit()
    
    # Forecasting future steps
    forecast = model_fit.get_forecast(steps=future_steps)
    forecast_values = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=(1 - conf_interval / 100))
    
    forecast_df = pd.DataFrame({
        'Forecast': forecast_values,
        'Lower Bound': conf_int.iloc[:, 0],
        'Upper Bound': conf_int.iloc[:, 1]
    })
    
    return forecast_df

# Function to forecast using Prophet
def forecast_prophet(data, target_column, future_steps):
    df = data[[target_column]].reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    
    # Future data frame
    future = model.make_future_dataframe(periods=future_steps)
    forecast = model.predict(future)
    
    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_steps)
    forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
    
    return forecast_df

# Function to forecast using StatsForecast AutoARIMA
def forecast_statsforecast(data, target_column, future_steps):
    sf = StatsForecast(models=[AutoARIMA()], freq='D')
    y = data.set_index(pd.date_range(start='2022-01-01', periods=len(data), freq='D'))[target_column]
    sf.fit(y.to_frame())
    
    # Forecasting
    forecast = sf.predict(future_steps)['AutoARIMA']
    forecast_df = pd.DataFrame({
        'Forecast': forecast
    })
    
    return forecast_df

# Streamlit App
st.title("Data Modeling and Forecasting")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Task Selection: Prediction or Forecasting
    task = st.selectbox("Choose Task", ["Prediction", "Forecasting"])
    
    if task == "Prediction":
        # Prediction Task Configuration
        algorithm = st.selectbox("Algorithm", ["Linear Regression", "Random Forest", "XGBoost"])
        target_column = st.selectbox("Field to predict", data.columns)
        feature_columns = st.multiselect("Fields to use for predicting", [col for col in data.columns if col != target_column])
        test_split = st.slider("Split for training/test", 0.1, 0.9, 0.3)
        fit_intercept = st.checkbox("Fit Intercept", value=True) if algorithm == "Linear Regression" else None
        
        # Train the model button
        if st.button("Fit Model"):
            if target_column and feature_columns:
                # Train the model
                model, mse, rmse, mae, r2, adj_r2, mape, results_df = train_model(
                    data, target_column, feature_columns, algorithm, test_split, fit_intercept
                )
                
                # Display regression metrics
                st.subheader("Model Performance Metrics")
                st.write(f"R² Statistic: {r2:.4f}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                
                # Display additional metrics in a two-column layout
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                with col2:
                    st.write(f"Adjusted R-squared: {adj_r2:.4f}")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                
                # Display results table
                st.subheader("Prediction Results")
                st.write(results_df.head(10))
                
                # Visualizations
                st.subheader("Actual vs. Predicted Line Chart")
                fig, ax = plt.subplots()
                ax.plot(results_df[target_column].values, label="Actual")
                ax.plot(results_df[f'predicted({target_column})'].values, label="Predicted")
                ax.set_xlabel("Sample Index")
                ax.set_ylabel(target_column)
                ax.legend()
                st.pyplot(fig)
                
    elif task == "Forecasting":
        # Forecasting Task Configuration
        forecast_algo = st.selectbox("Select Forecasting Algorithm:", ["ARIMA", "Prophet", "StatsForecast"])
        target_column = st.selectbox("Field to forecast", data.columns)

        if forecast_algo == "ARIMA":
            p = st.number_input("AR (autoregressive) - p", min_value=0, step=1, value=1)
            d = st.number_input("I (integrated) - d", min_value=0, step=1, value=1)
            q = st.number_input("MA (moving average) - q", min_value=0, step=1, value=1)
            future_steps = st.number_input("Future Timespan (days)", min_value=1, step=1, value=5)
            conf_interval = st.slider("Confidence Interval", min_value=0, max_value=100, value=95)
            
            if st.button("Forecast with ARIMA"):
                forecast_df = forecast_arima(data, target_column, p, d, q, future_steps, conf_interval)
                st.subheader("Forecasted Results")
                st.write(forecast_df)

        elif forecast_algo == "Prophet":
            future_steps = st.number_input("Future Timespan (days)", min_value=1, step=1, value=5)
            
            if st.button("Forecast with Prophet"):
                forecast_df = forecast_prophet(data, target_column, future_steps)
                st.subheader("Forecasted Results")
                st.write(forecast_df)

        elif forecast_algo == "StatsForecast":
            future_steps = st.number_input("Future Timespan (days)", min_value=1, step=1, value=5)
            
            if st.button("Forecast with StatsForecast"):
                forecast_df = forecast_statsforecast(data, target_column, future_steps)
                st.subheader("Forecasted Results")
                st.write(forecast_df)
else:
    st.write("Please upload a CSV file to get started.")

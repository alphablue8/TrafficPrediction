# Importing necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
#from statsmodels.tsa.arima.model import ARIMA
#from fbprophet import Prophet
#from statsforecast import StatsForecast
#from statsforecast.models import AutoARIMA

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# Function to train prediction model
def train_prediction_model(data, target_col, predictor_cols, algorithm):
    X = data[predictor_cols]  # Features
    y = data[target_col]       # Target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize model based on selected algorithm
    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif algorithm == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse, X_test, y_test, y_pred

# Function to train forecasting model
def train_forecasting_model(data, target_col, algorithm):
    y = data[target_col]  # Target for forecasting

    if algorithm == "ARIMA":
        model = ARIMA(y, order=(5,1,0))  # Order is an example, needs tuning
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)
    
    elif algorithm == "Prophet":
        df = data.rename(columns={target_col: 'y'})
        df['ds'] = pd.date_range(start='2022-01-01', periods=len(data), freq='D')
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=10)
        forecast = model.predict(future)['yhat'][-10:]
    
    elif algorithm == "StatsForecast (AutoARIMA)":
        model = StatsForecast(models=[AutoARIMA()], freq='D')
        model.fit(y.to_frame())
        forecast = model.predict(10)['AutoARIMA']
    
    return forecast

# Streamlit App
st.title("Cellular Traffic Prediction and Forecasting")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Step 2: Load data
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())
    
    # Step 3: Choose Task Type
    task_type = st.selectbox("Choose Task Type:", ["Predict", "Forecast"], key="task_type")

    if task_type == "Predict":
        # Prediction options
        algorithm = st.selectbox("Select Prediction Algorithm:", ["Linear Regression", "Random Forest", "XGBoost"], key="prediction_algorithm")
        
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Column to predict:", data.columns)
        with col2:
            predictor_cols = st.multiselect("Predictor columns:", [col for col in data.columns if col != target_col])
        
        # Train model for prediction
        if st.button("Train Prediction Model"):
            if target_col and predictor_cols:
                model, mse, X_test, y_test, y_pred = train_prediction_model(data, target_col, predictor_cols, algorithm)
                st.write(f"Mean Squared Error of the model: {mse}")
                
                # Display actual vs predicted values
                st.write("Actual vs Predicted:")
                result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                st.write(result_df.head())
                
                # Download prediction results
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name='cellular_traffic_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.warning("Please select both the target column and predictor columns.")

    elif task_type == "Forecast":
        # Forecasting options
        algorithm = st.selectbox("Select Forecasting Algorithm:", ["ARIMA", "Prophet", "StatsForecast (AutoARIMA)"], key="forecast_algorithm")
        target_col = st.selectbox("Column to forecast:", data.columns, key="forecast_target")
        
        # Train model for forecasting
        if st.button("Train Forecasting Model"):
            if target_col:
                forecast = train_forecasting_model(data, target_col, algorithm)
                st.write("Forecast for next 10 steps:")
                st.write(forecast)
                
                # Download forecast results
                forecast_df = pd.DataFrame({'Forecast': forecast})
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download forecast as CSV",
                    data=csv,
                    file_name='cellular_traffic_forecast.csv',
                    mime='text/csv'
                )
else:
    st.write("Please upload a CSV file to start the prediction or forecasting process.")

# Import necessary libraries
import streamlit as st
import pandas as pd

# Set page configuration
st.set_page_config(layout="wide")

# Title
st.title("Data Modeling and Forecasting")

# Step 1: Upload CSV File
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:

    # Load and display data
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Step 2: Select Cell Name (Single Selection)
    if 'Cell Name' in data.columns:
        unique_cell_names = data['Cell Name'].unique()
        
        # Allow only single cell name selection
        selected_cell = st.selectbox("Select Cell Name", unique_cell_names)
        
        # Filter data by selected cell name
        filtered_data = data[data['Cell Name'] == selected_cell]
    else:
        st.warning("Column 'Cell Name' not found in the uploaded data.")
        filtered_data = data

    # Step 3: Choose Prediction Type
    prediction_type = st.selectbox("Choose Prediction Type", ["Univariable", "Multivariable"])

    if prediction_type == "Univariable":
        # Univariable Prediction Configuration
        model_type = st.selectbox("Choose Model", ["ARIMA", "SARIMA"])
        target_column = st.selectbox("Field to forecast", filtered_data.columns)

        # ARIMA/SARIMA Parameters
        st.subheader(f"{model_type} Forecasting Configuration")
        p = st.number_input("AR (autoregressive) - p", min_value=0, step=1, value=1)
        d = st.number_input("I (integrated) - d", min_value=0, step=1, value=1)
        q = st.number_input("MA (moving average) - q", min_value=0, step=1, value=1)

        # Additional SARIMA seasonal parameter
        if model_type == "SARIMA":
            s = st.number_input("Seasonal period - s", min_value=1, step=1, value=12)

        future_steps = st.number_input("Future Timespan", min_value=1, step=1, value=5)
        conf_interval = st.slider("Confidence Interval", min_value=0, max_value=100, value=95)

        # Start Predict Button for Univariable
        if st.button("Start Predict"):
            st.write("Starting univariable prediction with the selected configuration...")
            # Here would be the prediction code for ARIMA/SARIMA

    elif prediction_type == "Multivariable":
        # Multivariable Prediction Configuration
        algorithm = st.selectbox("Choose Model", ["SVR", "Decision Tree", "Gradient Boosting", "Random Forest", "XGBoost", "LSTM"])
        target_column = st.selectbox("Field to predict", filtered_data.columns)

        # Feature Selection with "Select All" Option
        all_feature_columns = [col for col in filtered_data.columns if col != target_column]
        select_all_features = st.checkbox("Select All Feature Columns")
        feature_columns = all_feature_columns if select_all_features else st.multiselect("Fields to use for predicting", all_feature_columns)

        # Training/Test Split
        test_split = st.slider("Split for training/test", 0.1, 0.9, 0.3)

        # Start Predict Button for Multivariable
        if st.button("Start Predict"):
            st.write("Starting multivariable prediction with the selected configuration...")
            # Here would be the prediction code for the chosen algorithm

# Instructions if no file is uploaded
else:
    st.write("Please upload a CSV file to get started.")

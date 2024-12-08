# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import warnings

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

    # Step 2: Merge 'Date' and 'Time' into 'Datetime'
    if 'Date' in data.columns and 'Time' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%m/%d/%Y %H:%M')
        data = data.set_index('Datetime')
    else:
        st.warning("Columns 'Date' or 'Time' are not found in the dataset.")

    # Step 3: Select Cell Name (Single Selection)
    if 'Cell Name' in data.columns:
        unique_cell_names = data['Cell Name'].unique()

        # Allow only single cell name selection
        selected_cell = st.selectbox("Select Cell Name", unique_cell_names)

        # Filter data by selected cell name
        filtered_data = data[data['Cell Name'] == selected_cell]
        
        #st.write(f"Filtered Data for Cell: {selected_cell}", filtered_data.head())
    else:
        st.warning("Column 'Cell Name' not found in the dataset.")
        filtered_data = data

    # Step 4: Filter data based on the time range
    start_date = '2024-02-02 00:00:00'
    end_date = '2024-04-02 00:00:00'
        
    filtered_data = filtered_data.loc[start_date:end_date]
    st.write("Filtered Data:", filtered_data.head())

    # Step 5: Preprocessing: Hanya ambil kolom numerik dan isi nilai kosong dengan mean
    filtered_data = filtered_data.select_dtypes(include=[np.number])
    filtered_data.fillna(filtered_data.mean(), inplace=True)


    # Step 5: Choose Prediction Type
    prediction_type = st.selectbox("Choose Prediction Type", ["Multivariable", "Univariable"])

    if prediction_type == "Multivariable":
        # Multivariable Prediction Configuration
        algorithm = st.selectbox("Choose Model", ["Random Forest", "Decision Tree", "SVR", "Gradient Boosting", "XGBoost", "LSTM"])
        target_column = st.selectbox("Field to predict", filtered_data.columns)

        # Feature Selection with "Select All" Option
        all_feature_columns = [col for col in filtered_data.columns if col != target_column]
        select_all_features = st.checkbox("Select All Feature Columns")
        feature_columns = all_feature_columns if select_all_features else st.multiselect("Fields to use for predicting", all_feature_columns)

        # Training/Test Split
        test_split = st.slider("Split for training/test", 0.1, 0.9, 0.3)

        # Start Predict Button for Multivariable
        if st.button("Start Predict"):
            st.write("Starting multivariable prediction with", algorithm)
            # Here would be the prediction code for the chosen algorithm

            # Prepare data for prediction
            filtered_data['Hour'] = filtered_data.index.hour
            filtered_data['Day'] = filtered_data.index.day
            filtered_data['Month'] = filtered_data.index.month

            # Lag features
            filtered_data[target_column,'_lag1'] = filtered_data[target_column].shift(1)
            filtered_data[target_column,'_lag2'] = filtered_data[target_column].shift(2)
            filtered_data[target_column,'_lag3'] = filtered_data[target_column].shift(3)
            filtered_data = filtered_data.dropna()

            # Set predictor and target columns
            predictor_columns = [col for col in filtered_data.columns if col != target_column]
            X = filtered_data[predictor_columns]
            y = filtered_data[target_column]

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)


            # Random Forest Algorithm
            if algorithm == "Random Forest":

                # Pipeline with RandomForestRegressor
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('random_forest', RandomForestRegressor(random_state=42))
                ])

                # Hyperparameter tuning for Random Forest
                param_grid = {
                    'random_forest__n_estimators': [50, 100, 150],
                    'random_forest__max_depth': [5, 10, 15],
                    'random_forest__min_samples_split': [2, 5, 10]
                }

                # Cross-validation with TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Best model
                best_model = grid_search.best_estimator_

                # Model Prediction
                y_pred = best_model.predict(X_test)

            # Decision Tree Algorithm
            if algorithm == "Decision Tree":

                # Pipeline with StandardScaler dan DecisionTreeRegressor
                pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('dtr', DecisionTreeRegressor(random_state=42))
                ])

                # Hyperparameter tuning untuk DecisionTreeRegressor
                param_grid = {
                    'dtr__max_depth': [5, 10, 15, None],       # Kedalaman maksimum pohon
                    'dtr__min_samples_split': [2, 5, 10],      # Jumlah minimum sampel untuk memisahkan node
                    'dtr__min_samples_leaf': [1, 2, 4]         # Jumlah minimum sampel pada setiap daun
                }

                # Cross-validation with TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Best model
                best_model = grid_search.best_estimator_

                # Model Prediction
                y_pred = best_model.predict(X_test)
            
            # SVR Algorithm
            if algorithm == "SVR":
                # Membuat pipeline dengan StandardScaler dan SVR
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svr', SVR())
                ])

                # Menentukan grid parameter yang lebih luas untuk SVR
                param_grid = {
                    'svr__C': [1, 0.1, 1, 10, 1],        # Rentang nilai C yang lebih luas
                    'svr__epsilon': [0.1, 0.5],  # Rentang epsilon yang lebih luas
                    'svr__kernel': ['linear', 'rbf'],         # Kernel 'linear' dan 'rbf'
                    'svr__gamma': ['scale', 'auto']           # Gamma untuk kernel 'rbf'
                }
                
                # Cross-validation with TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Best model
                best_model = grid_search.best_estimator_

                # Model Prediction
                y_pred = best_model.predict(X_test)

            # Gradient Boosting Algorithm
            if algorithm == "Gradient Boosting":
                # Membuat pipeline dengan StandardScaler dan GradientBoostingRegressor
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('gbr', GradientBoostingRegressor())
                ])

                # Hyperparameter tuning untuk GradientBoostingRegressor
                param_grid = {
                    'gbr__n_estimators': [50, 100, 150],        # Jumlah estimator
                    'gbr__learning_rate': [0.01, 0.1, 0.2],     # Laju pembelajaran
                    'gbr__max_depth': [3, 5, 7]                 # Kedalaman maksimum pohon
                }

                # Cross-validation with TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Best model
                best_model = grid_search.best_estimator_

                # Model Prediction
                y_pred = best_model.predict(X_test)

            # XGBoost Algorithm
            if algorithm == "XGBoost":
                # Membuat pipeline dengan StandardScaler dan XGBRegressor
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('xgbr', XGBRegressor(objective='reg:squarederror', random_state=42))
                ])

                # Hyperparameter tuning untuk XGBRegressor
                param_grid = {
                    'xgbr__n_estimators': [50, 100, 150],       # Jumlah estimator
                    'xgbr__learning_rate': [0.01, 0.1, 0.2],    # Laju pembelajaran
                    'xgbr__max_depth': [3, 5, 7],               # Kedalaman pohon
                    'xgbr__subsample': [0.8, 1.0]               # Subsample
                }

                # Cross-validation with TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                grid_search.fit(X_train, y_train)

                # Best model
                best_model = grid_search.best_estimator_

                # Model Prediction
                y_pred = best_model.predict(X_test)

            # Model Evaluation (For RF, DT, SVR, GradBoost, and XGBoost)
            if algorithm != "LSTM":
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                msle = mean_squared_log_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            # LSTM Algorithm
            if algorithm == "LSTM":
                # Normalisasi data menggunakan StandardScaler
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                X = scaler_X.fit_transform(X)
                y = scaler_y.fit_transform(y.values.reshape(-1, 1))

                # Membagi data menjadi train dan test set tanpa shuffle
                train_size = int(len(X) * 0.7)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Reshape X agar sesuai dengan input yang diperlukan LSTM [samples, timesteps, features]
                X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                # Membangun model LSTM
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(LSTM(units=50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Menggunakan EarlyStopping untuk mencegah overfitting
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                # Melatih model
                history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

                # Prediksi menggunakan model
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Membalikkan normalisasi pada prediksi dan data sebenarnya
                y_train = scaler_y.inverse_transform(y_train)
                y_test = scaler_y.inverse_transform(y_test)
                y_pred_train = scaler_y.inverse_transform(y_pred_train)
                y_pred_test = scaler_y.inverse_transform(y_pred_test)

                # Evaluasi model pada data uji
                mse = mean_squared_error(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                msle = mean_squared_log_error(y_test, y_pred_test)
                mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

            # Display evaluation metrics
            st.write(f"\nEvaluation metrics for {target_column} using", algorithm, ":")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"R² Score: {r2:.4f}")
            st.write(f"Mean Squared Logarithmic Error (MSLE): {msle:.4f}")
            st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

            
            # Plotting Actual vs Predicted (For RF, DT, SVR, GradBoost, and XGBoost)
            plt.figure(figsize=(12, 6))
            # For RF, DT, SVR, GradBoost, and XGBoost
            if algorithm != "LSTM":
                plt.plot(y_test.index, y_test, label='Actual Data', color='blue')
                plt.plot(y_test.index, y_pred, label='Predicted Data', color='red', linestyle='--')
            # For LSTM
            if algorithm == "LSTM":
                plt.plot(y_test, label='Actual Date', color='blue')
                plt.plot(y_pred_test, label='Predicted', color='red', linestyle='--')
            plt.title(f"Actual vs Predicted {target_column} using {algorithm}")
            plt.xlabel("Datetime")
            plt.ylabel(target_column)
            plt.legend()
            st.pyplot(plt)

    elif prediction_type == "Univariable":
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

# Instructions if no file is uploaded
else:
    st.write("Please upload a CSV file to get started.")

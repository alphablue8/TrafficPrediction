import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
#from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import warnings

# Set page configuration
st.set_page_config(layout="wide")

# Title
st.title("Cellular Network Traffic Prediction System")

# Sidebar: Upload CSV File
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:

    # Load and display data
    data = pd.read_csv(uploaded_file)

    # Preprocessing: Merge 'Date' and 'Time' into 'Datetime'
    if 'Date' in data.columns and 'Time' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%m/%d/%Y %H:%M')
        data = data.set_index('Datetime')

    # Preprocessing: Filter data based on the time range
    start_date = '2024-02-02 00:00:00'
    end_date = '2024-04-02 00:00:00'
    data = data.loc[start_date:end_date]

    # Sidebar menu
    menu = st.sidebar.radio("Menu", ["Traffic Prediction", "Data Visualization"])

    if menu == "Traffic Prediction":
        # Sidebar: Prediction Configuration
        if 'Cell Name' in data.columns:
            unique_cell_names = data['Cell Name'].unique()
            selected_cell = st.sidebar.selectbox("Select Cell Name", unique_cell_names)
            filtered_data = data[data['Cell Name'] == selected_cell]
        else:
            st.warning("Column 'Cell Name' not found in the dataset.")
            filtered_data = data
        
        # Display Filtered Data
        st.write(f"### Filtered Data for Selected Cell", filtered_data.head())
        
        # Step 5: Preprocessing: Hanya ambil kolom numerik
        filtered_data = filtered_data.select_dtypes(include=[np.number])

        # Sidebar: Choose Prediction Type
        prediction_type = st.sidebar.selectbox("Choose Prediction Type", ["Deep Learning", "Machine Learning", "Conventional"])

        if prediction_type == "Deep Learning":
             # Deep Learning Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["LSTM", "GRU"])
            target_column = st.sidebar.selectbox("Field to predict", filtered_data.select_dtypes(include=[np.number]).columns)

            feature_columns = [col for col in filtered_data.columns if col != target_column]
            
            # Training/Test Split
            test_split = st.sidebar.slider("Split for training/test", 0.1, 0.9, 0.3)

            if st.sidebar.button("Start Predict"):
                st.sidebar.write("Starting Deep Learning prediction with", algorithm, "...")
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
                X = filtered_data[feature_columns]
                y = filtered_data[target_column]

                # Train/Test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)

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
                
                # LSTM Algorithm
                if algorithm == "GRU":
                    # Normalisasi data menggunakan StandardScaler
                    scaler_X = StandardScaler()
                    scaler_y = StandardScaler()
                    X = scaler_X.fit_transform(X)
                    y = scaler_y.fit_transform(y.values.reshape(-1, 1))

                    # Membagi data menjadi train dan test set tanpa shuffle
                    train_size = int(len(X) * 0.7)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    # Reshape X agar sesuai dengan input yang diperlukan GRU [samples, timesteps, features]
                    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
                    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

                    # Membangun model GRU
                    model = Sequential()
                    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                    model.add(GRU(units=50))
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
                col1, col2 = st.columns(2)
                # Left column for MSE and MAE
                with col1:
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Squared Error (MSE):</strong> {mse:.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Absolute Error (MAE):</strong> {mae:.4f}</div>", unsafe_allow_html=True)

                # Right column for MSLE and MAPE
                with col2:
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Squared Logarithmic Error (MSLE):</strong> {msle:.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Absolute Percentage Error (MAPE):</strong> {mape:.2f}%</div>", unsafe_allow_html=True)

                # Center R² Score at the bottom
                st.markdown(f"<div style='text-align: center; margin-top: 20px;'><strong>R² Score:</strong> {r2:.4f}</div>", unsafe_allow_html=True)
                st.write(" ")
                
                # Plotting Actual vs Predicted (For LSTM and GRU)
                plt.figure(figsize=(12, 6))
                plt.plot(y_test, label='Actual Date', color='blue')
                plt.plot(y_pred_test, label='Predicted', color='red', linestyle='--')
                plt.title(f"Actual vs Predicted {target_column} using {algorithm}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column)
                plt.legend()
                st.pyplot(plt)


        if prediction_type == "Machine Learning":
            # Machine Learning Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["Random Forest", "Decision Tree", "KNN", "SVR", "Gradient Boosting", "XGBoost"])
            target_column = st.sidebar.selectbox("Field to predict", filtered_data.columns)

            feature_columns = [col for col in filtered_data.columns if col != target_column]

            # Training/Test Split
            test_split = st.sidebar.slider("Split for training/test", 0.1, 0.9, 0.3)

            # Start Predict Button for Machine Learning
            if st.sidebar.button("Start Predict"):
                st.sidebar.write("Starting Machine Learning prediction with", algorithm, "...")
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
                X = filtered_data[feature_columns]
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
                
                # KNN Regressor Algorithm
                if algorithm == "KNN":
                    # Membuat pipeline dengan StandardScaler dan KNeighborsRegressor
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('knn', KNeighborsRegressor())
                    ])

                    # Hyperparameter tuning untuk KNeighborsRegressor
                    param_grid = {
                        'knn__n_neighbors': [3, 5, 10],        # Jumlah tetangga terdekat
                        'knn__weights': ['uniform', 'distance'],  # Bobot jarak
                        'knn__p': [1, 2]                        # Jenis jarak (1 = Manhattan, 2 = Euclidean)
                    }

                    # Menggunakan TimeSeriesSplit untuk cross-validation
                    tscv = TimeSeriesSplit(n_splits=5)

                    # GridSearchCV untuk hyperparameter tuning
                    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                    grid_search.fit(X_train, y_train)

                    # Model terbaik berdasarkan GridSearchCV
                    best_model = grid_search.best_estimator_

                    # Evaluasi model pada data uji
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
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                msle = mean_squared_log_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                # Display evaluation metrics
                
                st.write(f"\nEvaluation metrics for {target_column} using", algorithm, ":")
                col1, col2 = st.columns(2)
                # Left column for MSE and MAE
                with col1:
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Squared Error (MSE):</strong> {mse:.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Absolute Error (MAE):</strong> {mae:.4f}</div>", unsafe_allow_html=True)

                # Right column for MSLE and MAPE
                with col2:
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Squared Logarithmic Error (MSLE):</strong> {msle:.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Absolute Percentage Error (MAPE):</strong> {mape:.2f}%</div>", unsafe_allow_html=True)

                # Center R² Score at the bottom
                st.markdown(f"<div style='text-align: center; margin-top: 20px;'><strong>R² Score:</strong> {r2:.4f}</div>", unsafe_allow_html=True)
                st.write(" ")

                # Plotting Actual vs Predicted (For RF, DT, SVR, GradBoost, and XGBoost)
                plt.figure(figsize=(12, 6))
                plt.plot(y_test.index, y_test, label='Actual Data', color='blue')
                plt.plot(y_test.index, y_pred, label='Predicted Data', color='red', linestyle='--')
                plt.title(f"Actual vs Predicted {target_column} using {algorithm}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column)
                plt.legend()
                st.pyplot(plt)

        if prediction_type == "Conventional":
            # Conventional Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["ARIMA", "SARIMA"])
            target_column = st.sidebar.selectbox("Field to predict", filtered_data.columns)

            # Start Predict Button for Conventional Prediction
            if st.sidebar.button("Start Predict"):
                st.sidebar.write("Starting Machine Learning prediction with", algorithm, "...")
                y = filtered_data[target_column]
                # Memisahkan data menjadi set pelatihan dan pengujian
                train_size = int(len(y) * 0.7)
                y_train, y_test = y[:train_size], y[train_size:]
                # Random Forest Algorithm
                if algorithm == "ARIMA":
                    # Menentukan parameter ARIMA (p, d, q)
                    p, d, q = 1, 1, 1  # Pencarian nilai p, d, q yang tepat dapat dilakukan dengan trial-and-error atau grid search

                    # Membangun model ARIMA
                    arima_model = ARIMA(y_train, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
                    arima_fit = arima_model.fit()

                    # Prediksi pada data uji
                    y_pred = arima_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False)
                if algorithm == "SARIMA":
                    # Menentukan parameter SARIMA
                    # (p, d, q) untuk ARIMA, dan (P, D, Q, S) untuk komponen musiman
                    p, d, q = 1, 1, 2         # Tentukan parameter ARIMA yang sesuai
                    P, D, Q, S = 1, 1, 1, 24  # P, D, Q untuk musiman dan S untuk periodisitas musiman (misalnya 24 untuk data harian per jam)

                    # Membangun model SARIMA
                    sarima_model = SARIMAX(y_train, order=(p, d, q), seasonal_order=(P, D, Q, S), enforce_stationarity=False, enforce_invertibility=False)
                    sarima_fit = sarima_model.fit(disp=False)

                    # Prediksi pada data uji
                    y_pred = sarima_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False)

                # Evaluasi model pada data uji
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                msle = mean_squared_log_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                # Display evaluation metrics
                st.write(f"\nEvaluation metrics for {target_column} using", algorithm, ":")
                col1, col2 = st.columns(2)
                # Left column for MSE and MAE
                with col1:
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Squared Error (MSE):</strong> {mse:.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Absolute Error (MAE):</strong> {mae:.4f}</div>", unsafe_allow_html=True)

                # Right column for MSLE and MAPE
                with col2:
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Squared Logarithmic Error (MSLE):</strong> {msle:.4f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center;'><strong>Mean Absolute Percentage Error (MAPE):</strong> {mape:.2f}%</div>", unsafe_allow_html=True)

                # Center R² Score at the bottom
                st.markdown(f"<div style='text-align: center; margin-top: 20px;'><strong>R² Score:</strong> {r2:.4f}</div>", unsafe_allow_html=True)
                st.write(" ")

                # Plotting Actual vs Predicted (For RF, DT, SVR, GradBoost, and XGBoost)
                plt.figure(figsize=(12, 6))
                plt.plot(y_test.index, y_test, label='Actual Data', color='blue')
                plt.plot(y_test.index, y_pred, label='Predicted Data', color='red', linestyle='--')
                plt.title(f"Actual vs Predicted {target_column} using {algorithm}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column)
                plt.legend()
                st.pyplot(plt)
        
    elif menu == "Data Visualization":
        st.subheader("Data Visualization")

        # Sidebar: Visualization Configuration
        if 'Cell Name' in data.columns:
            unique_cell_names = data['Cell Name'].unique()
            selected_cell_vis = st.sidebar.selectbox("Select Cell Name for Visualization", unique_cell_names)
            data_vis = data[data['Cell Name'] == selected_cell_vis]
        else:
            st.warning("Column 'Cell Name' not found in the dataset.")
            data_vis = data  

        # Display Filtered Data
        st.write(f"### Filtered Data for Selected Cell", data_vis.head())

        target_column_vis = st.sidebar.selectbox("Select Column to Visualize", data_vis.select_dtypes(include=[np.number]).columns)
    
        st.write("### Time Series Visualization")
        plt.figure(figsize=(20, 5))
        plt.plot(data_vis.index, data_vis[target_column_vis])
        plt.title(f"Time Series of {target_column_vis} for {selected_cell_vis}")
        plt.xlabel("Datetime")
        plt.ylabel(target_column_vis)
        st.pyplot(plt)


else:
    st.write("Please upload a CSV file to get started.")

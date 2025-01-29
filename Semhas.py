import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import warnings
import time

# Set page configuration
st.set_page_config(layout="wide")

# Title
st.title("Cellular Network Traffic Prediction System")

# Sidebar: Upload CSV File
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:

    # Load and display data
    data = pd.read_csv(uploaded_file)

    # Daftar kolom yang ingin dikecualikan dari konversi
    excluded_columns = ['4G Avg UL Interference','Integrity','4G RSSI (Cells)', 'Date', 'Time', 'eNodeB Name', 'Cell Name']

    # Mengubah semua kolom kecuali kolom yang dikecualikan menjadi tipe numerik
    for col in data.columns:
        if col not in excluded_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Mengubah kolom '4G RSSI (Cells)' menjadi tipe object
    data['4G RSSI (Cells)'] = data['4G RSSI (Cells)'].astype(object)

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
        st.write(f"### Filtered Data for Selected Cell")

        # Show table with all data but initial display is limited to a scrollable view
        st.dataframe(filtered_data, height=200)  # Adjust the height to limit visible rows
        
        # Display number of rows and columns
        num_rows, num_cols = filtered_data.shape
        st.caption(f"Jumlah baris: {num_rows}, Jumlah kolom: {num_cols}")
        
        # Step 5: Preprocessing: Hanya ambil kolom numerik
        filtered_data = filtered_data.select_dtypes(include=[np.number])
        filtered_data.fillna(filtered_data.mean(), inplace=True)

        # Sidebar: Choose Prediction Type
        prediction_type = st.sidebar.selectbox("Choose Prediction Type", ["Deep Learning", "Machine Learning", "Statistic"])

        if prediction_type == "Deep Learning":
             # Deep Learning Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["LSTM", "GRU"])
            target_column = st.sidebar.selectbox("Field to predict", filtered_data.select_dtypes(include=[np.number]).columns)

            feature_columns = [col for col in filtered_data.columns if col != target_column]
            
            # Training/Test Split
            test_split = st.sidebar.slider("Split for test/training", 0.1, 0.9, 0.3)

            if st.sidebar.button("Start Predict"):
                # Catat waktu mulai
                start_time = time.time()
                progress = st.progress(0)
                with st.spinner(f"Starting Deep Learning Prediction with {algorithm}..."):
                    # Update progress to 20%
                    progress.progress(20)
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
                    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)

                    progress.progress(40)  # Update progress to 60%

                    # LSTM Algorithm
                    if algorithm == "LSTM":
                        # Normalisasi data menggunakan StandardScaler
                        scaler_X = StandardScaler()
                        scaler_y = StandardScaler()
                        X = scaler_X.fit_transform(X)
                        y = scaler_y.fit_transform(y.values.reshape(-1, 1))

                        # Membagi data menjadi train dan test set tanpa shuffle
                        train_size = int(len(X) * (1 - test_split))
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

                        progress.progress(60)  # Update progress to 100%

                        # Menggunakan EarlyStopping untuk mencegah overfitting
                        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                        # Melatih model
                        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

                        progress.progress(80)  # Update progress to 100%

                        # Prediksi untuk 3 hari ke depan (72 jam)
                        future_steps = 72
                        last_known_values = X_test[-1].reshape(1, 1, -1)  # Pastikan input memiliki dimensi [1, 1, features]
                        predictions = []

                        for _ in range(future_steps):
                            # Prediksi nilai baru
                            prediction = model.predict(last_known_values, verbose=0)[0][0]
                            predictions.append(prediction)

                            # Update input dengan nilai prediksi untuk iterasi berikutnya
                            new_input = np.append(last_known_values[0, 0, 1:], prediction).reshape(1, 1, -1)
                            last_known_values = new_input

                        # Membalikkan normalisasi pada prediksi
                        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

                        # Membuat DataFrame untuk prediksi
                        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='H')[1:]
                        future_df = pd.DataFrame({
                            'Datetime': future_dates,
                            'Predicted 4G Total Traffic (GB)': predictions.flatten()
})

                        # Prediksi menggunakan model
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        # Membalikkan normalisasi pada prediksi dan data sebenarnya
                        y_train = scaler_y.inverse_transform(y_train)
                        y_test = scaler_y.inverse_transform(y_test)
                        y_pred_train = scaler_y.inverse_transform(y_pred_train)
                        y_pred_test = scaler_y.inverse_transform(y_pred_test)
                    
                        progress.progress(100)  # Update progress to 100%

                    # GRU Algorithm
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

                        progress.progress(60)  # Update progress to 60%

                        # Menggunakan EarlyStopping untuk mencegah overfitting
                        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

                        # Melatih model
                        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

                        progress.progress(80)  # Update progress to 100%
                        # Prediksi untuk 3 hari ke depan (72 jam)
                        future_steps = 72
                        last_known_values = X_test[-1].reshape(1, 1, -1)  # Pastikan input memiliki dimensi [1, 1, features]
                        predictions = []

                        for _ in range(future_steps):
                            # Prediksi nilai baru
                            prediction = model.predict(last_known_values, verbose=0)[0][0]
                            predictions.append(prediction)

                            # Update input dengan nilai prediksi untuk iterasi berikutnya
                            new_input = np.append(last_known_values[0, 0, 1:], prediction).reshape(1, 1, -1)
                            last_known_values = new_input

                        # Membalikkan normalisasi pada prediksi
                        predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))

                        # Membuat DataFrame untuk prediksi
                        future_dates = pd.date_range(start=data.index[-1], periods=future_steps + 1, freq='H')[1:]
                        future_df = pd.DataFrame({
                            'Datetime': future_dates,
                            'Predicted 4G Total Traffic (GB)': predictions.flatten()
})

                        # Prediksi menggunakan model
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        # Membalikkan normalisasi pada prediksi dan data sebenarnya
                        y_train = scaler_y.inverse_transform(y_train)
                        y_test = scaler_y.inverse_transform(y_test)
                        y_pred_train = scaler_y.inverse_transform(y_pred_train)
                        y_pred_test = scaler_y.inverse_transform(y_pred_test)
                        
                        progress.progress(100)  # Update progress to 100%
                
                st.success("Prediction complete!")
                # Catat waktu selesai
                end_time = time.time()

                # Hitung durasi
                duration = end_time - start_time

                # Tampilkan hasil prediksi dan waktu proses
                st.write(f"Time taken for prediction: {duration:.2f} seconds")
                    
                # Evaluasi model pada data uji
                mse = mean_squared_error(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                r2 = r2_score(y_test, y_pred_test)
                msle = mean_squared_log_error(y_test, y_pred_test)
                mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

                # Display metrics
                st.markdown(f"### Evaluation metrics for {target_column} using {algorithm} in {selected_cell}:")
                col1, col2, col3, col4, col5 = st.columns(5)  # Menambahkan satu kolom lagi
                with col1:
                    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                with col2:
                    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                with col3:
                    st.metric(label="R² Score", value=f"{r2:.4f}")
                with col4:
                    st.metric(label="MSLE", value=f"{msle:.4f}")
                with col5:
                    st.metric(label="MAPE", value=f"{mape:.2f}%")
                
                # Pastikan akses ke index asli sebelum split
                original_index = filtered_data.index[-len(y_test):]  # Ambil index data uji (y_test)

                # Kolom untuk Actual vs Predicted dan Prediksi 3 Hari ke Depan
                col1, col2 = st.columns(2)

                with col1:
                    plt.figure(figsize=(12, 6))  # Adjusted size for column layout
                    plt.plot(original_index, y_test, label='Actual', color='blue')
                    plt.plot(original_index, y_pred_test, label='Predicted', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} using {algorithm}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                with col2:
                    plt.figure(figsize=(12, 6))  # Adjusted size for column layout
                    plt.plot(original_index[-120:], y_test[-120:], label='Data Sebenarnya', color='blue')
                    plt.plot(original_index[-120:], y_pred_test[-120:], label='Data Prediksi', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} (5 Hari Terakhir) using {algorithm}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                # Kolom untuk Actual vs Predicted (5 Hari Terakhir) dan tabel komparasi
                col3, col4 = st.columns(2)

                with col3:                
                    st.write(f"### Prediksi 3 Hari ke Depan 4G Total Traffic in {selected_cell}")
                    plt.figure(figsize=(12, 6))  # Adjusted size for column layout
                    plt.plot(future_df['Datetime'], future_df['Predicted 4G Total Traffic (GB)'], label='Prediksi', color='green')
                    plt.title("Prediksi 3 Hari ke Depan: 4G Total Traffic (GB)")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                with col4:
                    # Tambahkan kembali identitas data ke dalam DataFrame
                    comparison_df = pd.DataFrame({
                        "Datetime": original_index,  # Gunakan index asli dari data uji
                        "Cell Name": selected_cell,  # Cell yang dipilih oleh pengguna
                        "Actual": y_test.flatten(),  # Data aktual
                        "Predicted": y_pred_test.flatten(),  # Prediksi
                        "Difference": y_test.flatten() - y_pred_test.flatten()  # Selisih antara Actual dan Predicted
                    })

                    # Filter tabel berdasarkan cell
                    comparison_df_filtered = comparison_df[comparison_df["Cell Name"] == selected_cell]

                    # Tampilkan tabel
                    st.write(f"### Actual vs Predicted Data for Cell: {selected_cell}")
                    st.dataframe(comparison_df_filtered)

                    # Tambahkan tombol untuk mengunduh tabel
                    csv = comparison_df_filtered.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Actual vs Predicted Data for {selected_cell} as CSV",
                        data=csv,
                        file_name=f'actual_vs_predicted_{selected_cell}.csv',
                        mime='text/csv',
                    )
                
        if prediction_type == "Machine Learning":
            # Machine Learning Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", [ "Random Forest", "Decision Tree", "KNN", "XGBoost"])
            target_column = st.sidebar.selectbox("Field to predict", filtered_data.columns)

            feature_columns = [col for col in filtered_data.columns if col != target_column]

            # Training/Test Split
            test_split = st.sidebar.slider("Split for test/training", 0.1, 0.9, 0.3)

            # Start Predict Button for Machine Learning
            if st.sidebar.button("Start Predict"):
                # Catat waktu mulai
                start_time = time.time()
                progress = st.progress(0)
                with st.spinner(f"Starting Machine Learning Prediction with {algorithm}..."):
                    # Update progress to 20%
                    progress.progress(20)
                    # Prepare data for prediction
                    filtered_data['Hour'] = filtered_data.index.hour
                    filtered_data['Day'] = filtered_data.index.day
                    filtered_data['Month'] = filtered_data.index.month

                    for lag in range(1, 4):
                        filtered_data[f"{target_column}_lag{lag}"] = filtered_data[target_column].shift(lag)

                    filtered_data = filtered_data.dropna()

                    # Set predictor and target columns
                    X = filtered_data[feature_columns]
                    y = filtered_data[target_column]

                    # Train/Test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, shuffle=False)

                    progress.progress(40)  # Update progress to 40%

                    # Random Forest Algorithm
                    if algorithm == "Random Forest":

                        # Pipeline with RandomForestRegressor
                        pipeline = Pipeline([
                            ('scaler', StandardScaler()),
                            ('random_forest', RandomForestRegressor(random_state=42))
                        ])

                        # Hyperparameter tuning for Random Forest
                        param_grid = {
                            'random_forest__n_estimators': [50, 100],
                            'random_forest__max_depth': [10, 20],
                            'random_forest__min_samples_split': [2, 5],
                        }

                        # Cross-validation with TimeSeriesSplit
                        tscv = TimeSeriesSplit(n_splits=5)
                        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                        grid_search.fit(X_train, y_train)

                        progress.progress(60)  # Update progress to 60%

                        # Best model
                        best_model = grid_search.best_estimator_

                        progress.progress(80)  # Update progress to 80%

                        # Model Prediction
                        y_pred = best_model.predict(X_test)

                        progress.progress(100)  # Update progress to 100%

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

                        progress.progress(60)  # Update progress to 60

                        # Best model
                        best_model = grid_search.best_estimator_

                        progress.progress(80)  # Update progress to 100%

                        # Model Prediction
                        y_pred = best_model.predict(X_test)

                        progress.progress(100)  # Update progress to 100%
                    
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

                        progress.progress(60)  # Update progress to 60%

                        # Model terbaik berdasarkan GridSearchCV
                        best_model = grid_search.best_estimator_

                        progress.progress(80)  # Update progress to 100%

                        # Evaluasi model pada data uji
                        y_pred = best_model.predict(X_test)

                        progress.progress(100)  # Update progress to 100%
                    
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
                        grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-2)
                        grid_search.fit(X_train, y_train)

                        progress.progress(60)  # Update progress to 60%

                        # Best model
                        best_model = grid_search.best_estimator_

                        progress.progress(80)  # Update progress to 80%

                        # Model Prediction
                        y_pred = best_model.predict(X_test)

                        progress.progress(100)  # Update progress to 100%

                st.success("Prediction complete!")
                # Catat waktu selesai
                end_time = time.time()

                # Hitung durasi
                duration = end_time - start_time

                # Tampilkan hasil prediksi dan waktu proses
                st.write(f"Time taken for prediction: {duration:.2f} seconds")

                # Model Evaluation (For RF, DT, LGBoost and XGBoost)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                msle = mean_squared_log_error(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                # Display metrics
                st.markdown(f"### Evaluation metrics for {target_column} using {algorithm} in {selected_cell}:")
                col1, col2, col3, col4, col5 = st.columns(5)  # Menambahkan satu kolom lagi
                with col1:
                    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                with col2:
                    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                with col3:
                    st.metric(label="R² Score", value=f"{r2:.4f}")
                with col4:
                    st.metric(label="MSLE", value=f"{msle:.4f}")
                with col5:
                    st.metric(label="MAPE", value=f"{mape:.2f}%")

                # Pastikan akses ke index asli sebelum split
                original_index = filtered_data.index[-len(y_test):]  # Ambil index data uji (y_test)

                # Kolom untuk Actual vs Predicted dan Prediksi 3 Hari ke Depan
                col1, col2 = st.columns(2)

                with col1:
                    # Plotting Actual vs Predicted (For LR, RF, DT, and XGBoost)
                    plt.figure(figsize=(12, 6))
                    plt.plot(y_test.index, y_test, label='Actual Data', color='blue')
                    plt.plot(y_test.index, y_pred, label='Predicted Data', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} using {algorithm} in {selected_cell}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                with col2:
                    # Plot perbandingan antara data sebenarnya dan prediksi (default 120 jam terakhir)
                    plt.figure(figsize=(12, 6))
                    plt.plot(original_index[-120:], y_test[-120:], label='Data Sebenarnya', color='blue')
                    plt.plot(original_index[-120:], y_pred[-120:], label='Data Prediksi', color='red', linestyle='--')
                    plt.title(f"Actual vs Predicted {target_column} (120 Jam Terakhir) using {algorithm} in {selected_cell}")
                    plt.xlabel("Datetime")
                    plt.ylabel(target_column)
                    plt.legend()
                    st.pyplot(plt)

                # Tambahkan kembali identitas data ke dalam DataFrame
                comparison_df = pd.DataFrame({
                    "Datetime": filtered_data.index[-len(y_test):],  # Ambil tanggal dari data asli untuk subset data uji
                    "Cell Name": selected_cell,  # Cell yang dipilih
                    "Actual": y_test.values,  # Data aktual
                    "Predicted": y_pred,  # Data prediksi
                    "Difference": y_test.values - y_pred  # Selisih antara actual dan predicted
                })

                # Tampilkan tabel hasil
                st.write(f"### Actual vs Predicted Data for Cell: {selected_cell}")
                st.dataframe(comparison_df)

                # Tambahkan tombol untuk mengunduh tabel
                csv = comparison_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Actual vs Predicted Data for {selected_cell} as CSV",
                    data=csv,
                    file_name=f'actual_vs_predicted_{selected_cell}.csv',
                    mime='text/csv',
                )

                # Mencetak parameter terbaik dari grid search
                st.write("Parameter terbaik:", grid_search.best_params_)


        if prediction_type == "Statistic":
            # Statistic Prediction Configuration
            algorithm = st.sidebar.selectbox("Choose Model", ["SARIMA"])
            target_column = st.sidebar.selectbox("Field to predict", filtered_data.columns)

            # Start Predict Button for Statistic Prediction
            if st.sidebar.button("Start Predict"):
                # Catat waktu mulai
                start_time = time.time()
                st.sidebar.write("Starting Statistic prediction with", algorithm, "...")
                progress = st.progress(0)
                with st.spinner(f"Starting Statistic prediction with {algorithm}..."):

                    # Update progress to 20%
                    progress.progress(20)

                    if algorithm == "SARIMA":
                        y = filtered_data[target_column]
                        # Memisahkan data menjadi set pelatihan dan pengujian
                        train_size = int(len(y) * 0.7)
                        y_train, y_test = y[:train_size], y[train_size:]

                        progress.progress(40)  # Update progress to 40%
                        # Menentukan parameter SARIMA
                        # (p, d, q) untuk ARIMA, dan (P, D, Q, S) untuk komponen musiman
                        p, d, q = 1, 1, 2         # Tentukan parameter ARIMA yang sesuai
                        P, D, Q, S = 1, 1, 1, 24  # P, D, Q untuk musiman dan S untuk periodisitas musiman (misalnya 24 untuk data harian per jam)

                        progress.progress(60)  # Update progress to 160%

                        # Membangun model SARIMA
                        sarima_model = SARIMAX(y_train, order=(p, d, q), seasonal_order=(P, D, Q, S), enforce_stationarity=False, enforce_invertibility=False)
                        sarima_fit = sarima_model.fit(disp=False)

                        progress.progress(80)  # Update progress to 80%

                        # Prediksi pada data uji
                        y_pred = sarima_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, dynamic=False)
                        
                        progress.progress(100)  # Update progress to 100%
                        st.success("Prediction complete!")

                        # Catat waktu selesai
                        end_time = time.time()

                        # Hitung durasi
                        duration = end_time - start_time

                        # Tampilkan hasil prediksi dan waktu proses
                        st.write(f"Time taken for prediction: {duration:.2f} seconds")
                        
                        # Evaluasi model pada data uji
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        msle = mean_squared_log_error(y_test, y_pred)
                        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                        # Display metrics
                        st.markdown(f"### Evaluation metrics for {target_column} using {algorithm} in {selected_cell}:")
                        col1, col2, col3, col4, col5 = st.columns(5)  # Menambahkan satu kolom lagi
                        with col1:
                            st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}")
                        with col2:
                            st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                        with col3:
                            st.metric(label="R² Score", value=f"{r2:.4f}")
                        with col4:
                            st.metric(label="MSLE", value=f"{msle:.4f}")
                        with col5:
                            st.metric(label="MAPE", value=f"{mape:.2f}%")

                        # Kolom untuk Actual vs Predicted dan Prediksi 3 Hari ke Depan
                        col1, col2 = st.columns(2)

                        # Pastikan akses ke index asli sebelum split
                        original_index = filtered_data.index[-len(y_test):]  # Ambil index data uji (y_test)

                        with col1:
                            # Plotting Actual vs Predicted (For SARIMA)
                            plt.figure(figsize=(12, 6))
                            plt.plot(y_test.index, y_test, label='Actual Data', color='blue')
                            plt.plot(y_test.index, y_pred, label='Predicted Data', color='red', linestyle='--')
                            plt.title(f"Actual vs Predicted {target_column} using {algorithm} in {selected_cell}")
                            plt.xlabel("Datetime")
                            plt.ylabel(target_column)
                            plt.legend()
                            st.pyplot(plt)

                        with col2:
                            # Plot perbandingan antara data sebenarnya dan prediksi (default 120 jam terakhir)
                            plt.figure(figsize=(12, 6))
                            plt.plot(original_index[-120:], y_test[-120:], label='Data Sebenarnya', color='blue')
                            plt.plot(original_index[-120:], y_pred[-120:], label='Data Prediksi', color='red', linestyle='--')
                            plt.title(f"Actual vs Predicted {target_column} (120 Jam Terakhir) using {algorithm} in {selected_cell}")
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
        st.write(f"### Filtered Data for Selected Cell")

        # Show table with all data but initial display is limited to a scrollable view
        st.dataframe(data_vis, height=200)  # Adjust the height to limit visible rows

        # Display number of rows and columns
        num_rows, num_cols = data_vis.shape
        st.caption(f"Jumlah baris: {num_rows}, Jumlah kolom: {num_cols}")
        
        # Visualisasi 4G Total Traffic Per Cell Name
        st.write("### 4G Total Traffic Per Cell Name")
        traffic_per_cell = (
            data.groupby([data.index, 'Cell Name'])['4G Total Traffic (GB)']
            .sum()
            .reset_index()
        )

        # Pivot data agar tiap Cell Name menjadi kolom
        pivot_data = traffic_per_cell.pivot(index='Datetime', columns='Cell Name', values='4G Total Traffic (GB)')
        pivot_data = pivot_data.fillna(0)

        # Plot 4G Total Traffic
        plt.figure(figsize=(12, 6))
        for cell in pivot_data.columns:
            plt.plot(pivot_data.index, pivot_data[cell], label=cell)

        # Menambahkan elemen pada grafik
        plt.xlabel('Waktu')
        plt.ylabel('4G Total Traffic (GB)')
        plt.title('4G Total Traffic Per Cell Name')
        plt.legend(title='Cell Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(plt)

        # Kolom untuk 4G Total Traffic dan 4G Active User
        col_4g_total_traffic = "4G Total Traffic (GB)"
        col_4g_active_user = "4G Active User"
       
        # Menampilkan Top 5 Cell Name untuk Total Traffic dan Active User
        st.write("### Top 5 Cell Name by Total Traffic and Active User")

        # Menghitung top 5 cell dengan total traffic tertinggi
        top_5_traffic = data.groupby('Cell Name')[col_4g_total_traffic].sum().nlargest(5)

        # Menghitung top 5 cell dengan active user tertinggi
        top_5_users = data.groupby('Cell Name')[col_4g_active_user].sum().nlargest(5)

        # Membuat dua kolom sejajar
        col1, col2 = st.columns(2)

        # Plot untuk Top 5 Total Traffic
        with col1:
            plt.figure(figsize=(6, 6))
            top_5_traffic.plot(kind='bar', color='mediumseagreen', edgecolor='black')
            plt.title(f"Top 5 Cell Name by {col_4g_total_traffic}")
            plt.xlabel("Cell Name")
            plt.ylabel(col_4g_total_traffic)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

        # Plot untuk Top 5 Active User
        with col2:
            plt.figure(figsize=(6, 6))
            top_5_users.plot(kind='bar', color='steelblue', edgecolor='black')
            plt.title(f"Top 5 Cell Name by {col_4g_active_user}")
            plt.xlabel("Cell Name")
            plt.ylabel(col_4g_active_user)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(plt)

        target_column_vis = st.sidebar.selectbox("Select Column to Visualize", data_vis.select_dtypes(include=[np.number]).columns)
        if st.sidebar.button("Start Visualization"):
            progress = st.progress(0)
            with st.spinner(f"Starting Visualization for {target_column_vis}..."):
                progress.progress(0)  # Update progress to 0%
                st.write("###", target_column_vis, "Visualization")
                plt.figure(figsize=(20, 5))
                plt.plot(data_vis.index, data_vis[target_column_vis])
                plt.title(f"{target_column_vis} for {selected_cell_vis}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column_vis)
                st.pyplot(plt)
                progress.progress(30)  # Update progress to 30%

                # Weekly
                start_weekly = '2024-02-02 00:00:00'
                end_weekly = '2024-02-09 23:59:59'
                data_mingguan = data_vis.loc[start_weekly:end_weekly]
                date = data_mingguan.index  
                st.write("### Weekly", target_column_vis, "Visualization")
                plt.figure(figsize=(20, 5))
                plt.plot(data_mingguan.index, data_mingguan[target_column_vis])
                plt.title(f"Daily {target_column_vis} for {selected_cell_vis}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column_vis)
                st.pyplot(plt)
                progress.progress(60)  # Update progress to 60%

                # Daily
                start_daily = '2024-02-02 00:00:00'
                end_daily = '2024-02-02 23:59:59'
                data_harian = data_vis.loc[start_daily:end_daily]
                date = data_harian.index  
                st.write("### Daily", target_column_vis, "Visualization")
                plt.figure(figsize=(20, 5))
                plt.plot(data_harian.index, data_harian[target_column_vis])
                plt.title(f"Daily {target_column_vis} for {selected_cell_vis}")
                plt.xlabel("Datetime")
                plt.ylabel(target_column_vis)
                st.pyplot(plt)
                progress.progress(100)  # Update progress to 100%

else:
    st.write("Please upload a CSV file to get started.")

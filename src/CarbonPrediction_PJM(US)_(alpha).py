import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========== Mode Selection: Single-step or Multi-step ==========
IS_MULTISTEP = True   # True => Multi-step output, False => Single-step output
OUTPUT_WINDOW = 24    # Number of steps to predict in multi-step mode

# 0. Read and merge data (Assuming energy_data, weather_data, emission_data have been time-aligned)
energy_data = pd.read_csv('./data/PJM 5 minute standardized data_2023-05-01T00_00_00-04_00_2024-10-31T23_59_59.999000-04_00.csv')
weather_data = pd.read_csv('./data/new york 2022-11-01 to 2024-10-31.csv')
emission_data = pd.read_csv('./data/new_hourly_emission_rates.csv')

# Unify the time column and convert to datetime
energy_data['datetime'] = pd.to_datetime(energy_data['interval_start_utc']).dt.tz_localize(None)
weather_data['datetime'] = pd.to_datetime(weather_data['datetime']).dt.tz_localize(None)
emission_data['datetime'] = pd.to_datetime(emission_data['datetime_utc']).dt.tz_localize(None)

energy_data.set_index('datetime', inplace=True)
weather_data.set_index('datetime', inplace=True)
emission_data.set_index('datetime', inplace=True)

# Merge (Ensure that if the data is at 5-minute intervals, resample to 1H => mean/sum before merging with hourly data)
combined_data = pd.merge(energy_data, weather_data, left_index=True, right_index=True, how='inner')
combined_data = pd.merge(
    combined_data, 
    emission_data[['total_hourly_emission_co2_kg']], 
    left_index=True, 
    right_index=True, 
    how='inner'
)

# Remove duplicate indices
duplicate_indices = combined_data.index.duplicated(keep=False)
if duplicate_indices.any():
    combined_data = combined_data[~duplicate_indices]

# === 1. Feature Engineering: Time-based cyclical encoding + additional fuel mix + weather features, etc. ===
# (1) Time-based features: hour, dayofweek, month
combined_data['hour'] = combined_data.index.hour
combined_data['dayofweek'] = combined_data.index.dayofweek
combined_data['month'] = combined_data.index.month

# Use sin/cos for cyclical encoding (example: hour 0-23, dayofweek 0-6, month 1-12)
combined_data['hour_sin'] = np.sin(2 * np.pi * combined_data['hour'] / 24)
combined_data['hour_cos'] = np.cos(2 * np.pi * combined_data['hour'] / 24)
combined_data['dayofweek_sin'] = np.sin(2 * np.pi * combined_data['dayofweek'] / 7)
combined_data['dayofweek_cos'] = np.cos(2 * np.pi * combined_data['dayofweek'] / 7)
combined_data['month_sin'] = np.sin(2 * np.pi * (combined_data['month'] - 1) / 12)
combined_data['month_cos'] = np.cos(2 * np.pi * (combined_data['month'] - 1) / 12)

# (2) You can consider more fuel mix features (e.g. combined_data['fuel_mix.nuclear'], combined_data['fuel_mix.hydro'], ...)
# (3) More weather features (e.g. windspeed, solar_radiation, etc.)
# combined_data['windspeed'] = ...
# combined_data['solar_radiation'] = ...
# ...

# (4) Select feature list
features = [
    # Load & renewable output
    'net_load', 'renewables', 'renewables_to_load_ratio', 
    'fuel_mix.coal', 'fuel_mix.gas', 'fuel_mix.oil', 
    'fuel_mix.solar', 'fuel_mix.wind', 'fuel_mix.storage', 
    'fuel_mix.nuclear', 'fuel_mix.hydro',
    # optional: 'fuel_mix.nuclear', 'fuel_mix.hydro', 'fuel_mix.solar', etc.
    # Weather features
    'temp', 'humidity', 'cloudcover',
    # Time-based cyclical encoding
    'hour_sin','hour_cos','dayofweek_sin','dayofweek_cos','month_sin','month_cos',
    # Target column
    'total_hourly_emission_co2_kg'
]

combined_data = combined_data[features]  # Keep only these columns

# === 2. Data preprocessing: interpolation, ensuring hourly index, fill missing values ===
combined_data = combined_data.sort_index()
full_index = pd.date_range(
    start=combined_data.index.min(), 
    end=combined_data.index.max(), 
    freq='h'
)
combined_data = combined_data.reindex(full_index)
combined_data = combined_data.interpolate(method='time')  # or combined_data.fillna(method='ffill')
combined_data.index.name = 'datetime'

# If there are still NaNs, drop them
combined_data.dropna(inplace=True)

# ========== 3. Normalization ==========
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data.values)
scaled_df = pd.DataFrame(scaled_data, index=combined_data.index, columns=combined_data.columns)

sequence_length = 24
target_col = scaled_df.columns.get_loc('total_hourly_emission_co2_kg')

X_list, y_list = [], []

# ========== 4. Building (X, y) for single-step or multi-step ==========
if not IS_MULTISTEP:
    # ---- (A) Single-step output ----
    for i in range(sequence_length, len(scaled_df)):
        X_block = scaled_df.iloc[i - sequence_length : i].values  # shape: (24, n_features)
        # Single-step => y is the i-th value
        y_value = scaled_df.iloc[i, target_col]  # shape ()
        X_list.append(X_block)
        y_list.append(y_value)

    X = np.array(X_list)  # (N,24,n_features)
    y = np.array(y_list)  # (N,)

else:
    # ---- (B) Multi-step output ----
    for i in range(sequence_length, len(scaled_df) - OUTPUT_WINDOW + 1):
        X_block = scaled_df.iloc[i - sequence_length : i].values  # shape: (24, n_features)
        y_block = scaled_df.iloc[i : i + OUTPUT_WINDOW, target_col].values  # shape: (OUTPUT_WINDOW,)
        X_list.append(X_block)
        y_list.append(y_block)

    X = np.array(X_list)   # (N,24,n_features)
    y = np.array(y_list)   # (N, OUTPUT_WINDOW)

# Training/testing split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ========== 5. Build the model: Single-step vs. Multi-step ==========

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_rnn_model(input_shape, 
                    is_multistep=False, 
                    output_window=1, 
                    rnn_type='LSTM', 
                    rnn_units=128, 
                    n_layers=2, 
                    dropout_rate=0.2, 
                    lr=1e-3):
    model = Sequential()

    # First layer
    if rnn_type == 'LSTM':
        model.add(LSTM(rnn_units, return_sequences=(n_layers > 1), input_shape=input_shape))
    else:
        model.add(GRU(rnn_units, return_sequences=(n_layers > 1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    # Middle layers (optional multiple layers)
    for _ in range(n_layers - 1):
        if rnn_type == 'LSTM':
            model.add(LSTM(rnn_units, return_sequences=False))
        else:
            model.add(GRU(rnn_units, return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation='relu'))
    # Output layer => depends on is_multistep
    if not is_multistep:
        # Single-step
        model.add(Dense(1))
    else:
        # Multi-step
        model.add(Dense(output_window))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Determine final output dimension based on IS_MULTISTEP
if not IS_MULTISTEP:
    final_output_dim = 1
else:
    final_output_dim = OUTPUT_WINDOW

model = build_rnn_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                        is_multistep=IS_MULTISTEP,
                        output_window=final_output_dim,
                        rnn_type='LSTM',  # or 'GRU' if desired
                        rnn_units=128,
                        n_layers=2,
                        dropout_rate=0.2,
                        lr=1e-3)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

# Training
model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_split=0.1,
          callbacks=[early_stopping, reduce_lr])

# ========== 6. Test Evaluation: Single-step vs. Multi-step ==========

y_pred_test = model.predict(X_test)

if not IS_MULTISTEP:
    # ---- (A) Single-step output => y_pred_test shape (N,1), y_test shape (N,)
    # Inverse scaling
    test_pred_full = np.zeros((len(y_pred_test), scaled_df.shape[1]))
    test_true_full = np.zeros((len(y_test), scaled_df.shape[1]))
    for i in range(len(y_test)):
        test_pred_full[i, target_col] = y_pred_test[i, 0]
        test_true_full[i, target_col] = y_test[i]
    test_pred_inversed = scaler.inverse_transform(test_pred_full)
    test_true_inversed = scaler.inverse_transform(test_true_full)

    y_pred_final = test_pred_inversed[:, target_col]
    y_true_final = test_true_inversed[:, target_col]

    mse = mean_squared_error(y_true_final, y_pred_final)
    mae = mean_absolute_error(y_true_final, y_pred_final)
    print("Single-step results => MSE:", mse, "MAE:", mae)

    # Optionally add “evaluate_different_horizons” for rolling forecast => omitted

else:
    # ---- (B) Multi-step output => y_pred_test shape (N,24), y_test shape (N,24)
    # Inverse scaling for each step => fill => inverse_transform
    N, OUT_WIN = y_pred_test.shape

    test_pred_full = np.zeros((N, OUT_WIN, scaled_df.shape[1]))
    test_true_full = np.zeros((N, OUT_WIN, scaled_df.shape[1]))

    for i in range(N):
        for step in range(OUT_WIN):
            test_pred_full[i, step, target_col] = y_pred_test[i, step]
            test_true_full[i, step, target_col] = y_test[i, step]

    test_pred_inversed = np.zeros_like(test_pred_full)
    test_true_inversed = np.zeros_like(test_true_full)

    for step in range(OUT_WIN):
        test_pred_inversed[:, step, :] = scaler.inverse_transform(test_pred_full[:, step, :])
        test_true_inversed[:, step, :] = scaler.inverse_transform(test_true_full[:, step, :])

    y_pred_final = test_pred_inversed[:, :, target_col]  # (N,24)
    y_true_final = test_true_inversed[:, :, target_col]  # (N,24)

    # Overall error => flatten
    mse_all = mean_squared_error(y_true_final.flatten(), y_pred_final.flatten())
    mae_all = mean_absolute_error(y_true_final.flatten(), y_pred_final.flatten())
    print("Multi-step results => Overall MSE:", mse_all, "MAE:", mae_all)

    # If you want to see each step, define evaluate_multi_step
    def evaluate_multi_step(model, X_test, y_test, scaler, target_col, horizons=[1,6,12,24]):
        y_pred_test = model.predict(X_test)  # shape (N, OUT_WIN)
        N, OUT_WIN = y_pred_test.shape
        y_pred_test_inv = np.zeros((N, OUT_WIN))
        y_test_inv = np.zeros((N, OUT_WIN))
        for i in range(N):
            for st in range(OUT_WIN):
                arr_pred = np.zeros((1, scaler.n_features_in_))
                arr_true = np.zeros((1, scaler.n_features_in_))
                arr_pred[0, target_col] = y_pred_test[i, st]
                arr_true[0, target_col] = y_test[i, st]
                inv_pred = scaler.inverse_transform(arr_pred)
                inv_true = scaler.inverse_transform(arr_true)
                y_pred_test_inv[i, st] = inv_pred[0, target_col]
                y_test_inv[i, st] = inv_true[0, target_col]
        results = {}
        for h in horizons:
            if h > OUT_WIN:
                results[h] = (None, None)
                continue
            idx = h - 1
            mae_h = mean_absolute_error(y_test_inv[:, idx], y_pred_test_inv[:, idx])
            mse_h = mean_squared_error(y_test_inv[:, idx], y_pred_test_inv[:, idx])
            results[h] = (mae_h, mse_h)
        return results
    
    horizons_to_test = [1, 6, 12, 24]
    multi_res = evaluate_multi_step(model, X_test, y_test, scaler, target_col, horizons=horizons_to_test)
    for h in horizons_to_test:
        mae_h, mse_h = multi_res[h]
        print(f"Horizon={h} => MAE={mae_h:.2f}, MSE={mse_h:.2f}")


# Visualize a small portion
plt.figure(figsize=(12, 5))
plt.plot(y_true_final[-200:], label='Actual', color='blue')
plt.plot(y_pred_final[-200:], label='Predicted', color='red')
plt.title('Prediction vs. Actual (last 200 points in the test set)')
plt.legend()
plt.show()


# ========== 7. Rolling Prediction for the Next 24 Hours ==========

# Use the last sample in the test set as input
last_sequence = X_test[-1]  # shape: (sequence_length, n_features)

# Rolling forecast
forecast_list = []
current_sequence = last_sequence.copy()

for _ in range(OUTPUT_WINDOW):  # Rolling forecast for 24 steps
    X_pred = current_sequence.reshape(1, sequence_length, -1)  # Reshape input dimension
    # If it's a multi-step model, might be model.predict(X_pred)[0]
    pred_scaled = model.predict(X_pred)[0, 0]
    forecast_list.append(pred_scaled)

    # Update sequence: rolling window
    new_entry = current_sequence[-1].copy()
    new_entry[target_col] = pred_scaled  # Replace target column with the predicted value
    current_sequence = np.vstack([current_sequence[1:], new_entry])

# Inverse scale the forecast values
forecast_full = np.zeros((OUTPUT_WINDOW, scaled_df.shape[1]))
for i in range(OUTPUT_WINDOW):
    forecast_full[i, target_col] = forecast_list[i]
forecast_inversed = scaler.inverse_transform(forecast_full)
forecast_final = forecast_inversed[:, target_col]

# Print predicted values
print("Future 24-hour Predicted Carbon Emissions:")
print(forecast_final)

# Past 72 hours of actual data
past_72 = scaled_df.iloc[-(72 + sequence_length) : -sequence_length].copy()
past_72_full = scaler.inverse_transform(past_72.values)  # Inverse scaling
past_72_emission = past_72_full[:, target_col]
past_72_time = past_72.index  # Time index

# Future 24-hour time index
future_dates = pd.date_range(
    start=past_72_time[-1] + pd.Timedelta(hours=1), 
    periods=OUTPUT_WINDOW, 
    freq='h'
)

# Plot the chart
plt.figure(figsize=(14, 7))

# Plot the past 72 hours of actual data
plt.plot(past_72_time, past_72_emission, label='Actual Carbon Emission (last 72h)', color='blue')

# Plot the next 24 hours of predicted data
plt.plot(future_dates, forecast_final, marker='o', linestyle='--', color='red', 
         label='Predicted Carbon Emission (next 24h)')

# Chart style
plt.title("Past 72 Hours and Forecasted Next 24 Hours Carbon Emission")
plt.xlabel("Time")
plt.ylabel("Carbon Emission (kg CO₂)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Done!")


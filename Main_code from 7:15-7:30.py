import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Data loading and preprocessing
weather_emissions_df = pd.read_csv('california 2024-07-15 to 2024-07-30_2.csv')
weather_emissions_df['date'] = pd.to_datetime(weather_emissions_df['date'])
weather_emissions_df.set_index('date', inplace=True)

energy_df = pd.read_csv('combined_energy_data_with_emissions_2.csv')
energy_df['date'] = pd.to_datetime(energy_df['date'])
energy_df.set_index('date', inplace=True)

# Merge datasets
df = pd.merge(weather_emissions_df, energy_df, left_index=True, right_index=True)

# Data cleaning
df['temp'] = df['temp'].astype(str).str.replace('°F', '').astype(float)
df['humidity'] = df['humidity'].astype(str).str.replace('%', '').astype(float)
df['precip'] = df['precip'].astype(str).str.replace('in', '').astype(float)
df.fillna(method='ffill', inplace=True)

# Feature selection
features = ['carbon_emissions_mTCO2_per_hour', 'temp', 'humidity', 'precip', 
            'net_load', 'renewables', 'renewables_to_load_ratio', 
            'load.load', 'load_forecast.load_forecast', 
            'fuel_mix.biomass', 'fuel_mix.biogas', 'fuel_mix.coal', 
            'fuel_mix.natural_gas', 'fuel_mix.large_hydro', 
            'fuel_mix.imports', 'fuel_mix.other', 
            'storage.stand_alone_batteries']
df = df[features]

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Define sequence length
sequence_length = 24  # Trying a longer sequence length

# Create sequence data
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, :])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Build hybrid model architecture (LSTM + GRU + Conv1D)
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=200, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=150, return_sequences=True))
model.add(Dropout(0.2))
model.add(GRU(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Use early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Predict the next 24 time steps
last_sequence = scaled_data[-sequence_length:]
forecast = []

for _ in range(24):
    X_pred = last_sequence.reshape((1, sequence_length, last_sequence.shape[1]))
    pred = model.predict(X_pred)
    forecast.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], [np.concatenate([[pred[0, 0]], [0] * (last_sequence.shape[1] - 1)])], axis=0)

# Inverse the scaling of forecasted results
forecast_array = np.zeros((24, scaled_data.shape[1]))
forecast_array[:, 0] = forecast
forecast_original = scaler.inverse_transform(forecast_array)[:, 0]

# Generate future date range
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['carbon_emissions_mTCO2_per_hour'], color='blue', label='Actual Data')
plt.plot(future_dates, forecast_original, color='red', marker='o', label='Forecasted Data')
plt.xlabel('Date')
plt.ylabel('Carbon Emissions (mTCO2/h)')
plt.title('Optimized Carbon Emissions Forecast using LSTM + GRU + CNN')
plt.legend()
plt.show()

print("Forecasted values for the next 24 data points:")
print(forecast_original)








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 加载数据
energy_data_path = 'combined_energy_data_with_emissions.csv'
weather_data_path = 'weather_data_of_california 2024-09-30 to 2024-10-30.csv'

energy_data = pd.read_csv(energy_data_path)
weather_data = pd.read_csv(weather_data_path)

# 预处理数据
energy_data['datetime'] = pd.to_datetime(energy_data['interval_start_utc']).dt.tz_localize(None)
weather_data['datetime'] = pd.to_datetime(weather_data['datetime']).dt.tz_localize(None)

energy_data.set_index('datetime', inplace=True)
weather_data.set_index('datetime', inplace=True)

# 合并数据
combined_data = pd.merge(energy_data, weather_data, left_index=True, right_index=True)

# 提取碳排放数据
combined_data['carbon_intensity'] = combined_data['carbon_emissions_mTCO2_per_hour']

# 使用 StandardScaler 进行标准化
scaler = StandardScaler()
combined_data['carbon_intensity_scaled'] = scaler.fit_transform(combined_data[['carbon_intensity']])

# 准备 LSTM 模型数据
sequence_length = 24
X, y = [], []
for i in range(sequence_length, len(combined_data)):
    X.append(combined_data['carbon_intensity_scaled'].values[i-sequence_length:i])
    y.append(combined_data['carbon_intensity_scaled'].values[i])

X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)

# 构建 LSTM 模型
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_lstm_model((X.shape[1], X.shape[2]))
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# 预测未来24小时
last_sequence = combined_data['carbon_intensity_scaled'].values[-sequence_length:]
forecast = []
for _ in range(24):
    X_pred = last_sequence.reshape((1, sequence_length, 1))
    pred = model.predict(X_pred)
    forecast.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred[0, 0])

# 逆标准化预测结果
forecast_emission = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

# 可视化
past_72_hours = combined_data['carbon_intensity'][-72:]
past_dates = pd.date_range(end=combined_data.index[-1], periods=72, freq='h')
future_dates = pd.date_range(start=past_dates[-1] + pd.Timedelta(hours=1), periods=24, freq='h')

plt.figure(figsize=(14, 7), dpi=120)
plt.plot(past_dates, past_72_hours, color='blue', label='Actual Carbon Intensity (Last 72 hours)', linewidth=2)
plt.plot(future_dates, forecast_emission, color='red', marker='o', linestyle='--', label='Predicted Carbon Emission (Next 24 hours)', linewidth=2)

plt.title("Past 72 Hours and Forecasted Next 24 Hours Carbon Emission")
plt.xlabel("Time")
plt.ylabel("Carbon Emission (mTCO₂/h)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 输出预测结果
print("Forecasted values for the next 24 hours:")
print(forecast_emission)













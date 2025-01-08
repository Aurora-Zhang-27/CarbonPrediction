import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========== 切换模式: 单步 or 多步 ==========
IS_MULTISTEP = True   # True => 多步输出, False => 单步输出
OUTPUT_WINDOW = 24    # 多步时一次输出几步

# 0. 读取并合并数据（假设你已经确保energy_data, weather_data, emission_data时间对齐）
energy_data = pd.read_csv('/Users/mac/Desktop/CAISO2/data/PJM 5 minute standardized data_2023-05-01T00_00_00-04_00_2024-10-31T23_59_59.999000-04_00.csv')
weather_data = pd.read_csv('/Users/mac/Desktop/CAISO2/data/new york 2022-11-01 to 2024-10-31.csv')
emission_data = pd.read_csv('/Users/mac/Desktop/CAISO2/data/new_hourly_emission_rates.csv')

# 统一时间列，并转为datetime
energy_data['datetime'] = pd.to_datetime(energy_data['interval_start_utc']).dt.tz_localize(None)
weather_data['datetime'] = pd.to_datetime(weather_data['datetime']).dt.tz_localize(None)
emission_data['datetime'] = pd.to_datetime(emission_data['datetime_utc']).dt.tz_localize(None)

energy_data.set_index('datetime', inplace=True)
weather_data.set_index('datetime', inplace=True)
emission_data.set_index('datetime', inplace=True)

# 合并 (确保你在此之前若是5分钟数据，要先 resample('1H') => mean/sum，再与小时级数据做merge)
combined_data = pd.merge(energy_data, weather_data, left_index=True, right_index=True, how='inner')
combined_data = pd.merge(
    combined_data, 
    emission_data[['total_hourly_emission_co2_kg']], 
    left_index=True, 
    right_index=True, 
    how='inner'
)

# 去除重复索引
duplicate_indices = combined_data.index.duplicated(keep=False)
if duplicate_indices.any():
    combined_data = combined_data[~duplicate_indices]

# === 1. 特征工程：时间周期编码 + 更多燃料占比 + 天气特征等 ===
# (1) 时间周期：hour, dayofweek, month
combined_data['hour'] = combined_data.index.hour
combined_data['dayofweek'] = combined_data.index.dayofweek
combined_data['month'] = combined_data.index.month

# 用sin/cos进行周期编码（示例：hour 0-23, dayofweek 0-6, month 1-12）
combined_data['hour_sin'] = np.sin(2 * np.pi * combined_data['hour'] / 24)
combined_data['hour_cos'] = np.cos(2 * np.pi * combined_data['hour'] / 24)
combined_data['dayofweek_sin'] = np.sin(2 * np.pi * combined_data['dayofweek'] / 7)
combined_data['dayofweek_cos'] = np.cos(2 * np.pi * combined_data['dayofweek'] / 7)
combined_data['month_sin'] = np.sin(2 * np.pi * (combined_data['month']-1) / 12)
combined_data['month_cos'] = np.cos(2 * np.pi * (combined_data['month']-1) / 12)

# (2) 可以再考虑更多燃料占比
# e.g. combined_data['fuel_mix.nuclear'], combined_data['fuel_mix.hydro'], ...
# (3) 更多天气特征
# combined_data['windspeed'] = ...
# combined_data['solar_radiation'] = ...
# ...

# (4) 选取特征列表
features = [
    # 负荷 & 可再生出力
    'net_load', 'renewables', 'renewables_to_load_ratio', 
    'fuel_mix.coal', 'fuel_mix.gas', 'fuel_mix.oil', 
    'fuel_mix.solar', 'fuel_mix.wind', 'fuel_mix.storage', 
    'fuel_mix.nuclear', 'fuel_mix.hydro', 
    # 可选：'fuel_mix.nuclear', 'fuel_mix.hydro', 'fuel_mix.solar', ...
    # 天气特征
    'temp', 'humidity', 'cloudcover',
    # 时间周期编码
    'hour_sin','hour_cos','dayofweek_sin','dayofweek_cos','month_sin','month_cos',
    # 目标列
    'total_hourly_emission_co2_kg'
]

combined_data = combined_data[features]  # 保留这些列

# === 2. 数据预处理：插值、保证整点索引、填充缺失 ===
combined_data = combined_data.sort_index()
full_index = pd.date_range(
    start=combined_data.index.min(), 
    end=combined_data.index.max(), 
    freq='h'
)
combined_data = combined_data.reindex(full_index)
combined_data = combined_data.interpolate(method='time')  # 或 combined_data.fillna(method='ffill')
combined_data.index.name = 'datetime'

# 如果依旧有 NaN，可再 dropna()
combined_data.dropna(inplace=True)
    
# ========== 3. 归一化 ==========
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data.values)
scaled_df = pd.DataFrame(scaled_data, index=combined_data.index, columns=combined_data.columns)

sequence_length = 24
target_col = scaled_df.columns.get_loc('total_hourly_emission_co2_kg')

X_list, y_list = [], []

# ========== 4. 构造 (X, y) 分单步 or 多步 ==========
if not IS_MULTISTEP:
    # ---- (A) 单步输出 ----
    for i in range(sequence_length, len(scaled_df)):
        X_block = scaled_df.iloc[i-sequence_length:i].values  # shape (24, n_features)
        # 单步 => y只取 i 这个点
        y_value = scaled_df.iloc[i, target_col]  # shape ()
        X_list.append(X_block)
        y_list.append(y_value)

    X = np.array(X_list)  # (N,24,n_features)
    y = np.array(y_list)  # (N,)

else:
    # ---- (B) 多步输出 ----
    for i in range(sequence_length, len(scaled_df) - OUTPUT_WINDOW + 1):
        X_block = scaled_df.iloc[i-sequence_length:i].values  # shape (24, n_features)
        y_block = scaled_df.iloc[i:i+OUTPUT_WINDOW, target_col].values  # shape (OUTPUT_WINDOW,)
        X_list.append(X_block)
        y_list.append(y_block)

    X = np.array(X_list)   # (N,24,n_features)
    y = np.array(y_list)   # (N, OUTPUT_WINDOW)

# 切分训练/测试
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ========== 5. 构建模型: 单步 vs. 多步 ==========

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

    # 第一层
    if rnn_type == 'LSTM':
        model.add(LSTM(rnn_units, return_sequences=(n_layers>1), input_shape=input_shape))
    else:
        model.add(GRU(rnn_units, return_sequences=(n_layers>1), input_shape=input_shape))
    model.add(Dropout(dropout_rate))

    # 中间层(可选多层)
    for _ in range(n_layers - 1):
        if rnn_type == 'LSTM':
            model.add(LSTM(rnn_units, return_sequences=False))
        else:
            model.add(GRU(rnn_units, return_sequences=False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(64, activation='relu'))
    # 输出层 => 看 is_multistep
    if not is_multistep:
        # 单步
        model.add(Dense(1))
    else:
        # 多步
        model.add(Dense(output_window))

    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 根据 IS_MULTISTEP 来决定 final output
if not IS_MULTISTEP:
    final_output_dim = 1
else:
    final_output_dim = OUTPUT_WINDOW

model = build_rnn_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                        is_multistep=IS_MULTISTEP,
                        output_window=final_output_dim,
                        rnn_type='LSTM',  #or 'GRU' depends on how much time it will take
                        rnn_units=128,
                        n_layers=2,
                        dropout_rate=0.2,
                        lr=1e-3)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

# 训练
model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_split=0.1,
          callbacks=[early_stopping, reduce_lr])

# ========== 6. 测试评估: 单步 vs. 多步 ==========

y_pred_test = model.predict(X_test)

if not IS_MULTISTEP:
    # ---- (A) 单步输出 => y_pred_test shape (N,1), y_test shape (N,)
    # 反归一化
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

    # 你也可加“evaluate_different_horizons”函数做滚动预测 => omitted

else:
    # ---- (B) 多步输出 => y_pred_test shape (N,24), y_test shape (N,24)
    # 反归一化: for each step => fill => inverse_transform
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

    # 整体误差 => flatten
    mse_all = mean_squared_error(y_true_final.flatten(), y_pred_final.flatten())
    mae_all = mean_absolute_error(y_true_final.flatten(), y_pred_final.flatten())
    print("Multi-step results => Overall MSE:", mse_all, "MAE:", mae_all)

    # 若想看每步,写个 evaluate_multi_step
    def evaluate_multi_step(model, X_test, y_test, scaler, target_col, horizons=[1,6,12,24]):
        y_pred_test = model.predict(X_test)  # shape (N, OUT_WIN)
        N, OUT_WIN = y_pred_test.shape
        y_pred_test_inv = np.zeros((N, OUT_WIN))
        y_test_inv = np.zeros((N, OUT_WIN))
        for i in range(N):
            for st in range(OUT_WIN):
                arr_pred = np.zeros((1, scaler.n_features_in_))
                arr_true = np.zeros((1, scaler.n_features_in_))
                arr_pred[0,target_col] = y_pred_test[i, st]
                arr_true[0,target_col] = y_test[i, st]
                inv_pred = scaler.inverse_transform(arr_pred)
                inv_true = scaler.inverse_transform(arr_true)
                y_pred_test_inv[i, st] = inv_pred[0, target_col]
                y_test_inv[i, st] = inv_true[0, target_col]
        results = {}
        for h in horizons:
            if h>OUT_WIN:
                results[h] = (None,None)
                continue
            idx = h-1
            mae_h = mean_absolute_error(y_test_inv[:, idx], y_pred_test_inv[:, idx])
            mse_h = mean_squared_error(y_test_inv[:, idx], y_pred_test_inv[:, idx])
            results[h] = (mae_h, mse_h)
        return results
    
    horizons_to_test = [1,6,12,24]
    multi_res = evaluate_multi_step(model, X_test, y_test, scaler, target_col, horizons=horizons_to_test)
    for h in horizons_to_test:
        mae_h, mse_h = multi_res[h]
        print(f"Horizon={h} => MAE={mae_h:.2f}, MSE={mse_h:.2f}")

print("Done!")



'''
# 可视化一小段
plt.figure(figsize=(12,5))
plt.plot(y_true_final[-200:], label='Actual', color='blue')
plt.plot(y_pred_final[-200:], label='Predicted', color='red')
plt.title('Prediction vs Actual (last 200 points in test set)')
plt.legend()
plt.show()
'''


'''
# ========== 7. 滚动预测未来 24 小时 ==========

# 使用测试集最后一个样本的序列作为输入
last_sequence = X_test[-1]  # shape: (sequence_length, n_features)

# 滚动预测
forecast_list = []
current_sequence = last_sequence.copy()

for _ in range(OUTPUT_WINDOW):  # 滚动预测24步
    X_pred = current_sequence.reshape(1, sequence_length, -1)  # 调整输入维度
    pred_scaled = model.predict(X_pred)[0, 0]  # 如果是多步预测模型，可能是 model.predict(X_pred)[0]
    forecast_list.append(pred_scaled)

    # 更新序列: 滚动窗口
    new_entry = current_sequence[-1].copy()
    new_entry[target_col] = pred_scaled  # 用预测值替换目标列
    current_sequence = np.vstack([current_sequence[1:], new_entry])

# 反归一化预测值
forecast_full = np.zeros((OUTPUT_WINDOW, scaled_df.shape[1]))
for i in range(OUTPUT_WINDOW):
    forecast_full[i, target_col] = forecast_list[i]
forecast_inversed = scaler.inverse_transform(forecast_full)
forecast_final = forecast_inversed[:, target_col]

# 输出预测值
print("Future 24-hour Predicted Carbon Emissions:")
print(forecast_final)

# 过去 72 小时的真实数据
past_72 = scaled_df.iloc[-(72 + sequence_length):-sequence_length].copy()
past_72_full = scaler.inverse_transform(past_72.values)  # 反归一化
past_72_emission = past_72_full[:, target_col]
past_72_time = past_72.index  # 时间索引

# 未来 24 小时的时间索引
future_dates = pd.date_range(
    start=past_72_time[-1] + pd.Timedelta(hours=1), 
    periods=OUTPUT_WINDOW, 
    freq='h'
)

# 绘制图表
plt.figure(figsize=(14, 7))

# 绘制过去 72 小时的真实数据
plt.plot(past_72_time, past_72_emission, label='Actual Carbon Emission (last 72h)', color='blue')

# 绘制未来 24 小时的预测数据
plt.plot(future_dates, forecast_final, marker='o', linestyle='--', color='red', 
         label='Predicted Carbon Emission (next 24h)')

# 图表样式
plt.title("Past 72 Hours and Forecasted Next 24 Hours Carbon Emission")
plt.xlabel("Time")
plt.ylabel("Carbon Emission (kg CO₂)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''
"""
Generate From Copilot
股票預測模型
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 讀取股票數據（需替換為實際資料）
df = pd.read_csv("stock_data.csv")  # 確保有日期、開盤價、最高價、最低價、收盤價等欄位

# 數據預處理
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Close']])  # 僅使用收盤價作為特徵

# 建立訓練集
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])  # 預測下一時間步的價格
    return np.array(sequences), np.array(labels)

seq_length = 50  # 使用過去 50 天的數據來預測
X, y = create_sequences(df_scaled, seq_length)

# 切分訓練集與測試集
split = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# 構建 LSTM 模型
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(1)  # 輸出一個數值，表示預測價格
])

model.compile(loss='mse', optimizer='adam')

# 訓練模型
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# 預測結果
preds = model.predict(X_test)
preds = scaler.inverse_transform(preds)  # 轉換回原始價格尺度
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# 繪製預測 vs 實際價格
plt.plot(y_test_real, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.legend()
plt.show()


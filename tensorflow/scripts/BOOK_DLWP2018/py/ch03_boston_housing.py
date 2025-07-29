# 3-24 加載波士頓房價數據
from tensorflow import keras
from keras.datasets import boston_housing

print("=== PGM START ===")

print("\n=== 3-24 ===")
(train_data, train_targets), (test_data, test_targets) = \
    boston_housing.load_data()

print("train_data.shape= ", train_data.shape)
print("test_data.spahe = ", test_data.shape)
#print("train_targets= ", train_targets)

# 3-25 數據標準化
print("\n=== 3-25 ===")
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

# 3-26 模型定義
print("\n=== 3-26 ===")
from keras import models
from keras import layers

def build_model() :
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', 
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

# 3-27 K折驗證
print("\n=== 3-27 ===")
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k) :
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
         axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
         axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

    print("all_scores= ", all_scores)
    print("np.mean(all_scores)= ", np.mean(all_scores))

# 3-28 保存每折的驗證結果
print("\n=== 3-28 ===")
num_epochs = 500
all_mae_histories = [] 
for i in range(k):
    print('processing fold #', i) 
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples] 
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
#   print("history.history.keys()= ", history.history.keys())
#   history.history.keys()=  dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])    
#   mae_history = history.history['val_mean_absolute_error'] 
    mae_history = history.history['val_mae'] 
    all_mae_histories.append(mae_history)

# 3-29 計算所有輪次中的K折驗證分數平均值
print("\n=== 3-29 ===")
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 3-30 繪製驗證分數
print("\n=== 3-30 ===")
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history) 
plt.xlabel('Epochs')
plt.ylabel('Validation MAE') 
plt.show()

# 3-31 繪製驗證分數(刪除前10個數據點)
print("\n=== 3-31 ===")
def smooth_curve(points, factor=0.9):
    smoothed_points = [] 
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1] 
            smoothed_points.append(previous * factor + point * (1 - factor)) 
        else:
            smoothed_points.append(point) 
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history) 
plt.xlabel('Epochs')
plt.ylabel('Validation MAE') 
plt.show()

# 3-32 訓練最終模型
print("\n=== 3-32 ===")
model = build_model()
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print("test_mae_score= ", test_mae_score)

print("=== PGM END ===")


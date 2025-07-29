"""
20250616 此版本不work，目前找不出問題…
"""

####################
# [01]匯入必要套件 #
####################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
from keras import layers

print(os.getcwd())

######################
# [02]資料讀取並分析 #
######################
# windows use
#data = pd.read_csv("data\\kc_house_data.csv")
# mac / linux use
data = pd.read_csv("data/kc_house_data.csv")

print(data.shape)

pd.options.display.max_columns = 25
#print(data.head)
print(data.dtypes)

##################
# [03]資料前處理 #
##################
data['year'] = pd.to_numeric(data['date'].str.slice(0, 4))
data['month'] = pd.to_numeric(data['date'].str.slice(4, 6))
data['day'] = pd.to_numeric(data['date'].str.slice(6, 8))

data.drop(['id'], axis="columns", inplace=True)
data.drop(['date'], axis="columns", inplace=True)
data.head()

data_num = data.shape[0]
indexes = np.random.permutation(data_num)

train_indexes = indexes[:int(data_num * 0.6)]
val_indexes = indexes[int(data_num * 0.6):int(data_num * 0.8)]
test_indexes = indexes[int(data_num * 0.8):]

train_data = data.loc[train_indexes]
val_data = data.loc[val_indexes]
test_data = data.loc[test_indexes]

##############
# [04]標準化 #
##############
train_validation_data = pd.concat([train_data, val_data])
mean = train_validation_data.mean()
std = train_validation_data.std()

train_data = (train_data - mean) / std
val_data = (val_data - mean) / std

#####################################
# [05]建立Numpy array格式的訓練資料 #
#####################################
x_train = np.array(train_data.drop('price', axis='columns'))
y_train = np.array(train_data['price'])
x_val = np.array(val_data.drop('price', axis='columns'))
y_val = np.array(val_data['price'])

print(x_train.shape)

##########################
# [06]建立並訓練網路模型 #
##########################
model = keras.Sequential(name=' model-1')
model.add(layers.Dense(64, activation='relu', input_shape=(21,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.summary()

model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.MeanAbsoluteError()])

#windows use
#model_dir = 'lab2-logs\\models\\'
#mac /linux use
model_dir = 'lab2-logs/models/'
os.makedirs(model_dir, exist_ok=True)

"""
log_dir = os.path.join('lab2-logs', 'model-1')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)

model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5',
                                             monitor='val_mean_absolute_error',
                                             save_best_only=True,
                                             mode='min')

"""                                             # TensorBoard回調函數會幫忙紀錄訓練資訊，並存成TensorBoard的紀錄檔
log_dir = os.path.join('lab2-logs', 'model-1')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')
""" 
history = model.fit(x_train, y_train,
                   batch_size=64,
                   epochs=300,
                   validation_data=(x_val, y_val),
                   callbacks=[model_cbk, model_mckp])
"""
history = model.fit(x_train, y_train,  # 傳入訓練數據
               batch_size=64,  # 批次大小設為64
               epochs=300,  # 整個dataset訓練300遍
               validation_data=(x_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp])  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型


"""
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
"""

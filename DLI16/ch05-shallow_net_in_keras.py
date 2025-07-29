from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape= ", x_train.shape)

np.set_printoptions(linewidth=np.inf)
#print("x_train[0]= ", x_train[0])

print("y_train.shape= ", y_train.shape)

print("y_train[0:12]= ", y_train[0:12])

plt.figure(figsize=(5,5))
for k in range(12) :
    plt.subplot(3, 4, k+1)
    plt.imshow(x_train[k], cmap='gray')
plt.tight_layout()
plt.show()

"""
x_train, x_test : 資料樣本
y_train, y_test : 資料標籤
"""

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

x_train /= 255
x_test /= 255

print("y_train[0]= ", y_train[0])

n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

print("y_train[0]= ", y_train[0])

model = Sequential()
model.add(Dense(64, activation='sigmoid', input_shape=(784,))) # 隱藏層；激活函數 sigmoid
model.add(Dense(10, activation='softmax')) # 輸出層









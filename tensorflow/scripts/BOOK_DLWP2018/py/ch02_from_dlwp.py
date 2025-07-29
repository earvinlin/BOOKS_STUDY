import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

""" marked 1
###--- § 2-1 加載Keras中的MNIST數據集 ---###
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("len(train_labels): ", len(train_labels))
print("train_labels     : ", train_labels)
print("len(test_labels): ", len(test_labels))
print("test_labels     : ", test_labels)

print("len(train_images): ", len(train_images))
print("train_images[0]= ", train_images[0])

###--- § 2-2 網路架構 ---###
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

###--- § 2-3 編譯步驟 ---###
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

###--- § 2-4 准備圖像數據 ---###
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

print("train_images[2]= ", train_images[2])

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
"""

###--- § 2-5 准備標籤 ---###
"""
to_categorical : 
    Keras 庫中的一個便捷函數，用於將類別標籤轉換為 one-hot 編碼的格式。
    在深度學習中，尤其是在分類任務中，這種格式對於訓練神經網絡非常有用。
"""
""" marked 2
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_loss= ", test_loss, ", test_acc= ", test_acc)
""" 

###--- § 2-6 顯示第4個數字 ---###
""" marked 3 
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

my_slice = train_images[10:100]
print("my_slice.shape(1) : ", my_slice.shape)

my_slice = train_images[10:100, 0:28, 0:28]
print("my_slice.shape(2) : ", my_slice.shape)

my_slice = train_images[:, 14:, 14:]
plt.imshow(my_slice[4], cmap=plt.cm.binary)
plt.show()

my_slice = train_images[:, 7:-7, 7:-7]
plt.imshow(my_slice[4], cmap=plt.cm.binary)
plt.show()
"""

#################
# §2.3.1 (p.30) #
#################
def naive_relu(x) :
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            x[i, j] = max(x[i, j], 0)
    return x

a0 = np.array([[1,-3,3], [-2,5,7]])
print("a0= ", naive_relu(a0))



#################
# §2.3.2 (p.31) #
#################
def naive_add_matrix_and_vector(x, y) :
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            x[i, j] += y[j]
    return x

a = np.array([[1,12,3], [2,3,9]])
b = np.array([5,6,7])
print("a.shape= ", a.shape)
print("b.shape= ", b.shape)

print("len(a.shape)= ", len(a.shape))
print("len(b.shape)= ", len(b.shape))
print("z= ", np.maximum(a,b))

print("x= ", naive_add_matrix_and_vector(a,b))


#################
# §2.3.3 (p.32) #
#################
def naive_vector_dot(x, y) :
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]) :
        z += x[i] * y[i]
    return z

a1 = np.array([1,3,2])
a2 = np.array([2,5,-1])

print("z2= ", naive_vector_dot(a1, a2))


#################
# §2.3.3 (p.32) #
#################
def naive_matrix_vector_dot(x, y) :
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            z[i] += x[i, j] * y[j]
    return z

def naive_matrix_vector_dot(x, y) :
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]) :
        z[i] = naive_vector_dot(x[i,:], y)
    return z

def naive_matrix_dot(x, y) :
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]) :
        for j in range(y.shape[1]) :
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


















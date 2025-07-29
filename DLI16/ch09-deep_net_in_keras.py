from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) =mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

x_train /= 255
x_test /= 255

n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,))) # 第一隱藏層
model.add(BatchNormalization())

model.add(Dense(64, activation='relu')) # 第二隱藏層
model.add(BatchNormalization())

model.add(Dense(64, activation='relu')) # 第三隱藏層
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))



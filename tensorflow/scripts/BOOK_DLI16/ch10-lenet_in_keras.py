from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')

# 將數值範圍縮成 0 ~ 1
x_train /= 255
x_test /= 255

n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)

# Setting CNN
model = Sequential()
# 第1卷積層
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
            input_shape=(28, 28, 1)))
# 第2卷積層，並搭配最大池化層對丟棄層
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
# 搭配丟棄法的密集隱藏層
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# 輸出層
model.add(Dense(n_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', 
        metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,
        validation_data=(x_test, y_test))


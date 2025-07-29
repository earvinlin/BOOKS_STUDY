from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
# 20250419 好像不支援了，無法測試…
import tflearn.datasets.oxflower17 as oxflower17

X, Y = oxflower17.load_data(one_hot=True)

model = Sequential()

# 第1卷積塊
model.add(Conv2D(96, kernel_size=(11, 11),
            strides=(4, 4), activation='relu',
            input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
# 第2卷積塊
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
# 第3卷積塊
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())
# 密集層
model.add(Flatten())
model.add(Dense(4096,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
# 輸出層
model.add(Dense(17, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1,
        validation_data=(x_test, y_test))



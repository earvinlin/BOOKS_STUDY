import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# x_train.shape = (404, 13)
# x_test.shape  = (102, 13)
# y_train[0] = 15.2 (房價，15200)

model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, activation='linear'))

"""
model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
"""
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, 
        batch_size=8, epochs=32, verbose=1, 
        validation_data=(x_test, y_test))

# 進行實際預測
print('result= ', model.predict(np.reshape(x_test[42], [1, 13])))




from tensorflow import keras
from keras.datasets import reuters

print("=== PGM START ===")

# 3-12 加載路透社數據集
(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data(num_words=10000)

print("train_data's size= ", len(train_data))
print("test_data's size= ", len(test_data))
print("train_data[0]= ", train_data[10])

# 3-13 將索引解碼為新聞文本
word_index = reuters.get_word_index()
reverse_word_index = \
    dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = \
    ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print("train_labels[10]= ", train_labels[10])

# 3-14 編碼數據 (數據向量化)
import numpy as np
def vectorize_sequences(sequences, dimension=10000) :
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data) # 將訓練數據向量化
x_test = vectorize_sequences(test_data)  # 將測試數據向量化

def to_one_hot(labels, dimension=46) :
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels) :
        results[i, label] = 1.
    return results

# Use one hot coding
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print("train_labels(one hot)= ", one_hot_train_labels)
print("test_labels(ont hot) = ", one_hot_test_labels)
print("test_data[1]= ", test_data[1])
print("test_data[][]= ", test_data[1][1])
print("test_labels[][]= ", x_test[1][2768])

# 3-15 模型定義
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 3-16 編譯模型
model.compile(optimizer='rmsprop', 
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# 3-17 留出驗證集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 3-18 訓練模型
history = model.fit(partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val))

# 3-19 繪製訓練損失和驗證損失
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 3-20 繪製訓練精度和驗證精度
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'bo', label='Trainging acc')
plt.plot(epochs, val_acc, 'b', label= 'Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 3-21 從頭開始重新訓練一個模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
model.fit(partial_x_train,
        partial_y_train,
        epochs=9,
        batch_size=512,
        validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

print("results= ", results)

# 隨機分類
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels_copy)
print(float(np.sum(hits_array)) / len(test_labels))
    
# 3-22 在新數據上生成預測結果
predictions = model.predict(x_test)
print("predictions[0].shape= ", predictions[0].shape)
print("np.argmax(predictions[0]= ", np.argmax(predictions[0]))

y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(optimizer='rmsprop',
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

# 3-23 具有信息瓶頸的模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy'])
model.fit(partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_val, y_val))

print("=== PGM END ===")


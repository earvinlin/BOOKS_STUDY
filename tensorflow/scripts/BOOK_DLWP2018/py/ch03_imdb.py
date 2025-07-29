from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from keras.datasets import imdb

print("=== PGM START ===")

(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)

#print("train_data[]=", train_data[0])

word_index = imdb.get_word_index()

#print(word_index['this'])

reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
# 「?」: 指定值不存在時，返回此值
decode_review = ' '.join(
        [reverse_word_index.get(i-3, '?') for i in train_data[0]])
#print("decode_review=", decode_review)


#print("len=", len(decode_review))
#print("type=", type(decode_review))

# 3-2 
import numpy as np
def vectorize_sequences(sequences, dimension=10000) :
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 3-3
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 3-4
model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

# 3-5
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'])

# 3-6
from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy])

# 3-7
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 3-8
model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['acc'])
history = model.fit(partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())
# >>> manmini-m2 : dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

# 3.9
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#plt.show()

# 3.10
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

#plt.show()

# 3-11
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print("results= ", results)

print("=== PGM END ===")


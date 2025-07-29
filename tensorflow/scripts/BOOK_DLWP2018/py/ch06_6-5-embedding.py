# Listing 6.5 Instantiating an Embedding layer
# ref: https://stats.stackexchange.com/questions/270546/how-does-keras-embedding-layer-work
#      https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
from keras.layers import Embedding

# input_dim = 1000 ; outpu_dim = 64
embedding_layer = Embedding(1000, 64)
print(type(embedding_layer))

# 20240210 How to get embedding's shape ??
# print(embedding_layer.shape())

# Listing 6.6 Loading the IMDB data for use with an Embedding layer
from keras.datasets import imdb
# 20240115 新版已將該函數移至它處
#from keras import preprocessing
from keras.utils import pad_sequences

max_features = 10000
maxlen = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

#print(x_train[:2]) # contents to integer
#print(x_test[:2]) # contents to integer
#print(y_train[:2]) # comment , only 0 or 1

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences
#x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
#x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)


# Listing 6.7 Using an Embedding layer and classifier on the IMDB data
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential() # 全連接層
model.add(Embedding(10000, 8, input_length=maxlen))
#emb = Embedding(10000, 8, input_length=maxlen)
#print(type(emb))
#model.add(emb)

model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


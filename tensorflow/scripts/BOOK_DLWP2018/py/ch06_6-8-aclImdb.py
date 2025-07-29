# Listing 6.8 Processing the labels of the raw IMDB data
import os
import platform

if platform.system() == "Windows" :
#    imdb_dir = 'C:\\Workspaces\\Datasets\\aclImdb' 
    imdb_dir = 'D:\\Workspaces\\Datasets\\aclImdb' 
elif platform.system() == "Linux" :
    imdb_dir = '/home/earvin/workspaces/datasets/aclImdb'
else : # Mac path 
    imdb_dir = '/Users/earvin/workspaces/datasets/aclImdb'

train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
#        print('fname= ', fname)
        if fname[-4:] == '.txt':
            # on windows platform, 會因為編碼問題報錯，所以要加上指定編碼
            f = open(os.path.join(dir_name, fname), encoding="utf-8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                 labels.append(1)
#print("labels counts= ", len(labels))
#print("texts  counts= ", len(texts))

#print("labels[:10]= ", labels[:10])
#print("texts[:1]= ", texts[:1])
#print("texts[1:2]= ", texts[1:2])
#print("texts[2:3]= ", texts[2:3])



# Listing 6.9 Tokenizing the text of the raw IMDB datai
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer # 分詞器；tf v2.15 keras
#from keras.utils import Tokenizers
# 20240130 tf2.12 
#from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
import numpy as np

maxlen = 100               # 100個單詞後截斷評論
training_samples = 200     # 在200個樣本上訓練
validation_samples = 10000 # 在10000個樣本上驗證
max_words = 10000          # 只考慮數據集中前10000個最常見的單詞

tokenizer = Tokenizer(num_words=max_words)
#print("tokenizer= ", type(tokenizer));

"""
fit_on_texts() :
用來建立詞彙表，並計算每個詞在文本中的出現次數。這個方法會將輸入的文本轉換為詞典，並為
每個詞分配一個唯一的索引。
[Example]
from keras.preprocessing.text import Tokenizer

texts = ["I love cats", "I love dogs"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

print(tokenizer.word_index)  # {'i': 1, 'love': 2, 'cats': 3, 'dogs': 4}
print(tokenizer.word_counts)  # OrderedDict([('i', 2), ('love', 2), ('cats', 1), ('dogs', 1)])
"""
tokenizer.fit_on_texts(texts)
print("=== 顯示tokenizer屬性 ===") 
"""
document_count :
代表 處理過的文本數量。當你使用 Tokenizer.fit_on_texts(texts) 方法來適配文本時，
document_count 會記錄 texts 中的文本數量。

word_docs :
在 Keras 的 Tokenizer 類中，word_docs 是一個字典（dictionary），它記錄了每個詞（word）
在多少個文本（documents）中出現過。
[EXAMPLE]
word_docs=  defaultdict(<class 'int'>, {'themes': 369, 'i': 19230, 'aside': 445, ...})
"""
#print("document_count= ", tokenizer.document_count)
#print("word_docs= ", tokenizer.word_docs)

"""
texts_to_sequences() :
是一個方法，用來將文本轉換為 數字序列，其中每個單詞都會被映射到一個唯一的索引。
主要功能：
- 將文本轉換為數字序列：每個單詞會被替換為它在 word_index 字典中的索引。
- 忽略未出現在 word_index 中的詞：如果某個詞沒有在 Tokenizer.fit_on_texts() 訓練過的
  文本中出現，它將被忽略。
"""
sequences = tokenizer.texts_to_sequences(texts)
#print("sequences= ", len(sequences))
#print("sequence[0]= ", sequences[0])
#print("sequence[1]= ", sequences[1])

"""
word_index :
是一個字典（dictionary），它將每個詞映射到一個唯一的索引數字。這個索引是根據 
fit_on_texts() 方法處理的文本建立的，並且可以用來將文本轉換為數字序列。
- 建立詞彙表：每個詞都會被分配一個唯一的索引。
- 用於文本數字化：可以將文本轉換為數字序列，以便機器學習模型處理。
- 忽略未出現在詞彙表中的詞：如果某個詞沒有在 fit_on_texts() 訓練過的文本中出現，它將不
  會被包含在 word_index 中。
[EXAMPLE]
from keras.preprocessing.text import Tokenizer

texts = ["I love cats", "I love dogs"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

print(tokenizer.word_index)  # {'i': 1, 'love': 2, 'cats': 3, 'dogs': 4}
"""
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 將數據劃分為訓練集和驗證集，首先要打亂數據
# 因為一開始數據中的樣本是排序好的
# (所有負面評論在前面，然後是所有正面評論)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]



# Listing 6.10 Parsing the GloVe word-embeddings file

if platform.system() == "Windows" :
    glove_dir = 'D:\\Workspaces\\Datasets\\glove.6B'
elif platform.system() == "Linux" :
    glove_dir = '/home/earvin/workspaces/datasets/glove.6B'
else :
    # Mac path (Not finished)
    glove_dir = '/home/earvin/workspaces/datasets/glove.6B'

embeddings_index = {}

# on windows platform, 會因為編碼問題報錯，所以要加上指定編碼
# f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf-8")

flag = 0

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32') # 後面100個預訓練的值
    embeddings_index[word] = coefs

    if flag < 2 :
        print("i= ", flag, ",word= ", word, ",coefs= ", coefs)
        flag += 1

f.close()

print('Found %s word vectors.' % len(embeddings_index))
"""
the:[-0.038194 -0.24487   0.72812  -0.39961   0.083172  0.043953 -0.39141
  0.3344   -0.57545   0.087459  0.28787  -0.06731   0.30906  -0.26384
 -0.13231  -0.20757   0.33395  -0.33848  -0.31743  -0.48336   0.1464
 -0.37304   0.34577   0.052041  0.44946  -0.46971   0.02628  -0.54155
 -0.15518  -0.14107  -0.039722  0.28277   0.14393   0.23464  -0.31021
  0.086173  0.20397   0.52624   0.17164  -0.082378 -0.71787  -0.41531
  0.20335  -0.12763   0.41367   0.55187   0.57908  -0.33477  -0.36559
 -0.54857  -0.062892  0.26584   0.30205   0.99775  -0.80481  -3.0243
  0.01254  -0.36942   2.2167    0.72201  -0.24978   0.92136   0.034514
  0.46745   1.1079   -0.19358  -0.074575  0.23353  -0.052062 -0.22044
  0.057162 -0.15806  -0.30798  -0.41625   0.37972   0.15006  -0.53212
 -0.2055   -1.2526    0.071624  0.70565   0.49744  -0.42063   0.26148
 -1.538    -0.30223  -0.073438 -0.28312   0.37104  -0.25217   0.016215
 -0.017099 -0.38984   0.87424  -0.72569  -0.51058  -0.52028  -0.1459
  0.8278    0.27062 ]
"""

# Listing 6.11 Preparing the GloVe word-embeddings matrix
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim)) # matrix(10000,100) initialize 0
#print(embeddings_index)
for word, i in word_index.items():
#    if i < 10 :
#        print(word, i)
#        print("word: ", word)
#        print(", value: ", embeddings_index.get(word))
        
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            if i < 10 :
                print("word: ", word)
                print(", value: ", embeddings_index.get(word))
            embedding_matrix[i] = embedding_vector


# Listing 6.12 Model definition
from keras.models import Sequential 
from keras.layers import Embedding, Flatten, Dense

model = Sequential() 
model.add(Embedding(max_words, embedding_dim, input_length=maxlen)) 
model.add(Flatten()) 
model.add(Dense(32, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
model.summary()

# Listing 6.13 Loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# Listing 6.14 Training and evaluation
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32,
validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

# Listing 6.15 Plotting the results
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




    

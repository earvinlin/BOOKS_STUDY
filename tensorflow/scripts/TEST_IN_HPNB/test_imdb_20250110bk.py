from keras.datasets import imdb
import numpy as np

def vectorize_sequences(sequences, dimension=10000) :
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences) :
        results[i, sequence] = 1.
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#print("train_data  = ", train_data[0])
#print("train_labels= ", train_labels[0])

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
"""
將評論解碼。索引減3是因為0,1,2是為'padding'(填充)、'start of sequence'(序列開始)、
'unknown'(未知詞)分別保留的索引 
"""
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print("Before vectorize train_data[0]= ", train_data[0])
x_train = vectorize_sequences(train_data) # 訓練數據向量化
y_train = vectorize_sequences(test_data)  # 測試數據向量化
print("After vectorize x_train[0]= ", x_train[0])

# 標籤向量化
x_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

import numpy as np

# Listing 6.4 Word-level one-hot encoding with hashing trick (toy example)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
dimensionality = 1000
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))

print("samples= ", list(enumerate(samples)))
print("samples[0]= ", list(enumerate(samples[0].split())))
print("samples[1]= ", list(enumerate(samples[1].split())))

# 使用hash function計算位置
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

print("results.shape= ", results.shape)
#print(results[0:1:1])

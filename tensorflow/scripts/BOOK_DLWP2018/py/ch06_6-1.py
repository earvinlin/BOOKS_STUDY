# Listing 6.1 Word-level one-hot encoding (toy example)
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {} # 宣告一個字典變數

for sample in samples:
    print("sample= ", sample)
    for word in sample.split():
        print("word= ", word)
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10
"""
numpy zeros() :
shape：定義傳回陣列的形狀
dtype：產生矩陣的資料型，可選參數，預設為numpy.float64
order：{'C'，'F'}，可選，預設：'C'，是否在內容中以行（C）或列（F）順序儲存多維資料。
"""
results = np.zeros(shape=(len(samples), 
                          max_length, 
                          max(token_index.values()) + 1))

print("results.shape= ", results.shape)
print("reslts= ", results)
print("token_index= ", token_index)
print("type(token_index)= ", type(token_index))
print("\n");

# one-hot encoding
for i, sample in enumerate(samples):
#    print("i, sample =", i, sample)
    for j, word in list(enumerate(sample.split()))[:max_length]:
#        print("j, word =", j, word)
        index = token_index.get(word)
#        print("index =", index)
        results[i, j, index] = 1.
#        print(results[i, j, index])

print("one-hot results= ",results)
print("\n\n");



# Listing 6.2 Character-level one-hot encoding (toy example)  one-hot編碼，逐字元
import string
np.set_printoptions(threshold=100000)

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# string.printable
# value : 「0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$ ...」
# 由被視為可打印符號的ASCII字符組成的字符串。
# 這是digits, ascii_letters, punctuation 和 whitespace 的總和。
characters = string.printable
#print("characters orders : ", characters)

# 20240205 Notes
# dict d = {key1 : value1, key2 : value2, ...} 
# zip(X, Y) => [(x1, y1), (x2, y2), ...]
# 20240121 這一行應該寫錯, ref dlwp_ch06_test.ipynb
#          output : character= T index=  None i=  0 ,j=  0
#token_index = dict(zip(range(1, len(characters) + 1), characters))
"""
{1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9', 
11: 'a', 12: 'b', 13: 'c', 14: 'd', 15: 'e', 16: 'f', 17: 'g', 18: 'h', 19: 'i', 20: 'j', 
21: 'k', 22: 'l', 23: 'm', 24: 'n', 25: 'o', 26: 'p', 27: 'q', 28: 'r', 29: 's', 30: 't', 
31: 'u', 32: 'v', 33: 'w', 34: 'x', 35: 'y', 36: 'z', 37: 'A', 38: 'B', 39: 'C', 40: 'D', 
41: 'E', 42: 'F', 43: 'G', 44: 'H', 45: 'I', 46: 'J', 47: 'K', 48: 'L', 49: 'M', 50: 'N', 
51: 'O', 52: 'P', 53: 'Q', 54: 'R', 55: 'S', 56: 'T', 57: 'U', 58: 'V', 59: 'W', 60: 'X', 
61: 'Y', 62: 'Z', 63: '!', 64: '"', 65: '#', 66: '$', 67: '%', 68: '&', 69: "'", 70: '(', 
71: ')', 72: '*', 73: '+', 74: ',', 75: '-', 76: '.', 77: '/', 78: ':', 79: ';', 80: '<', 
81: '=', 82: '>', 83: '?', 84: '@', 85: '[', 86: '\\', 87: ']', 88: '^', 89: '_', 90: '`', 
91: '{', 92: '|', 93: '}', 94: '~', 95: ' ', 96: '\t', 97: '\n', 98: '\r', 99: '\x0b', 100: '\x0c'}
"""

token_index = dict(zip(characters, range(1, len(characters) + 1)))
"""
output :
{'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10, 'a': 
11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16, 'g': 17, 'h': 18, 'i': 19, 'j': 20, 'k': 
21, 'l': 22, 'm': 23, 'n': 24, 'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 'u': 
31, 'v': 32, 'w': 33, 'x': 34, 'y': 35, 'z': 36, 'A': 37, 'B': 38, 'C': 39, 'D': 40, 'E': 
41, 'F': 42, 'G': 43, 'H': 44, 'I': 45, 'J': 46, 'K': 47, 'L': 48, 'M': 49, 'N': 50, 'O': 
51, 'P': 52, 'Q': 53, 'R': 54, 'S': 55, 'T': 56, 'U': 57, 'V': 58, 'W': 59, 'X': 60, 'Y': 
61, 'Z': 62, '!': 63, '"': 64, '#': 65, '$': 66, '%': 67, '&': 68, "'": 69, '(': 70, ')': 
71, '*': 72, '+': 73, ',': 74, '-': 75, '.': 76, '/': 77, ':': 78, ';': 79, '<': 80, '=': 
81, '>': 82, '?': 83, '@': 84, '[': 85, '\\': 86, ']': 87, '^': 88, '_': 89, '`': 90, '{': 
91, '|': 92, '}': 93, '~': 94, ' ': 95, '\t': 96, '\n': 97, '\r': 98, '\x0b': 99, '\x0c': 100}
"""

#print(token_index)
#print(token_index.get(5))

max_length = 50
print("== init results ==")
results = np.zeros((len(samples), max_length, len(token_index) + 1)) # 初始化陣列值均為 0
#print(results)
print("results object type : ", results.shape)

print("== middle line ==")

for i, sample in enumerate(samples):
#    print("First loop : i, sample =", i, sample) # output --> First loop : i, sample = 0 The cat sat on the mat.
    for j, character in enumerate(sample):
#        print("Second Loop : j, character =", j, character)
        """
        output : 
          Second Loop : j, character = 0 T
          Second Loop : j, character = 1 h
          ...
        """
        index = token_index.get(character)
#        print("character=", character, ", index= ", index, ", i= ", i, ", j= ", j)
        results[i, j, index] = 1. # index == None, 整個陣列的值會被填入 1

print("== end line ==")
#print(results.shape)
print(results)



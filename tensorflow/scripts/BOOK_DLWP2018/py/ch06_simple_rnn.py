# Listing 6.21 Numpy implementation of a simple RNN
import numpy as np

timesteps = 100      # 輸入序列的時間步數
input_features = 32  # 輸入特徵空間的維度
output_features = 64 # 輸出特徵空間的維度

inputs = np.random.random((timesteps, input_features)) # 輸入數據：隨機噪聲，僅作為示例

state_t = np.zeros((output_features,)) # 初始狀態：全零向量

# 創建隨機的權重矩陣W、U、b
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,)) 

successive_outputs = []
for input_t in inputs: # input_t是形狀為 (input_features,) 向量
#   由輸入和當前狀態(前一個輸出)計算得到當前輸出
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b) 
#    print(output_t)

#   將這個輸出保存一個列表中
    successive_outputs.append(output_t)
    state_t = output_t

# 最終輸出是一個形狀為(timesteps, output_features)的二維張量
final_output_sequence = np.concatenate(successive_outputs, axis=0)

print("final_output_sequence= ", final_output_sequence)


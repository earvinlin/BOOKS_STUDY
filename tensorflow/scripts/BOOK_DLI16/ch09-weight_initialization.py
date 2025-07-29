import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.initializers import Zeros, RandomNormal, glorot_normal, glorot_uniform

n_input = 784 # 此密集層接收784個輸入值
n_dense = 256 # 256個神經元
b_init = Zeros() # 偏值的初始值
#w_init = RandomNormal(stddev=1.0) # 表示此值是從標準常態分佈中取樣
# use Glorot常態分佈
#w_init = glorot_normal()
# use Glorot均勻分佈
w_init = glorot_uniform()

model = Sequential()
model.add(Dense(n_dense,
            input_dim=n_input,
            kernel_initializer=w_init, # 設定權重初始值
            bias_initializer=b_init))

# 使用的激活函數 (為了方便修改程式碼，將此獨立出來)
#model.add(Activation('sigmoid'))
model.add(Activation('tanh'))
#model.add(Activation('relu'))



# 產生輸入資料
x = np.random.random((1, n_input))
# use predict()將輸入資料x傳入密集層，以前向傳播算出這一層所有神經元(256個)的激活值a
a = model.predict(x)

_ = plt.hist(np.transpose(a))

plt.show()

# To pdf.94 , 20250430




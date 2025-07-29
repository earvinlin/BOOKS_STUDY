import numpy as np

# 對於該矩陣可以認為有三個維度，即0, 1, 2
A = np.arange(24).reshape((2,3,4))
print("A.shape= ", A.shape)
print("A= ", A)

# 如果不改變原矩陣，那麼正常的參數順序是：(0,1,2)
T1 = A.transpose(0,1,2)
print("T1.shape= ", T1.shape)
print("T1= ", T1)
print("----------------------")

# 如果要置換第1和第2個維度，則參數順序是：(1,0,2)
T2 = A.transpose(1,0,2)
print("T2.shape= ", T2.shape)
print("T2= ", T2)
print("----------------------")

# 如果要置換第1和第3個維度，則參數順序是：(2,1,0)
T3 = A.transpose(2,1,0)
print("T3.shape= ", T3.shape)
print("T3= ", T3)
print("----------------------")

# 轉置 (2,3,4) -- after call transpose() --> (4,3,2)
print("A.shape= ", A.shape)
print("A= ", A)
A1 = np.transpose(A)
print("A1.shape= ", A1.shape)
print("A1= ", A1)

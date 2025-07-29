"""
numpy.linalg.det() : 用來計算矩陣的行列式
  param.
    a(N, M, M) array_like
    要計算的行列式的矩陣，這個輸入形狀要求最後的兩個維度要相等
"""
import numpy as np

a = np.arange(4).reshape(2,2)
e = np.eye(4)

print("a= ", a)
print("e= ", e)

print("det(e)= ", np.linalg.det(e))
print("det(a)= ", np.linalg.det(a))



from sympy import *
from sympy.plotting import plot3d
from scipy.stats import binom
from scipy.stats import beta
import numpy as np

#-- Sample 4-8 --#
from numpy import array

v = array([1, 1])
i_hat = array([2, 1])
j_hat = array([0, 3])

print("=== case 1 ===")
basis = array([i_hat, j_hat])
print(basis)
new_v0 = basis.dot(v)
print(new_v0)


"""
numpy.transpose(a, axes=None)
a : input array, 輸入數組
axes : 可選，整型list。默認情況下，反轉維度，
       否則根據給定的值對軸進行排列。
"""
a1 = array([[1,2,3], [4,5,6]])
print("Before transpose: ", a1)
a2 = a1.transpose()
print("After transpose: ", a2)


print("=== case 2 ===")
basis = array([i_hat, j_hat]).transpose()
print(basis)
basis.dot(v)
new_v1 = basis.dot(v)
print(new_v1)

#-- Sample 4-5 --#
v = array([3,2])
w = array([[2],[-1]])
print("v data: ", v, ", v shap: ", v.shape)
print("w data: ", w, ", w shap: ", w.shape)

print("------------------")

v1 = np.transpose(v)
print("v1 transpose: ", v1)
print("v1 transpose shape: ", v1.shape)

#v_plus_w = v + w
#print("v+w= ", v_plus_w)

#-- Sample 4-11 --#
print("\n#-- Sample 4-11 --#")
from numpy import array
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()
print("transform1= \n", transform1)

i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()
print("transform2= \n", transform2)

combined = transform2 @ transform1
print("COMBINED MATRIX:\n {}".format(combined))

v = array([1, 2])
print("combined.dot(v) : ",combined.dot(v)) # [-1, 1]

#-- 矩陣的點積是不可交換的，即不可能翻轉順序並期望得到相同結果 --#
rotated = transform1.dot(v)
sheared = transform2.dot(rotated)
print("        sheared : ", sheared)

#-- Sample 4-12 反向應用轉換 --#
print("\n#-- Sample 4-12 --#")
from numpy import array
i_hat1 = array([0, 1])
j_hat1 = array([-1, 0])
transform1 = array([i_hat1, j_hat1]).transpose()
print("transform1= \n", transform1)

i_hat2 = array([1, 0])
j_hat2 = array([1, 1])
transform2 = array([i_hat2, j_hat2]).transpose()
print("transform2= \n", transform2)

combined = transform1 @ transform2
print("COMBINED MATRIX:\n {}".format(combined))

v = array([1, 2])
print("combined.dot(v) : ",combined.dot(v)) # [-1, 1]


"""
#-- Sample 4-13 計算行列式 --#
det() : 
計算矩陣的行列式，要求輸入計算的行列式的矩陣，輸入形狀要求最後的兩個維度相等，
並且返回形狀為N的行列式，需要保證輸入的矩陣形狀最後的兩個維度
  Parameters :
    a(N, M, M) array_like
  Returns :
    det(N) array_like
det A = |a b| = ad - bc
        |c d|
    
transpose(a, axes=None) :
  parameter :
    a : input array
    axes : opt. list. 預設情況下，反轉維度，否則根據給定的值對軸進行排列。
    ex :
    (1) no param.
        shape (2, 3, 4) --> (4, 3, 2)
"""
from numpy.linalg import det
from numpy import array

i_hat = array([3, 0])
j_hat = array([0, 2])

basis = array([i_hat, j_hat]).transpose()
determinant = det(basis)
print(determinant) # print 6.0

#-- Sample 4-14 --#
print("\n#-- Sample 4-14 --#")

from numpy.linalg import det
from numpy import array

i_hat = array([1, 0, 2])
j_hat = array([1, 1, 1])
print(i_hat.shape, j_hat.shape)

print(array([i_hat, j_hat]).shape, array([i_hat, j_hat]))
basis = array([i_hat, j_hat]).transpose()
print(basis.shape, basis)

#determinant = det(basis)
#print(determinant) # print 1.0


#-- Sample 4-16 --#
print("\n#-- Sample 4-16 --#")

from numpy.linalg import det
from numpy import array

i_hat = array([-2, 1])
j_hat = array([3, -1.5])

basis = array([i_hat, j_hat]).transpose()
determinant = det(basis)

print(determinant) # print 0.0


#-- Sample 4-17 --#
print("\n#-- Sample 4-17 --#")
from sympy import *

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72

A = Matrix([[4, 2, 4], [5, 3, 7], [9, 3, 6]])

# A和它的反矩陣之間的點積會產生單位矩陣
inverse = A.inv()
identity = inverse * A

# print Matrix([[-1/2, 0, 1/3], [11/2, -2, -4/3], [-2, 1, 1/3]])
print("INVERSE: {}".format(inverse))

# print Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print("IDENTITY: {}".format(identity))


#-- Sample 4-18 --#
print("\n#-- Sample 4-18 --#")
from numpy import array
from numpy.linalg import inv

# 4x + 2y + 4z = 44
# 5x + 3y + 7z = 56
# 9x + 3y + 6z = 72

A = array([[4, 2, 4], [5, 3, 7], [9, 3, 6]])
B = array([44, 56, 72])
X = inv(A).dot(B)
print(X)  # [2. 34. -8.]


#-- Sample 4-19 --#
# 使用SymPy求解方程組
print("\n#-- Sample 4-19 --#")
from sympy import * 

A = Matrix([[4, 2, 4], [5, 3, 7], [9, 3, 6]])
B = Matrix([44, 56, 72])
X = A.inv() * B
print(X)  # Matrix([[2], [34], [-8]])


#-- Sample 4-20 --#
# 在NumPy中執行特徵分解
print("\n#-- Sample 4-20 --#")
from numpy import array, diag
from numpy.linalg import eig, inv

A = array([[1, 2], [4, 5]])
eigenvals, eigenvecs = eig(A)

print("EIGENVALUES")
print(eigenvals)
print("\nEIGENVECTORS")
print(eigenvecs)

"""
EIGENVALUES
[-0.46410162  6.46410162]

EIGENVECTORS
[[-0.80689822 -0.34372377]
 [ 0.59069049 -0.9390708 ]]
"""


#-- Sample 4-21 --#
# 在NumPy中分解和重組矩陣
print("\n#-- Sample 4-21 --#")
from numpy import array, diag
from numpy.linalg import eig, inv

A = array([[1, 2], [4, 5]])
eigenvals, eigenvecs = eig(A)

print("EIGENVALUES")
print(eigenvals)
print("\nEIGENVECTORS")
print(eigenvecs)

print("\nREBUILD MATRIX")
Q = eigenvecs
R = inv(Q)
L = diag(eigenvals)
B = Q @ L @ R
print(B)

"""
EIGENVALUES
[-0.46410162  6.46410162]

EIGENVECTORS
[[-0.80689822 -0.34372377]
 [ 0.59069049 -0.9390708 ]]

REBUILD MATRIX
[[1. 2.]
 [4. 5.]]
"""















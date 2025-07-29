import numpy as np
import pandas as pd

def naive_vector_dot(x, y) :
    z = 0.
    for i in range(x.shape[0]) :
        z += x[i] * y[i]
    return z


def naive_matrix_vector_dot(x, y) :
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            z[i] += x[i, j] * y[j]
    return z


def naive_matrix_dot(x, y) :
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]) :
        for j in range(y.shape[1]) :
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


print("=== START TESTING ===")
            
x = np.array([[1,2,3],[4,5,6]])
y = np.array([2,1,1])

z = naive_matrix_vector_dot(x, y)
print(z)


print("=== TEST 2 ===")


x1 = np.array([
        [1,2,3,4],
        [2,2,3,1],
        [3,1,5,3]]
        )
y1 = np.array(
        [[3,3],
        [4,1],
        [5,3],
        [3,3]])
z1 = naive_matrix_dot(x1, y1)
print(z1)


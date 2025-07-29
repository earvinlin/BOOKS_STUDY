import numpy as np
import pandas as pd

inf = 99999999
e = np.zeros((10,10), dtype = np.int16)

n, m = map(int, input().split(' '))

#initialize
for i in range(1, n+1):
	for j in range(1, n+1):
		if i==j:
			e[i][j]=0
		else:
			e[i][j]=inf

#read line
for i in range(1, m+1):
	t1, t2, t3 = map(int, input().split(' '))
	e[t1][t2]=t3

#Floyd-Warshall algorithm
for k in range(1, n+1):
	for i in range(1, n+1):
		for j in range(1, n+1):
			if e[i][j] > e[i][k]+e[k][j]:
				e[i][j] = e[i][k]+e[k][j]

# test data:
# 4 8
# 1 2 2
# 1 3 6
# 1 4 4
# 2 3 3
# 3 1 szsazsa
# 3 4 1
# 4 1 5
# 4 3 12

#output final consequences
for i in range(1, n+1):
    print(e[i][1:n+1])

# ---------------------
# [0 2 5 4]
# [9 0 3 4]
# [6 8 0 1]
# [ 5  7 10  0]
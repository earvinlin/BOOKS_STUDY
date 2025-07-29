import numpy as np
import pandas as pd


ary = [[0,3,1,999,999],
       [3,0,1,7,6],
       [1,1,0,5,2],
       [999,1,5,0,4],
       [999,6,2,4,0]
       ]

def FloydWarshallAlgorithm(ary, k) :
    dist = ary
    print("in fun: ", ary)

    for i in range(5) :
        for j in range(5) :
            if dist[i][j] > dist[i][k] + dist[k][j] :
                dist[i][j] = dist[i][k] + dist[k][j]

    return dist

for k in range(5) :
    aa = FloydWarshallAlgorithm(ary,k)
    print("k=", k, "array=", aa)

#print(FloydWarshallAlgorithm(FloydWarshallAlgorithm(ary,3),4))


"""
    for k in range(5) :
        for i in range(5) :
            for j in range(5) :
                if dist[i][j] > dist[i][k] + dist[k][j] :
                    dist[i][j] = dist[i][k] + dist[k][j]
"""
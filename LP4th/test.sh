#!/usr/bin/env python3
# --### !/usr/bin/python3 

import sys
import os

print("python version: ", sys.version)

def gen() :
    for i in range(10) :
        X = yield i
        print (X)

print("init gen()")
G = gen()

#print(next(G))
#print(next(G))
#print(next(G))



print("call next")
print(next(G))

print("call send")
print(G.send(77))

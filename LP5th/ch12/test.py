import sys
import os

def fun1():
    print("== fun1() ==")
    return 0 

def fun2():
    print("** fun2() **")
    return 1 

#f1 = fun1()
#f2 = fun2()

print("Run Start.")

#if f1 or f2 :
if fun1() or fun2() :
#    print(str(fun1()))
    print("In")

print("Run End.")


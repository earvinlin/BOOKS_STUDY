import sys
import os
from ch20 import myTimer as mt
from ch20 import myTimer2 as mt2
from ch20 import timeseqs as ts
from ch20 import inter2 as in2

def testFun(n) :
    i = 0
    j = range(n)
    for i in range(n) :
        i+= i
#        print(i)
    return i

mt.reps = 10000
mt.repslist = range(mt.reps)
print(mt.timer(testFun, 100))

# ============================== #
"""
print(sys.version)
for test in (ts.forLoop, ts.listComp, ts.mapCall, ts.genExpr, ts.genFunc) :
    elapsed, result = mt.timer(test)
    print('-' * 33)
    print('%-9s: %.5f => [%s ... %s]' % (test.__name__, elapsed, result[0], result[-1]))
"""
# ============================== #
"""
print("=== timer2 ===")
for test in (ts.forLoop, ts.listComp, ts.mapCall, ts.genExpr, ts.genFunc) :
    elapsed, result = mt2.timer(test)
    print('-' * 33)
    print('%-9s: %.5f => [%s ... %s]' % (test.__name__, elapsed, result[0], result[-1]))
"""
# (20230726) =================== #
"""
def func(*pargs) :
    print(pargs)

# return a tuple
func(*open('stockslist.txt'))

L = [1,2,3,4,5]
func(L)

D = {'a' : 1, 'b' : 2, 'c' : 3}
func(D)
"""

"""
inputfile = open('stockslist.txt')
for line in inputfile :
    print(line)
"""

"""
def func2(a, b=2, c=3, *d) :
    print(a,b,c,d)

# *vars : 會自動解包以對應呼叫函數的參數
func2(*L)

func2(*open('stockslist.txt'))


def func3(a, b, c, d) : print(a, b, c, d)
args = (1, 2)
args += (3, 4)
func3(*args)
"""

#
"""
def func5(a, *b, **c) : print(a, b, c)

L = [1,2,3,4]
func5(func2(*L), L)
"""

# python學習手冊4th(中文版) p.465~p.
"""
L = ['a', 'b', 'c', 'z', 'h', 'a', 'i', 'b']
L = 'abcdegsah'
L1 = 'abdfeig'
L2 = 'dfeadfg'
L3 = 'opqij'
args = (L1,L2,L3)

R = in2.intersect(*args)
print(R)
#=======
R2 = in2.intersect(L1, L3)
print(R2)
"""


"""
# p.475
def mysum0(L) :
    return 0 if not L else L[0] + mysum0(L[1:])

def mysum1(L) :
    return [0] if len(L) == 1 else L[0] + mysum1(L[1:])

def mysum2(L) :
    first, *rest = L 
    return first if not rest else first + mysum2(rest)

# p.477
def sumtree(L) :
    tot = 0
    for x in L :
        if not isinstance(x, list) :
            tot += x
        else :
            tot += sumtree(x)
    return tot

L = [1, [2, [3, 4], 5], 6, [7, 8]]
print("sumtree= ", sumtree(L))

# p.478
def echo(message) :
    print(message)

x = echo
x('Indirect call!!!')

def indirect(func, arg) :
    func(arg)

indirect(echo, 'Argument call~~')

schedule = [(echo, 'Spam!'), (echo, 'Ham!')]
for (func, arg) in schedule :
    func(arg)

# p.479 (我不大能理解此處的用法…)
def make(label) :
    def echo(message) :
        print(label + ':' + message)
    return echo

F = make('Spam')
F('Ham!')
F('Eggs!')
<<<<<<< HEAD

"""

print("\n=== START FROM HERE ===\n")

# p.585
#import ch23.spam
#print(ch23.spam)

from ch23 import spam
print(spam)

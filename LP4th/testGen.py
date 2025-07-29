import os

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

print("call send again")
print(G.send(88))

print("call next again")
print(next(G))


#==============================

G1 = (x ** 2 for x in range(4))

print(next(G1))
print(next(G1))
print(next(G1))
print(next(G1))

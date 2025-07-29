import os

def gen() :
    for i in range(10) :
        X = yield i
    print(X)

G = gen()

print(next(G))
print(G.send(77))
print(G.send(88))
print(next(G))

#=== 生成器表達式 ===#
print("*** 生成器表達式 ***")
G1 = (x **2 for x in range(4))
print(G1)

#next(G1)
#next(G1)
#next(G1)
#next(G1)

print(next(G1))
print(next(G1))
print(next(G1))
print(next(G1))


#=== use for loop ===#
print("*** use for loop ***")
for num in (x ** 2 for x in range(4)) :
    print('%s, %s' % (num, num / 2.0))


#=== auto iteration (ex: list fun.) === #
print("*** auto iteration (ex: list fun.) ***")
def timefour(S) :
    for c in S :
        yield c * 4

G2 = timefour('abcd')
res = list(G2)
print(res)


#=== auto iteration (ex: list fun.) === #
print("*** manual iteration (ex: list fun.) ***")
MG = timefour('hand')
I = iter(MG)
print(next(I))
print(next(I))

#=== Test 迭代器 ===#
print("~~~ Test 迭代器 ~~~")
G4 = (c * 4 for c in "SPAM")
I4_1 = iter(G4)
print(next(I4_1))
I4_2 = iter(G4)
print(next(I4_2))
print(next(I4_1))
print(next(I4_1))
I4_3 = iter(G4)
print(next(I4_3))


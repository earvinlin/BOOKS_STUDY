import os

# 呼叫函式的時候，如果在可迭代物件的引數前面加上 * 符號，
# 可以進行解包（unpack）：將可迭代物件內的每個內容物各自成為引數。
# 如果呼叫 func(my_list)，函式只會收到一個引數
# 如果呼叫 func(*my_list)，該 list 長度多長、函式就收到幾個引數
def mymap(func, *seqs) :
    return [func(*args) for args in zip(*seqs)]

print(mymap(abs, [-2, -1, 0, 1, 2]))
print("==")
print(mymap(pow, [1, 2, 3], [2, 3, 4, 5]))

def mymapPad(*seqs, pad=None) :
    seqs = [list(S) for S in seqs]
    res = []
#    print(seqs)
    while any(seqs) :
        res.append(tuple((S.pop(0) if S else pad) for S in seqs))
    return res

S1, S2 = 'abc', 'xyz123'
print(mymapPad(S1, S2, pad=99))

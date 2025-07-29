import os

def myzip(*seqs) :
    seqs = [list(S) for S in seqs]
    res = []
    while all(seqs) :
#        res.append(tuple(S.pop(0) for S in seqs))
#    return res
        yield tuple(S.pop(0) for S in seqs)

def mymapPad(*seqs, pad = None) :
    seqs = [list(S) for S in seqs]
    res = []
    while any(seqs) :
#        res.append(tuple((S.pop(0) if S else pad) for S in seqs))
#    return res
        yield tuple((S.pop(0) if S else pad) for S in seqs)

S1, S2 = 'abc', 'xyz123'

#print(myzip(S1, S2))
#print(mymapPad(S1, S2))
#print(mymapPad(S1, S2, pad = 99))

print(list(myzip(S1, S2)))
print(list(mymapPad(S1, S2)))
print(list(mymapPad(S1, S2, pad = 99)))

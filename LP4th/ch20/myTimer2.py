"""

"""
import time, sys


if sys.platform[:3] == 'win' :
    timefunc = time.process_time()
else :
    timefunc = time.time()

def trace(*args) : pass
#def trace(*args) : print("v=", *args)

def timer(func, *pargs, **kargs) :
    _reps = kargs.pop('_reps', 1000)
    trace(func, pargs, kargs, _reps)
    repslist = range(_reps)
    start = timefunc 
#    start = time.process_time()

    for i in repslist :
        ret = func(*pargs, **kargs)
    elapsed = timefunc - start
#    elapsed = time.process_time() - start
    return (elapsed, ret)

def best(func, *pargs, **kargs) :
    _reps = kargs.pop('_reps', 50)
    best = 2 ** 32
    for i in range(_reps) :
        (time, ret) = timer(func, *pargs, _reps = 1, **kargs)
        if time < best : best = time
    return (best, ret)

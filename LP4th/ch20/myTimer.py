import time

reps = 1000
repslist = range(reps)

"""
計算函數的執行時間
return tuple (執行時間, 函數回傳值)
"""
def timer(func, *pargs, **kargs) :
    start = time.process_time()
    for i in repslist :
        ret = func(*pargs, **kargs)
    elapsed = time.process_time() - start
    return (elapsed, ret)
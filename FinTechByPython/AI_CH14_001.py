import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
#import matplotlib.pylab as mp
#import seaborn as sns

Norm = np.random.normal(0, 1, size = 25)
a = np.arange(10)
#print(type(a))
#print(type(Norm))
print(stats.describe(Norm))

print(Norm.mean())

# 繪圖
#plt.hist(Norm)
#plt.show()



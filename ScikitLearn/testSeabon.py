import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

dat = [14.3, 15.8, 14.6, 16.1, 12.9, 15.1, 17.3, 14.0, 14.5, 13.9, 16.2, 14.3, 14.6, 13.3, 15.5, 11.8, 14.8, 13.5, 16.3, 15.4, 15.5, 13.9, 10.7, 14.8, 12.9, 15.4]
sns.set(style="whitegrid")
ax = sns.boxplot(x = dat, orient = "v", color = "skyblue", width=0.2)  # 畫盒圖
ax = sns.swarmplot(x = dat, orient = "v", color = "red")   # 加上資料點
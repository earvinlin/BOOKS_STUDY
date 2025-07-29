import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

# 20220607 檔案路徑不對
#TRD_Index = pd.read_table('.\_EXAMPLES\part2\index.csv', sep = '\t')

TRD_Index = pd.read_table('index.csv', sep = '\t')
TRD_Index.index = pd.to_datetime(TRD_Index.Date)
Taiex = TRD_Index[TRD_Index.CoName == 'TSE Taiex    '] # 注意要有4個半形空白(因為資料內容有4個空白)

#print(Taiex)

# 繪製台灣加權指數收益率的直方圖
TaiexRet = Taiex.ROI.astype(np.float64)
#TaiexRet.hist()
#plt.show()

# 求出台灣加權指數的收益率的均值
mu = TaiexRet.mean()
sigma = TaiexRet.std()
plt.hist(TaiexRet, density = True)
plt.plot(np.arange(-4, 4, 0.002), stats.norm.pdf(np.arange(-4, 4, 0.002), mu, sigma))
#plt.show()

# 進行區間估計
print(stats.t.interval(0.95, len(TaiexRet) - 1, mu, stats.sem(TaiexRet)))


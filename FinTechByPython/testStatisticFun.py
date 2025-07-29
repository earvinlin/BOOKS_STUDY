import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

"""
#import csv
# 開啟 CSV 檔案
with open('tsec.csv', newline='') as csvfile:
#    rows = csv.reader(csvfile)
    # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
    rows = csv.DictReader(csvfile)    
    for row in rows:
        print(row)
"""

tsec = pd.read_csv('tsec.csv', sep = ',')
end_prices = tsec['end_price']
print(end_prices.describe())
plt.hist(end_prices)

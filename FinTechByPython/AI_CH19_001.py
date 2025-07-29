import pandas as pd
import numpy as np

stock = pd.read_csv('2330.csv', index_col = 'Date', sep = '\t')
print("stock type is ", type(stock))
close = stock.Close
#print(close.head())
print(close)

close.index = pd.to_datetime(close.index)
print("data type: ", type(close))
print(close.index)

lagclose = close.shift(1)
print("== lagclose : ", lagclose, "\ntype :", type(lagclose))
# 計算單期簡單收益率
simpleret = (close - lagclose) / lagclose
simpleret.name = 'simpleret'
#print(simpleret.head())
print(simpleret)



#####################
# 計算2期簡單收益率 #
#####################
'''
simpleret2 = (close - close.shift(2)) / close.shift(2)
simpleret2.name = 'simpleret2'
print(simpleret2.head())

annualize = (1+simpleret).cumprod()
print("== Test cumprod() ==")
print(annualize)

annualize = (1+simpleret).cumprod()[-1]
print("== Test cumprod()[-1] ==")
print(annualize)

annualize = (1+simpleret).cumprod()[-1:]**(245/311)-1
print("== Test cumprod()[-1:]**(245/311)-1 ==")
print(annualize)
'''
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets, preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import time
import numpy as np
import pandas as pd
import mplfinance as mpf
from talib import abstract
from datetime import datetime

df =pd.read_csv('1101.csv')
#print(df)



# # Get Market Data
def GetKline(pair, symbol, interval, startTime = None, endTime = None):
    df = pair
#    df = pd.DataFrame(pair, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df.date = pd.to_datetime(df.date)
    df.set_index("date", inplace=True)
    df = df.astype(float)
    return df

"""
def GetHistoricalKline(url, symbol, interval, startTime):
    # init
    klines = GetKline(url, symbol, interval)
    tmptime = ToMs(klines.iloc[0].name)
    
    # Send request until tmptime > startTime
    while tmptime > startTime:
        tmptime -= PeriodToMs(interval) * 1000 # tmp minus period ms plus 1000 (1000 K)
        if tmptime < startTime:
            tmptime = startTime
        tmpdata = GetKline(url, symbol, interval, tmptime)
        klines  = pd.concat([tmpdata, klines])

    return klines.drop_duplicates(keep='first', inplace=False)
"""

# Math Tools
def ToMs(date):
    return int(time.mktime(time.strptime(str(date), "%Y-%m-%d %H:%M:%S")) * 1000) # Binance timestamp format is 13 digits

"""
def PeriodToMs(period):
    Ms = None
    ToSeconds = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }
    unit = period[-1]

    if unit in ToSeconds:
        try:
            Ms = int(period[:-1]) * ToSeconds[unit] * 1000
        except ValueError:
            pass
    return Ms
"""

def RSI(df, period):
    return abstract.RSI(df, timeperiod=period)


#sns.boxplot(data=df)
#plt.show()
symbol = '1101 Chart'

if __name__ == "__main__":
    klines = GetKline(df,'test', '1d')
    print(klines)
#    mpf.plot(klines)
#    mpf.plot(klines, type = 'candle') # 蠟燭圖
#    mpf.plot(klines, type = 'line') # 線圖
#    mpf.plot(klines, type = 'line', title = '1101 Chart')
#    mpf.plot(klines, type = 'candle', title = symbol, mav = 10)
    # 多組均線可以利用元組達成
#    mpf.plot(klines, type = 'candle', title = symbol, mav = (5, 10, 20))

    index  = mpf.make_addplot(RSI(klines, 14), panel = 2, ylabel = 'RSI')
    mpf.plot(klines, type = 'candle', title = symbol, addplot = [index], volume = True)
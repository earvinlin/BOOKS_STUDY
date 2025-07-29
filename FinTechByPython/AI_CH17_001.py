"""
import pandas as pd
import statsmodels.stats.anova as anova
from statsmodels.formula.api import ols

year_return = pd.read_csv('TRD_Year.csv', encoding = 'big5')

model = ols('Return ~ C(Industry)', data = year_return.dropna()).fit()
table1 = anova.anova_lm(model)
print(table1)
"""

import pandas as pd
import numpy as np
TRD_Index = pd.read_table('index.csv', sep = '\t')
TRD_Index.index = pd.to_datetime(TRD_Index.Date)
Taiex = TRD_Index[TRD_Index.CoName == 'TSE Taiex    ']
tw50 = TRD_Index[TRD_Index.CoName == 'TW 50 INDEX  ']
#print(Taiex)
#print(tw50)
#print(pd.concat([Taiex.ROI, tw50.ROI], axis = 1))

retData = pd.concat([Taiex.ROI, tw50.ROI], axis = 1).astype(np.float64)
retData = retData.dropna()
retData.columns = ['TAIEX', 'TW50']
print(retData)

import statsmodels.api as sm
model = sm.OLS(retData.TAIEX, sm.add_constant(retData.TW50)).fit()
print(model.summary())

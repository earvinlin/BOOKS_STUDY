from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

ds = datasets.load_iris()
print(ds.DESCR)

df =pd.DataFrame(ds.data, columns=ds.feature_names)
print(df)

y = ds.target   # 取得目標變數(Y)
print(y)

print(ds.target_names)  # 取得目標變數的類別名稱

print("== df.info() ==")
#print(df.info)  # 觀察資料集彙總資訊
df.info()

print("== df.describe() ==")
# 描述統計量
print(df.describe())

"""
print("== 顯示箱型圖 ==")
# 箱型圖 : 要使用matplotlib來顯示圖形
import matplotlib.pyplot as plt
import seaborn as sns 
sns.boxplot(data=df)
plt.show()

# 是否有含遺失值(Missing value)
print("== 是否有含遺失值(Missing value) ==")
print(df.isnull().sum())

# y 各類別資料筆數統計
print("== y 各類別資料筆數統計 ==")
sns.countplot(x=y)
plt.show()
"""

# 以pandas函數統計類別資料筆數
print("== 以pandas函數統計類別資料筆數 ==")
print(pd.Series(y).value_counts())

# 指定X，並轉為Numpy陣列
X = df.values
# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# 查看陣列維度
print("== 查看陣列維度 ==")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

scaler = preprocessing.StandardScaler()     # 依標準化(StandardScaler)類別建立物件
X_train_std = scaler.fit_transform(X_train) # 呼叫fit_transform，表先訓練，再作特縮放
X_test_std = scaler.transform(X_test)       # 僅呼叫transform，表測試資料不參與訓練，只作特徵縮放

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(X_train_std, y_train)
y_pred = clf.predict(X_test_std)
print("30筆測試資料預測結果(0~2為品種代碼)=\n ", y_pred)

# 計算準確率
print("計算準確率= ", f'{accuracy_score(y_test, y_pred)*100:.2f}%')

# 使用混淆矩陣
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# 混淆矩陣圖
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

disp =  ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=ds.target_names)
disp.plot()
plt.show()

# 模型儲存
import joblib

joblib.dump(clf, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')





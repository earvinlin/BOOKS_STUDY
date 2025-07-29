#-- Sample 5-1 --#
print("#-- Sample 5-1 --#")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cannot find the file
#df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",") # Not found
#df = pd.read_csv('Https://bit.ly/3cIH97A', delimiter=",") # OK
df = pd.read_csv('sample5-2_data.csv', delimiter=",") # 課本範例資料應該是這個才對

X = df.values[:, :-1]
Y = df.values[:, -1]
#print("X= ", X)
#print("Y= ", Y)

fit = LinearRegression().fit(X, Y)

m = fit.coef_.flatten()         # coef_      : 斜率
b = fit.intercept_.flatten()    # intercept_ : 截距
print("m = {0}".format(m))
print("b = {0}".format(b))

plt.plot(X, Y, 'o')
plt.plot(X, m*X+b)
plt.show()



#-- Sample 5-2 --#
print("#-- Sample 5-2 --#")
import pandas as pd

# Cannot find the file
#df = pd.read_csv('https://bit.ly/3go0Ant', delimiter=",") # Not found
#df = pd.read_csv('Https://bit.ly/3cIH97A', delimiter=",") # OK
points = pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples()

m = 1.93939394
b = 4.73333333

# 計算殘差
for p in points :
    y_actual = p.y
    y_predict = m * p.x + b
    residual = y_actual - y_predict
    print(residual)


#-- Sample5-4 --#
print("#-- Sample5-4 --#")
import pandas as pd

#df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",") # OK
points = pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples()

m = 1.93939
b = 4.73333

sum_of_squares = 0.0

#計算平方和
for p in points :
    y_actual = p.y
    y_predict = m * p.x + b
    residual_squared = (y_predict - y_actual)**2
    sum_of_squares += residual_squared
    
print("sum of squares = {}".format(sum_of_squares)) # 28.096969704500005


#-- Sample 5-5 --#
print("#-- Sample 5-5 --#")
import pandas as pd

points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",").itertuples())
#points = pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples()

n = len(points)

m = (n*sum(p.x*p.y for p in points) - sum(p.x for p in points) *
    sum(p.y for p in points)) / (n*sum(p.x**2 for p in points) -
    sum(p.x for p in points)**2)
b = (sum(p.y for p in points) / n) - m * sum(p.x for p in points) / n

print(m, b) # 1.9393939393939394 4.7333333333333325


#-- Sample 5-6 --#
print("#-- Sample 5-6 --#")
import pandas as pd
from numpy.linalg import inv
import numpy as np

# 匯入資料點
df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")

print("df= ", df, "\ndf.shape= ", df.shape)
"""
df= x   y
0   1   5
1   2  10
2   3  10
3   4  15
4   5  14
5   6  15
6   7  19
7   8  18
8   9  25
9  10  23
df.shape = (10,2)
"""

# 提取輸入變數(所有列、除了最後一行之外的所有行)
X = df.values[:, :-1].flatten()
print("X= ", X, "\nX.shape= ", X.shape)
# 添加占位符"1"行來產生截距 (np.ones(5) -> array([1., 1., 1., 1., 1.]))
X_1 = np.vstack([X, np.ones(len(X))]).T
print("X_1= ", X_1, "\nX_1.shape= ", X_1.shape)
# 提取輸出行(所有列、最後一行)
Y = df.values[:, -1]
print("Y= ", Y, "\nY.shape= ", Y.shape)

# 計算斜率和截距係數
b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
print(b)

#預測y值
y_predict = X_1.dot(b)


#-- Sample5-7 --#
"""
線性代數
使用矩陣X，並像前面一樣在它後面添加一個額外的1行以產生截距β0，然後再把它分為為兩個分量陣Q, R
X = Q * R
可以使用Q and R來找到陣形式b中的beta係數值
b = R(-1) * Q(T) * y  ### (-1), (T)表示上標
"""
print("#-- Sample5-7 --#")
import pandas as pd
from numpy.linalg import qr, inv
import numpy as np

#df = pd.read_csv('https://bit.ly/2KF29Bd', delimiter=",")
df = pd.read_csv('sample5-2_data.csv', delimiter=",")
print("df= ", df)

X = df.values[:, :-1].flatten()
print("X= ", X)

X_0 = np.vstack([X, np.ones(len(X))])
print("X_0= ", X_0)

X_1 = np.vstack([X, np.ones(len(X))]).transpose()
print("X_1= ", X_1)

Y= df.values[:, -1]
print("Y= ", Y)

Q, R = qr(X_1)
b = inv(R).dot(Q.transpose()).dot(Y)

print(b) # [1.93939394 4.73333333]


#-- Sample5-8 --#
print("#-- Sample5-8 --#")
import random

def f(x) :
    return (x - 3)**2 + 4

def dx_f(x) :
    return 2 * (x - 3)

# 學習率
L = 0.001
# 執行梯度下降之迭代次數
iterations = 100_000
# 從隨機的x開始
x = random.randint(-15, 15)

for i in range(iterations) :
    # 取得斜率
    d_x = dx_f(x)
    #透過減去 (學習率) * (斜率) 來更新 x
    x -= L * d_x

print(x, f(x))  # print 2.999999999999889 4.0


#-- Sample5-9 --#
print("#-- Sample5-9 --#")
import pandas as pd

df = pd.read_csv('sample5-2_data.csv', delimiter=",")

# 建構模型
m = 0.0
b = 0.0

# 學習率
L = .001

# 迭代次數
iterations = 100_000

n = float(len(points)) # X 的元素數量

# 執行梯度下降
for i in range(iterations) :
    # 對m的斜率
    D_m = sum(2 * p.x * ((m * p.x + b) - p.y) for p in points)
    # 對b的斜率
    D_b = sum(2 * ((m * p.x + b) - p.y) for p in points)
    # 更新m and b
    m -= L * D_m
    b -= L * D_b

print("y = {0}x + {1}".format(m, b)) 
# y = 1.9393939393939548x + 4.733333333333227


#-- Sample5-10 --#
print("#-- Sample5-10 --#")
from sympy import *

m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_squares = Sum((m*x(i)+b - y(i))**2, (i, 0, n))

d_m = diff(sum_of_squares, m)
d_b = diff(sum_of_squares, b)

print(d_m)
print(d_b)
# Output
# Sum(2*(b + m*x(i) - y(i))*x(i), (i, 0, n))
# Sum(2*b + 2*m*x(i) - 2*y(i), (i, 0, n))


#-- Sample5-11 --#
print("\n#-- Sample5-11 --#")
import pandas as pd
from sympy import *

df = pd.read_csv('sample5-2_data.csv', delimiter=",")

m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n))

d_m = diff(sum_of_squares, m) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)

d_b = diff(sum_of_squares, b) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)



#-- FOR TESTING START --#
"""
subs() usage : https://vimsky.com/examples/usage/python-sympy-subs-method-2.html
        subs方法是SymPy中常用的替代变量方法之一。它可以将符号变量替换为其他符号变量
        、数值或表达式。
doit() usage : https://vimsky.com/examples/usage/python-sympy-doit-method.html
        使用doit()simpy模組中的方法，我們可以評估預設未評估的對象，例如限制，積分，
        總和和乘積。請注意，將對所有此類物件進行遞歸求值。 
replace() usage : https://geek-docs.com/sympy/sympy-questions/139_sympy_sympy_subs_vs_replace_vs_xreplace.html#google_vignette
        replace方法是SymPy中另一个常用的替代变量方法。它功能类似于subs方法，可以将
        符号变量替换为其他符号变量、数值或表达式。
lambdify() usage : https://github.com/sympy/sympy/blob/2197797741156d9cb73a8d1462f7985598e9b1a9/sympy/utilities/lambdify.py#L187-L933
    sympy.utilities.lambdify.lambdify(args, expr, modules=None, printer=None, use_imps=True, dummify=False, cse=False, docstring_limit=1000)
    modules : str, optional
                Specifies the numeric library to use.
                If not specified, modules defaults to:
                ["scipy", "numpy"] if SciPy is installed
                ["numpy"] if only NumPy is installed
                ["math", "mpmath", "sympy"] if neither is installed.
"""
#d_m1 = diff(sum_of_squares, m).subs(n, len(points) - 1).doit()
d_m0 = diff(sum_of_squares, m)
d_m1 = diff(sum_of_squares, m).subs(n, len(points) - 1)
d_m2 = diff(sum_of_squares, m).subs(n, len(points) - 1).doit()
print("d_m0: ", d_m0)
print("d_m1: ", d_m1)
print("d_m2: ", d_m2)
#-- FOR TESTING END   --#



# use lambdify來編譯以加快計算
d_m = lambdify([m, b], d_m)
d_b = lambdify([m, b], d_b)

# 建構模型
m = 0.0
b = 0.0
# 學習率
L = .001
# 迭代次數
iterations = 100_000
# 執行梯度下降
for i in range(iterations) :
    # 更新執行梯度
    m -= d_m(m, b) * L
    b -= d_b(m, b) * L

print("y = {0}x + {1}".format(m, b))
# y = 1.939393939393954x + 4.733333333333231


#-- Sample5-12 --#
print("\n#-- Sample5-12 --#")
from sympy import *
from sympy.plotting import plot3d
import pandas as pd

points = list(pd.read_csv("sample5-2_data.csv").itertuples())
m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n)) \
    .subs(n, len(points)- 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)

plot3d(sum_of_squares)


#-- Sample5-13 --#
print("\n#-- Sample5-13 --#")
import pandas as pd
import numpy as np

# 輸入資料
data = pd.read_csv("sample5-2_data.csv", header=0)
#data = pd.read_csv("https://bit.ly/2KF29Bd", head=0)
print("data: ", data)
# 1 ~ 3 row 0 ~ 3 col
# df.iloc[1:3, 0:3]
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

print("X: ", X)
print("Y: ", Y)

n = data.shape[0]   # 列

# 建構模型
m = 0.0
b = 0.0

sample_size = 1     # 樣本大小
L = .0001           # 學習率
epochs = 1_000_000  # 執行梯度下降之迭代次數

#  執行隨機梯度下降
for i in range(epochs) :
    idx = np.random.choice(n, sample_size, replace=False)

x_sample = X[idx]
y_sample = Y[idx]

# Y 目前的預測值
Y_pred = m * x_sample + b

# 損失函數的d/dm微分
D_m = (-2 / sample_size) * sum(x_sample * (y_sample - Y_pred))

# 損失函數的d/db微分
D_b = (-2 / sample_size) * sum(y_sample - Y_pred)
m = m - L * D_m # 更新m
b = b - L * D_b # 更新b

# 印出進度
if i % 10000 == 0 :
    print(i, m, b)

print("y = {0}x + {1}".format(m, b))


#-- Sample5-14 --#
print("\n#-- Sample5-14 --#")
# 使用pandas來查看每對變數之間的相關係數
import pandas as pd

df = pd.read_csv("sample5-2_data.csv", delimiter=",")

# 印出變數間的相關係數
correlations = df.corr(method='pearson')
print(correlations)


#-- Sample5-15 --#
print("\n#-- Sample5-15 --#")
# 計算相關係數
import pandas as pd
from math import sqrt

points = list(pd.read_csv("sample5-2_data.csv").itertuples())
n = len(points)

#print("points= ", points, "; n= ", n)
numerator = n * sum(p.x * p.y for p in points) - \
    sum(p.x for p in points) * sum(p.y for p in points)
denominator = sqrt(n * sum(p.x**2 for p in points) - sum(p.x for p in points)**2) \
    * sqrt(n * sum(p.y**2 for p in points) - sum(p.y for p in points)**2)

corr = numerator / denominator

print(corr)


#-- Sample5-16 --#
print("\n#-- Sample5-16 --#")
# 從T分布計算臨界值
from scipy.stats import t

n = 10                
lower_cv = t(n-1).ppf(.025)
upper_cv = t(n-1).ppf(.975)

print(lower_cv, upper_cv)


#-- Sample5-17 --#
print("\n#-- Sample5-17 --#")
# 檢定看似線性資料的顯著性
from scipy.stats import t
from math import sqrt

n = 10
lower_cv = t(n-1).ppf(.025)
upper_cv = t(n-1).ppf(.975)

r = 0.957586

test_value = r / sqrt((1 - r**2) / (n - 2))

print("TEST VALUE: {}".format(test_value))
print("CRITICAL RANGE: {}, {}".format(lower_cv, upper_cv))

if test_value < lower_cv or test_value > upper_cv :
    print("CORRELATION PROVEN, REJECT H0")
else :
    print("CORRELATION NOT PROVEN, FAILED TO REJECT H0")

if test_value > 0 :
    p_value = 1.0 - t(n-1).cdf(test_value)
else :
    p_value = t(n-1).cdf(test_value)

p_value = p_value * 2
print("P-VALUE: {}".format(p_value))


#-- Sample5-18 --#
print("\n#-- Sample5-18 --#")
# 在Pandas中建立相關矩陣

df = pd.read_csv("sample5-2_data.csv")

# 印出變數間之相關性
coeff_determination = df.corr(method='pearson')**2
print(coeff_determination)


#-- Sample5-19 --#
print("\n#-- Sample5-19 --#")
# 計算估計標準誤差
import pandas as pd
from math import sqrt

points = list(pd.read_csv("sample5-2_data.csv").itertuples())
n = len(points)

# 迴歸線
m = 1.939
b = 4.733

# 計算估計標準誤差
S_e = sqrt((sum((p.y - (m * p.x + b))**2 for p in points)) / (n-2))
print(S_e)


#-- Sample5-20 --#
print("\n#-- Sample5-20 --#")
# 計算8.5歲狗的獸醫訪問次數預測區間
import pandas as pd
from scipy.stats import t
from math import sqrt

#points = list(pd.read_csv("sample5-2_data.csv").itertuples())
points = list(pd.read_csv('https://bit.ly/2KF29Bd', delimiter=',').itertuples())
n = len(points)

# 線性迴歸線
m = 1.939
b = 4.733

# 計算 x = 8.5 的預測區間
x_0 = 8.5
x_mean = sum(p.x for p in points) / len(points)
t_value = t(n-2).ppf(.975)
standard_error = sqrt(sum((p.y - (m*p.x+b))**2 for p in points) / (n-2))
margin_of_error = t_value * standard_error * \
    sqrt(1 + (1 / n) + (n * (x_0 - x_mean)**2) / \
         (n * sum(p.x**2 for p in points) - \
          sum(p.x for p in points)**2))
predicted_y = m * x_0 + b
print(predicted_y - margin_of_error, predicted_y + margin_of_error)






















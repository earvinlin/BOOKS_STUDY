import math

#-- Sample 6-1 --#
print("#-- Sample 6-1 --#")
def predict_probability(x, b0, b1) :
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p


#-- Sample 6-2 --#
print("\n#-- Sample 6-2 --#")
from sympy import *

b0, b1, x = symbols('b0 b1 x')
p = 1.0 / (1.0 + exp(-(b0 + b1 * x)))

p = p.subs(b0, -2.823)
p = p.subs(b1, 0.62)

print(p)
plot(p)


#-- Sample 6-3 --#
print("\n#-- Sample 6-3 --#")
# 在Scikit-learn中使用簡單的邏輯迴歸
import pandas as pd
from sklearn.linear_model import LogisticRegression

#df = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",")
df = pd.read_csv('sample6-3_data.csv', delimiter=",")

# 提取輸入變數(所有列，除了最後一行的所有行)
X = df.values[:, :-1]
# 提取輸出變數(所有列、最後一行)
Y = df.values[:, -1]

# 執行邏輯迴歸
# 關閉懲罰
model = LogisticRegression(penalty='none')
model.fit(X, Y)

# 印出beta1
print(model.coef_.flatten())      # 0.69267212
#印出beta0
print(model.intercept_.flatten()) # -3.17576395


#-- Sample 6-4 --#
print("\n#-- Sample 6-4 --#")
# 計算觀察給定邏輯迴歸的所有點的聯合概似度
import math
import pandas as pd

patient_data = pd.read_csv('sample6-3_data.csv', delimiter=",").itertuples()

b0 = -3.17576395
b1 = 0.69267212

def logistic_function(x) :
    p = 1.0 / (1.0 + math.exp(-(b0 + b1 * x)))
    return p

# 計算聯合概似度
joint_likelihood = 1.0

for p in patient_data :
    if p.y == 1.0 :
        joint_likelihood *= logistic_function(p.x)
    elif p.y == 0.0 :
        joint_likelihood *= (1.0 - logistic_function(p.x))

### another writing ###
"""
#-- Sample 6-5 --#
for p in patient_data :
    joint_likelihood *= logistic_function(p.x) ** p.y \
                        (1.0 - logistic_function(p.x)) ** (1.0 - p.y)
"""
print(joint_likelihood) # 4.7911180221699105e-05


#-- Sample 6-6 --#
print("\n#-- Sample 6-6 --#")
# 使用對數加法

# 計算聯合概似度
joint_likelihood = 0.0

for p in patient_data :
    joint_likelihood += math.log(logistic_function(p.x) ** p.y * \
                                 (1.0 - logistic_function(p.x) ** (1.0 - p.y)))
joint_likelihood = math.exp(joint_likelihood)

print(joint_likelihood)



"""
#-- Sample 6-8 --#
<<<執行後的錯誤訊息>>>
Traceback (most recent call last):
  File "D:\WORKSPACES\GithubProjects\BOOK_DSBM\testSymPy_ch06.py", line 109, in <module>
    joint_likelihood = sum(log((1.0 / (1.0 + exp(-(b0 + b1 * x(i))))) ** y(i) * \
TypeError: 'log' object is not iterable


#-- Sample 6-8 --#
print("\n#-- Sample 6-8 --#")
# 在邏輯迴歸中使用梯度下降
from sympy import *
import pandas as pd

#points = list(pd.read_csv("https://tinyurl.com/y2cocoo7").itertuples())
points = list(pd.read_csv('sample6-4_data.csv').itertuples())

b1, b0, i, n = symbols('b1 b0 i n')
x, y = symbols('x y', cls=Function)
joint_likelihood = sum(log((1.0 / (1.0 + exp(-(b0 + b1 * x(i))))) ** y(i) * \
                           (1.0 - (1.0 / (1.0 + exp(-(b0 + b1 * x(i)))))) ** (1 - y(i))), (i, 0, n))
# 對m進行偏微分，其中點被替換了
d_b1 = diff(joint_likelihood, b1) \
                    .subs(n, len(points) - 1).doit() \
                    .replace(x, lambda i: points[i].x) \
                    .replace(y, lambda i: points[i].y)

d_b0 = diff(joint_likelihood, b0) \
                    .subs(n, len(points) - 1).doit() \
                    .replace(x, lambda i: points[i].x) \
                    .replace(y, lambda i: points[i].y)

# 使用lambdify來編譯以計算地更快
d_b1 = lambdify([b1, b0], d_b1)
d_b0 = lambdify([b1, b0], d_b0)

# 執行梯度下降
b1 = 0.01
b0 = 0.01
L = .01

for j in range(10_000) :
    b1 += d_b1(b1, b0) * L
    b0 += d_b0(b1, b0) * L

print(b1, b0)
"""


#-- Sample 6-9 --#
print("\n#-- Sample 6-9 --#")
# 對員工資料進行多變數邏輯迴歸
import pandas as pd
from sklearn.linear_model import LogisticRegression

#employee_data = pd.read_csv("https://tinyurl.com/y6r7qjrp")
employee_data = pd.read_csv("sample6-5_data.csv")

# 抓取自變數行
inputs = employee_data.iloc[:, :-1]
# 抓取因變數"did_quit"行
output = employee_data.iloc[:, -1]
# 建構邏輯迴歸
fit = LogisticRegression(penalty='none').fit(inputs, output)
# 印出係數
print("COEFFICIENTS: {0}".format(fit.coef_.flatten()))
print("INTERCEPT: {0}".format(fit.coef_.flatten()))
#互動並測試新員工資料
def predict_employee_will_stay(sex, age, promotions, years_employed) :
    prediction = fit.predict([[sex, age, promotions, years_employed]])
    probabilities = fit.predict_proba([[sex, age, promotions, years_employed]])
    if prediction == [[1]] :
        return "WILL LEAVE: {0}".format(probabilities)
    else :
        return "WILL STAY: {0}".format(probabilities)
    
# 測試預測
while True :
    n = input("Predict employee will stay or leave {sex},{age},{promotions},{years employed} :")
    (sex, age, promotions, years_employed) = n.split(",")
    print(predict_employee_will_stay(int(sex), int(age), int(promotions), int(years_employed)))







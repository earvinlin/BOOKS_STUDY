from sympy import *
from sympy.plotting import plot3d
from scipy.stats import binom
from scipy.stats import beta

#-- Sample 3-9 機率密度函數(PDF) --#
def normal_pdf(x: float, mean: float, std_dev: float) -> float :
    return (1.0 / (2.0 * math.pi * std_dev ** 2) ** 0.5) * \
        math.exp(-1.0 *((x - mean) ** 2 / (2.0 * std_dev ** 2)))



"""
#-- Sample 3-15 --#
import random
import plotly.express as px

sample_size = 31
sample_count = 1000

# 中央極限定理，1000組樣本，每組包含31個
#介於 0.0 ~ 1.0之間的亂數
# i start with 0
x_values = [(sum([random.uniform(0.0, 1.0) for i in range(sample_size)]) / \
             sample_size) for _ in range(sample_count)]

y_values = [1 for _ in range(sample_count)]

px.histogram(x=x_values, y=y_values, nbins=20).show()
"""

#-- Sample 3-16 --#
print("#-- Sample 3-16 --#")

from scipy.stats import norm

def critical_z_value(p) :
    norm_dist = norm(loc = 0.0, scale = 1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 - ((1.0 - p) / 2.0)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)

print(critical_z_value(p = .95))
# (-1.959963984540054, 1.959963984540054)


# 信賴區間(confidence interval)
# 是一種範圍計算，顯示我們對樣本平均值(或其他參數)落在母體平均值範圍內的信心程度
#-- Sample 3-17 --#
# 以python從頭到尾計算信賴區間的方法
print("#-- Sample 3-17 --#")
from math import sqrt
from scipy.stats import norm

def critical_z_value(p) :
    # loc : mean ; scale : standard deviation
    norm_dist = norm(loc = 0.0, scale = 1.0)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 -((1.0 - p) / 2.0)
    # cpf : 累計分佈函數指定點的函數值
    # ppf : 累計分佈函數的逆函數(即分位點)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)

def confidence_interval(p, sample_mean, sample_std, n) :
    lower, upper = critical_z_value(p)
    print("aa=", lower, upper)
    lower_ci = lower * (sample_std / sqrt(n))
    upper_ci = upper * (sample_std / sqrt(n))
    print("bb=", lower_ci, upper_ci)
    return sample_mean + lower_ci, sample_mean + upper_ci

print(confidence_interval(p=.95, sample_mean=64.408, sample_std=2.05, n=31))
# (63.68635915701992, 65.12964084298008)

print(confidence_interval(p=.95, sample_mean=18, sample_std=1.5, n=40))
# (17.53515372577158, 18.46484627422842)

#-- Sample 3-18 --#
print("#-- Sample 3-17 --#")
from scipy.stats import norm
mean = 18
std_dev = 1.5
x = norm.cdf(21, mean, std_dev) - norm.cdf(15, mean, std_dev)
print(x)


#-- Sample 3-19 --#
print("#-- Sample 3-19 --#")
from scipy.stats import norm
mean = 18
std_dev = 1.5
x = norm.ppf(.05, mean, std_dev)
print("Sample 3-19 :", x)


#-- Sample 3-20 --#
print("#-- Sample 3-20 --#")
# 感冒的恢復時間平均為18天，標準差為1.5天
mean = 18
std_dev = 1.5
# 16天或更少天的機率
p_value = norm.cdf(16, mean, std_dev)

print("Sample 3-20 :", p_value)




"""
print("\n\n\n")
print("==============")
print("\n")
#-- Testing Area --#
import random

x_values = [(sum([random.uniform(0.0, 1.0) for i in range(31)]) / 31) for _ in range(10)]
#           ---------------------------------------------------------
print(x_values)
for i in range(11) :
    print(i)

y_values = [1 for _ in range(30)]
print(y_values)

"""

def critical_z_value_std1_5(p) :
    # loc : mean ; scale : standard deviation
    norm_dist = norm(loc = 0.0, scale = 1.5)
    left_tail_area = (1.0 - p) / 2.0
    upper_area = 1.0 -((1.0 - p) / 2.0)
    # cpf : 累計分佈函數指定點的函數值
    # ppf : 累計分佈函數的逆函數(即分位點)
    return norm_dist.ppf(left_tail_area), norm_dist.ppf(upper_area)
print(critical_z_value_std1_5(0.95))

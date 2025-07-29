"""
資料科學基礎數學
"""

from sympy import *

"""
# (p.8) sample 1-7 繪製線性函數
x = symbols('x')
f = 2*x + 1
plot(f)


# (p.9) sample 1-8 繪製指數函數
from sympy import *
x = symbols('x')
f = x**2 + 1
plot(f)


# (p.10) sample 1-9 宣告具有兩個自數的函數
from sympy.plotting import plot3d

x, y = symbols('x y')
f = 2*x + 3*y
plot3d(f)
"""

# (p.13) 
i, n = symbols('i n')

# 對從1 ~ n的所有i元素進行迭代，然後相乘並相加
summation = Sum(2*i,(i,1,n))

# 指明n為5，從數字1 ~ 5進行代
up_to_5 = summation.subs(n, 5)
print(up_to_5.doit())


# (p.15)
x = symbols('x')
expr = x**2 / x**5
print(expr)


# (P.20) Sample 1-14
p = 100
r = .20
t = 2.0

a = p * exp(r*t)
print(a)


# (p.23) use SymPy計算極限
from sympy import *

n = symbols('n')
f = (1 + (1/n))**n
result = limit(f, n, oo)

print("result= ", result)
print(result.evalf())


# (p.26) use Python實作的微分計算器
def derivative_x(f, x, step_size) :
    m = (f(x + step_size) - f(x)) / ((x + step_size) - x)
    return m

def my_function(x) :
    return x**2

slope_at_2 = derivative_x(my_function, 2, .00001)
print(slope_at_2)


# (p.27) Sample 1-18
x = symbols('x')
f = x**2
dx_f = diff(f)
print(dx_f)

# Sample 1-20 計算 x = 2 處的斜率
print(dx_f.subs(x, 2.1))


# (p.27) Sample 1-19
def f(x) :
    return x**2

def dx_f(x) :
    return 2*x

slope_at_2 = dx_f(2.0)
print(slope_at_2)


# (p.28) Sample 1-21
from sympy import *
from sympy.plotting import plot3d

x, y = symbols('x y')
f = 2*x**3 + 3*y**3
dx_f = diff(f, x)
dy_f = diff(f, y)

print("dx_f= ", dx_f)
print("dy_f= ", dy_f)

#plot3d(f)


# (p.30) Sample 1-22 使用極限來計算斜率
from sympy import *

x, s = symbols('x s')
f = x**2
slope_f = (f.subs(x, x + s) -f) / ((x+s) - x)
slope_2 = slope_f.subs(x, 2)
result = limit(slope_2, s, 0)
print(result)


# (p.30) Sample 1-23
from sympy import *
x, s = symbols('x s')
f = x**2
slope_f = (f.subs(x, x + s) - f) / ((x+s) - x)
result = limit(slope_f, s, 0)
print(result)

# (p.31) Sample 1-24 求 z 對 x 的微分
from sympy import *

z = (x**2 + 1)**3 - 2
dz_dx = diff(z, x)
print(dz_dx)

# (p.32) Sample 1-25
print("\n== Sample 1-25 ==")
from sympy import *

x, y = symbols('x y')
_y = x**2 + 1
dy_dx = diff(_y)
z = y**3 - 2
dz_dy = diff(z)

dz_dx_chain = (dy_dx * dz_dy).subs(y, _y)
dz_dx_no_chain = diff(z.subs(y, _y))

print(dz_dx_chain)
print(dz_dx_no_chain)


# (p.36) Sample 1-26
print("\n== Sample 1-26 ==")
def approximate_integral(a, b, n, f) :
    delta_x = (b - a) / n
    total_sum = 0

    for i in range(1, n + 1) :
#        midpoint = 0.5 * (2 * a + delta_x * (2 * i - 1))
        midpoint = a + delta_x * 0.5 + delta_x*(i-1)
        total_sum += f(midpoint)
    
    return total_sum * delta_x

def my_function(x) :
    return x**2 + 1

area = approximate_integral(a=0, b=1, n=5, f=my_function)
print(area)


# (p.37) Sample 1-29
print("\n== Sample 1-29 ==")
from sympy import *

x = symbols('x')
f = x**2 + 1

# 對x計算函數的積分；針對x = 0~1之間的區域
area = integrate(f, (x,0,1))
print(area) # output = 4/3


# (p.38) Sample 1-30
print("\n== Sample 1-30 ==")
from sympy import *
x, i, n = symbols('x i n')
f = x**2 + 1
lower, upper = 0, 1

delta_x = ((upper - lower) / n)
x_i = (lower + delta_x * i)
fx_i = f.subs(x, x_i)

n_rectangles = Sum(delta_x * fx_i, (i, 1, n)).doit()

area = limit(n_rectangles, n, oo)

print(area) # output = 4/3
 




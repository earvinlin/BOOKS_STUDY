from sympy import *

# (p.27) Sample 1-18
print("== Sample 1-18 ==")
x = symbols('x')
f = x**2
dx_f = diff(f)
print(dx_f)

# Sample 1-20 計算 x = 2 處的斜率
print("\n== Sample 1-20 ==")
print(dx_f.subs(x, 2))
print(dx_f.subs(x, 2.1))

# (p.27) Sample 1-19
print("\n== Sample 1-19 ==")
def f(x) :
    return x**2

def dx_f(x) :
    return 2*x

slope_at_2 = dx_f(2.0)
print(slope_at_2)

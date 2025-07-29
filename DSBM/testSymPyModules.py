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

#---------------------
# REF: https://hackmd.io/@yizhewang/Syj4ncoCm?type=view
#---------------------
x = symbols('x', communtative = True)
y = symbols('y', communtative = True)
f = symbols('f', cls = Function)
f = 3 * x**2 + 5 * y + 11
d_x = diff(f, x)
d_y = diff(f, y)

print("d_x= ", d_x)
print("d_y= ", d_y)

"""
symbols: 定義所用的符號，communtative = True 將符號設定為可交換，cls = Function 是將這些符號設定為函數的代號。
roots: 求多項式函數的根。
subs = {x:2}: 利用字典格式將函數中的 x 用 2 代入。
"""
print(f.evalf(subs = {x:2, y:1}))
print(d_x.evalf(subs = {x:2}))

#-- Example subs() Usage --#
from sympy import *
x, y = symbols('x y')
exp = x**2 + 1
print("Before Subs: {}".format(exp))

res_exp = exp.subs(x, y)
print("After Subs: {}".format(res_exp))
res_exp1 = exp.subs(x, 2)
print("After Subs: {}".format(res_exp1))

#--
iterations = 100_000
print("iterations: ", iterations)









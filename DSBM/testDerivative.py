def derivative_x(f, x, step_size) :
    m = (f(x + step_size) - f(x)) / ((x + step_size) - x)
    return m

def my_function(x) :
    return x**2

slope_at_2 = derivative_x(my_function, 2, .00001)

print(slope_at_2)

#-----------------------------------#
import pandas as pd
from sympy import *

#points = pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples()
points = list(pd.read_csv('sample5-2_data.csv', delimiter=",").itertuples())

df = pd.read_csv('sample5-2_data.csv', delimiter=",")
n = len(points)
m, b, i, n = symbols('m b i n')
x, y = symbols('x y', cls=Function)

sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n))

d_m = diff(sum_of_squares, m) \
    .subs(n, len(points) - 1).doit() \
    .replace(x, lambda i: points[i].x) \
    .replace(y, lambda i: points[i].y)

# use lambdify來編譯以加快計算
d_m = lambdify([m, b], d_m)

print(d_m)



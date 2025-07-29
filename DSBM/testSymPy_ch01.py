"""
sympy Python的科學計算庫
ref : https://oreil.ly/mgLyR
"""
from sympy import *
from sympy.plotting import plot3d

#x = symbols('x')
#f = 2*x + 1
#f = x**2 + 1
#plot(f)

#x, y = symbols('x, y')
#f = 2*x + 3*y
#plot3d(f)

#i, n = symbols('i n')
#sumation = Sum(2*i, (i,1,n))
#up_to_5 = sumation.subs(n, 5)
#print(up_to_5.doit())

#x = symbols('x')
#expr = x**2 / x**5
#print(expr)

#x = log(8, 2)
#print(x)

#x = symbols('x')
#f = 1 / x
#result = limit(f, x, oo)
#print(result)

#n = symbols('n')
#f = (1 + (1/n))**n
#result = limit(f, n, oo)
#print(result)   # E
#print(result.evalf())   # 2.7182845905

## Sample 1-18 ##
#x = symbols('x')
#f = x**2
#dx_f = diff(f)
#print(dx_f) # 2*x

## Sample 1-19 ##
"""
def f(x) :
    return x**2

def dx_f(x) :
    return 2*x

slope_at_2 = dx_f(2.0)
print(slope_at_2)
"""

## Sample 1-20 ## (需要搭配 Sample 1-18)
#print(dx_f.subs(x,2))

## Sample 1-21 ##
"""
x, y = symbols('x y')
f = 2*x**3 + 3*y**3
dx_f = diff(f, x)
dy_f = diff(f, y)

print(dx_f)
print(dy_f)

plot3d(f)
"""

## Sample 1-22 ##
x, s = symbols('x s')
f = x**2
slope_f = (f.subs(x, x+s) - f) / ((x+s) - x)
slope_2 = slope_f.subs(x, 2)
result = limit(slope_2, s, 0)
print(result) # 4



"""
subs(*args, **kwargs)
Substitutes old for new in an expression after sympifying args
符號替換
1.數值替換，以數值取代符號，進行帶入計算。
2.符號替換，用一些符號替換符號。
"""
## Sample 1-23 ##
x, s = symbols('x s')
f = x**2
slope_f = (f.subs(x, x+s) - f) / ((x+s) - x)
result = limit(slope_f, s, 0)
print(result) # 2x

## Sample 1-24 ##
z = (x**2 + 1)**3 - 2
dz_dx = diff(z, x)
print(dz_dx) # 6*x*(x**2 + 1)**2

## Sample 1-25 ##
x, y = symbols('x y')
_y = x**2 + 1
dy_dx = diff(_y)

z = y**3 - 2
dz_dy = diff(z)

dz_dx_chain = (dy_dx *dz_dy).subs(y, _y)
dz_dx_no_chain = diff(z.subs(y, _y))

print(dz_dx_chain)    # 6*x*(x**2 + 1)**2
print(dz_dx_no_chain) # 6*x*(x**2 + 1)**2

## 1-26 ##
def approximate_integral(a, b, n, f) :
    delta_x = (b - a) / n
    print("delta_x= ", delta_x)
    total_sum = 0

    for i in range(1, n+1) :
        midpoint = 0.5 * (2 * a + delta_x * (2 * i - 1))
        print("midpoint= ", midpoint)
        total_sum += f(midpoint)
        print("total_sum= ", total_sum)

    return total_sum * delta_x

def my_function(x) :
    print(x**2 + 1)
    return x**2 + 1

area = approximate_integral(a=0, b=1, n=5, f=my_function)
print(area) # 1.33

## Sample 1-29 ##
print("\n")
print("===================")
print("=== Sample 1-29 ===")
print("===================")
x = symbols('x')
f = x**2 + 1
area = integrate(f, (x, 0, 1))
print(area)


print("\n")
print("====================")
print("===    練習題    ===")
print("====================")

# (1) 10**7 * 10**(-5)

x = symbols('x y')
x = 10**7
y = 10**(-5)
f = x * y
print(f)

# (5)
p = 1000
r = 0.05
n = 12
t = 3
f = p * (1 + r / n)**(n*t)
print(f)


# (7) 微分
x, y = symbols('x y')
z = 3 * x**2 + 1
dx_dy = diff(z)
print(dx_dy.subs(x, 3))

# (8) 積分
# f(x) = 3*x**2 + 1, 在0和2之間的曲線下面積是多少？
x = symbols('x')
f = 3 * x**2 + 1
area = integrate(f, (x, 0, 2))
print(area)


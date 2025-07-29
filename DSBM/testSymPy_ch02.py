from sympy import *
from sympy.plotting import plot3d
from scipy.stats import binom
from scipy.stats import beta

#-- Sample 2-1 --#
p_coffee_drinker = .65
p_cancer = .005
p_coffee_drinker_given_cancer = .85

p_cancer_given_coffee_drinker = p_coffee_drinker_given_cancer * p_cancer / p_coffee_drinker
print(p_cancer_given_coffee_drinker)


#-- Sample 2-2 --#
n = 10
p = 0.9

for k in range(n+1) :
    probability = binom.pmf(k, n, p)
    print("{0} - {1}".format(k, probability))

#-- Sample 2-3 --#
a = 8
b = 2
p = beta.cdf(.90, a, b)
print(p)

#-- Sample 2-4 --#
a = 15
b = 4
p = 1 - beta.cdf(.50, a, b)
print(p)

#-- Practice 4 --#
n = 137
p = .40
p_50_or_more_noshows = 0.0

for x in range(50, 138) :
    p_50_or_more_noshows += binom.pmf(x, n, p)
print(p_50_or_more_noshows)


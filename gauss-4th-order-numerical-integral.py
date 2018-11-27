# -*- coding: utf-8 -*-
"""
author: christoph manucredo, chris.manucredo@gmail.com
about: 
    gauss4 numerically integrates a real function using the 4th 
    order gauss algortihm (quadrature). a test function "f" has
    been suplied.
"""

import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.sqrt((np.abs(0.5 - x))**3)


"""
this function is an implementation of the gauss-quadrature of 4th order.
it basically devides the desired intervall in smaller intervalls and 
culculates an approximation of the integral in those smaller intervalls
and sums them up.
"""
def gauss4(f,start,end,N):
    approx = 0
    delta = ((end-start)/N)
    I = np.arange(start,end,delta)
    for i in I:
        approx += delta * (1/2) * (f(i + (0.5 - np.sqrt(3)/6)*delta) + f(i + (0.5 + np.sqrt(3)/6)*delta))
    return approx 

# this is the "real" solution for the integral of f with bounds 0 and 1
realSolution = 0.1414212452341

#this is an array consisting of values we use for the work-accuracy plot
exponents = np.arange(1,20,1)
N = 2**exponents

#work-accuracy plot
plt.grid(True)
plt.loglog(N, [np.abs(gauss4(f,0,1,n) - realSolution) for n in N], N, 1/(N**(3)))
plt.legend(["Approximation"," N^(-3)"])

"""
we can see, that the gauss algortihm achieves an accuracy of approx. 10^-7 with less
than 100 intervalls.
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numba import jit

@jit
def tdma(a, b, c, d):
    n=len(d)
    w = np.zeros(n-1,float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)
    
    #perform first assignment
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]
    
    #perform forward propogation
    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    
    #assign last pressure
    p[n-1] = g[n-1]
    
    #begin backpropogation step
    for i in range(n-1,0, -1):
        p[i-1]=g[i-1]-w[i-1]*p[i]
    return p



if __name__ == '__main__':
   sol =  tdma([5.,5.,40.018],[-60.014259,-11.56944,-533.2242,-423846.87], [20.,5.,10.045],[ -4.00427781e+04,  -4.70833886e+03,  -1.55465924e+06, -1.27136072e+09])
   print(sol)
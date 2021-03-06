import numpy as np
from numba import jit
# this function performs the Tridiagonal Matrix Algorithm (Thomas Algorithm)
# It uses
@jit
def tdma(a,b,c,d):
    #initialize arrays
    n = len(d)
    w = np.zeros(n-1, float)
    g = np.zeros(n, float)
    p = np.zeros(n, float)

    #perform first assignment
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    #perform forward propogation
    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1, n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    #assign last pressure
    p[n-1] = g[n-1]
    # begin backpropogation step
    for i in range(n-1,0,-1):
        p[i-1]=g[i-1] - w[i-1]*p[i]
    return p


if __name__ == '__main__':
    print(tdma([5.,5.,20.],[-60.025,-12.7901,-931.2795,-912659.413],[20.,5.,5.],[-4.0076e+4,-8.37038e+3,-2.7638e+6,-2.7379e+9]))


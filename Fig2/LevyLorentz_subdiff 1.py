import numpy as np
from numba import jit

@jit(nopython=True)
def S(x0,t,a):
    X=x0
    T=0
    v=1
    while T<t and X<x0+1:
        T+=1
        if np.random.random()>1/max(1,np.abs(X))**(1-a)/2:
            v*=-1
        X+=v
    return T

a=0.5
for x0 in [10,100,1000]:
    np.save('LevyLor_SubDiff/'+str(a)+'/'+str(x0)+'/'+str(np.random.randint(10**5)),[S(x0,1e8,a) for k in range(10**4)])

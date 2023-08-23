import numpy as np
from numba import jit

@jit(nopython=True)
def S(x0,t,a):
    X=x0-1
    T=1
    v=-1
    while T<t and X<x0+1:
        T+=1
        if np.random.random()<1/max(1,np.abs(X))**(1-a)/2:
            v*=-1
        X+=v
    return T

a=0.4
for x0 in [100,1000,10000]:
    np.save('LevyLor_SupDiff/'+str(a)+'/'+str(x0)+'/'+str(np.random.randint(10**5)),[S(x0,1e9,a) for k in range(10**3)])


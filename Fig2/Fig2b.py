import numpy as np
from numba import jit

@jit(nopython=True)
def f(h,dt=0.1):
    dxh=np.array([dt*(h[i-1]-2*h[i]+h[(i+1)%len(h)])+np.sqrt(dt)*np.random.normal() for i in range(len(h))])
    return dxh

@jit(nopython=True)
def X(n,N=200,tmax=10**5,dt=0.1):
    h=np.zeros(N)
    M=[0. for k in range(N)]
    Nl=[0 for k in range(N)]
    tau=[0 for k in range(N)]
    T=[0 for k in range(N)]
    while min(Nl)<(n+1) and min(T)<2*tmax and min(tau)<2*tmax:
       h+=f(h,dt)
       for k in range(N):
           if Nl[k]<n:
               T[k]+=1
           if Nl[k]<n+1 and Nl[k]>=n:
               tau[k]+=1
           if h[k]>M[k]:
               Nl[k]+=1
           M[k]=max(h[k],M[k])
    for k in range(N):
        if T[k]>tmax:
            tau[k]=0
        if T[k]<=tmax and tau[k]==0:
            tau[k]=tmax
    return tau

for n in [5,10,20]:
  sr=str(np.random.randint(10**4))
  Xlis=[]
  for u in range(1):
      Xlis+=list(X(n,10000,10**6,0.1))
  np.save('fbmSemiInfH14/'+str(n)+'/'+sr,Xlis)


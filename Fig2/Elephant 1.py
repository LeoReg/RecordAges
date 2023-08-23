import numpy as np
from numba import jit

@jit(nopython=True)
def taun(n,beta,tmax):
    x=2*np.random.randint(0,2)-1
    steps=[0,0]
    steps[(x+1)//2]+=1
    M=max(x,0)
    tau=0
    T=1
    while M<n+1 and T<tmax and tau<tmax:
        nstep=max(n-x,1)
        Reverse=np.random.random(nstep)
        st=np.random.random(nstep)
        if M==n:
            tau+=nstep
        if M<n:
            T+=nstep
        for k in range(nstep):
            if st[k]<steps[0]/(steps[0]+steps[1]):
                if Reverse[k]<beta:
                  x+=-1
                  steps[0]+=1
                else:
                  x+=1
                  steps[1]+=1
            else:
                if Reverse[k]<beta:
                   x+=1
                   steps[1]+=1
                else:
                   x+=-1
                   steps[0]+=1
        M=max(M,x)
    if T==tmax:
        return 0
    return tau  

beta=0.6
for n in [10,25,50,100]:
    L=[taun(n,beta,10**8) for k in range(4000)]
    np.save('Elephant/'+str(beta)+'/'+str(n)+'/'+str(np.random.randint(10**4)),L)

              

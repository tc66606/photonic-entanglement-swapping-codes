"""Full symmetric depolarized-qubit source family (Appendix D).

The optical comparator is the full <=3-mixing class with one independent
helper. F4 denotes the selected midpoint-helper U4 event.
"""
import numpy as np

def coherence(p,c):
    p,c=np.broadcast_arrays(np.asarray(p,dtype=float),np.asarray(c,dtype=float))
    if np.any((p<0)|(p>1)|(c<0)|(c>1)):raise ValueError('p,c must be in [0,1]')
    k=2*c*c-1;t=(1+p*p*k)/2
    value=(t+p*c)/(1+p*c)
    tilt=p*(1-2*c*c)>c
    denom=2*np.sqrt(np.where(tilt,1-c*c,1.))
    return np.where(tilt,np.sqrt(np.maximum(0,1-p*p*k*k))/denom,value)

def selected(p,c):
    p,c=np.asarray(p),np.asarray(c);t=(1+p*p*(2*c*c-1))/2;a=p*c
    D=2+2*t-a
    return D/72,(1+t+a)/D

def response_gap(p,c):
    return 2*selected(p,c)[1]-1-coherence(p,c)

def attaining_helper(p,c):
    if p==1 and c==0:return np.array([0.,0.,-1.])
    if p*(1-2*c*c)<=c:return np.array([0.,0.,-1.])
    z=c/(p*(2*c*c-1))
    return np.array([np.sqrt(max(0.,1-z*z)),0.,z])

def threshold(c):
    """Threshold equation root from Appendix D; endpoints are limiting values."""
    c=np.asarray(c,dtype=float);lo=np.zeros_like(c);hi=np.ones_like(c)
    for _ in range(60):
        mid=(lo+hi)/2;positive=response_gap(mid,c)>0
        hi=np.where(positive,mid,hi);lo=np.where(positive,lo,mid)
    return np.where((c==0)|(c==1),1.,(lo+hi)/2)

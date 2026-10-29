"""Normalized creation-operator evaluator for independent internal packets.
The retained reference has four components. Spatial PNR occupations are
formed by tracing internal labels, preserving the reference state."""
from collections import defaultdict
from itertools import product, combinations_with_replacement
from math import sqrt
import numpy as np


def bs(m,i,j,theta=np.pi/4):
    U=np.eye(m,dtype=complex)
    U[np.ix_([i,j],[i,j])]=[[np.cos(theta),np.sin(theta)],
                            [np.sin(theta),-np.cos(theta)]]
    return U


def four_unitary(offsets=(0.,0.,0.,0.)):
    a,b,g,d=offsets
    return (bs(5,0,1,np.pi/4+d)@bs(5,1,4,np.arccos(1/np.sqrt(3))+g)
            @bs(5,1,3,np.pi/4+b)@bs(5,0,2,np.pi/4+a))


def two_helper_unitary():
    return bs(6,1,5)@bs(6,0,4)@bs(6,1,3)@bs(6,0,2)


def three_unitary():
    return bs(5,0,4) @ bs(5,1,3) @ bs(5,0,2)


def occupations(m, n):
    """Order induced by lexicographic nondecreasing lists of occupied ports."""
    return np.array([np.bincount(slots,minlength=m)
                     for slots in combinations_with_replacement(range(m),n)],int)


def mixed_instrument(U, densities):
    """Spectral mixture of independent sources, retaining every spatial record."""
    ensembles=[]
    for rho in np.asarray(densities,complex):
        if not np.allclose(rho,rho.conj().T,atol=1e-12) or abs(np.trace(rho)-1)>1e-12:
            raise ValueError('Source density operators must be Hermitian with trace one.')
        weights,vectors=np.linalg.eigh(rho)
        if weights.min() < -1e-12:
            raise ValueError('Source density operators must be positive.')
        ensembles.append([(float(w),vectors[:,i]) for i,w in enumerate(weights) if w>1e-14])
    records=occupations(len(U),len(densities))
    index={tuple(y):i for i,y in enumerate(records)}
    raw=np.zeros((len(records),4,4),complex)
    for terms in product(*ensembles):
        weight=np.prod([w for w,v in terms])
        for y,J in instrument(U,np.array([v for w,v in terms])).items():
            raw[index[y]] += weight*J
    return records,raw


def instrument(U,V):
    V=np.asarray(V,complex)
    m,d=len(U),V.shape[1]
    assert np.max(abs(U.conj().T@U-np.eye(m)))<1e-11
    assert np.max(abs(np.sum(abs(V)**2,axis=1)-1))<1e-11
    kets={}
    for a,b in product(range(2),repeat=2):
        terms={(0,)*(m*d):1.+0j}
        for source,imode in enumerate([a,2+b]+list(range(4,4+len(V)-2))):
            transitions=[(mode*d+label,U[mode,imode]*V[source,label])
                         for mode in range(m) for label in range(d)
                         if abs(U[mode,imode]*V[source,label])>1e-15]
            nxt=defaultdict(complex)
            for occ,amp in terms.items():
                for pos,val in transitions:
                    out=list(occ);out[pos]+=1
                    nxt[tuple(out)]+=amp*val*sqrt(occ[pos]+1)
            terms=nxt
        for occ,amp in terms.items():
            if occ not in kets:kets[occ]=np.zeros(4,complex)
            kets[occ][2*a+b]=amp
    raw={}
    for occ,ket in kets.items():
        record=tuple(sum(occ[i*d:(i+1)*d]) for i in range(m))
        raw[record]=raw.get(record,np.zeros((4,4),complex))+np.outer(ket,ket.conj())/4
    assert np.max(abs(sum(raw.values())-np.eye(4)/4))<1e-10
    return raw


def local_bell_weight(J):
    """Magic basis: real unit vector parameterizes all maximally entangled kets."""
    Q=np.array([[1,1j,0,0],[0,0,1j,1],[0,0,1j,-1],[1,-1j,0,0]],complex)/sqrt(2)
    return float(np.linalg.eigvalsh((Q.conj().T@J@Q).real)[-1])

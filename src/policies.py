"""Complete fixed-instrument acceptance under local-unitary recovery."""
import numpy as np


def bell_recoveries(raw):
    """Return Bell weights, targets and W=sqrt(2) Psi^dagger in row-major order.

    For a degenerate maximum choose the first nonzero column of its spectral
    projector, then make its first nonzero entry positive. A null record uses
    Phi+ and the identity recovery.
    """
    Q=np.array([[1,1j,0,0],[0,0,1j,1],[0,0,1j,-1],[1,-1j,0,0]],complex)/np.sqrt(2)
    ns=[];targets=[];recoveries=[]
    for J in raw:
        vals,vecs=np.linalg.eigh((Q.conj().T@J@Q).real)
        top=vecs[:,np.abs(vals-vals[-1])<1e-13]
        projector=top@top.T
        v=next(projector[:,i] for i in range(4) if np.linalg.norm(projector[:,i])>1e-10)
        v=v/np.linalg.norm(v)
        if v[np.flatnonzero(abs(v)>1e-12)[0]]<0:v=-v
        ket=Q@v;W=np.sqrt(2)*ket.reshape(2,2).conj().T
        ns.append(max(0.,float(vals[-1])));targets.append(ket);recoveries.append(W)
    return np.array(ns),np.array(targets),np.array(recoveries)


def acceptance_policy(p,n,target):
    """Complete average-fidelity policy, including sub-target records if feasible."""
    p=np.asarray(p,float);n=np.asarray(n,float);accept=np.zeros(len(p))
    active=np.flatnonzero(p>1e-14)
    if not len(active):return accept
    order=active[np.argsort(-n[active]/p[active],kind='stable')]
    if target>n[order[0]]/p[order[0]]+2e-13:return accept
    surplus=0.
    for i in order:
        delta=n[i]-target*p[i]
        if delta>=-1e-14 or surplus+delta>=-1e-14:
            accept[i]=1.;surplus+=delta
        else:
            accept[i]=max(0.,min(1.,surplus/(-delta)))
            break
    return accept

def probability_frontier(p,n,targets):
    p=np.asarray(p,float);n=np.asarray(n,float)
    mask=p>1e-14;p=p[mask];n=n[mask]
    order=np.argsort(-n/p,kind='stable');p=p[order];n=n[order]
    ratios=n/p;cp=np.cumsum(p);cn=np.cumsum(n)
    result=[]
    for target in targets:
        if target>ratios[0]+2e-13: result.append(0.);continue
        if target>=ratios[0]-2e-13:
            result.append(float(p[abs(ratios-ratios[0])<2e-12].sum()));continue
        if target<=cn[-1]/cp[-1]+1e-14:result.append(float(cp[-1]));continue
        k=int(np.flatnonzero(cn-target*cp < -1e-14)[0])
        pp=cp[k-1] if k else 0.;nn=cn[k-1] if k else 0.
        frac=(nn-target*pp)/(target*p[k]-n[k]);result.append(float(pp+frac*p[k]))
    return np.array(result),float(ratios[0]),np.r_[cn/cp,ratios]

def magic_n(raw):
    Q=np.array([[1,1j,0,0],[0,0,1j,1],[0,0,1j,-1],[1,-1j,0,0]],complex)/np.sqrt(2)
    return np.array([np.linalg.eigvalsh((Q.conj().T@J@Q).real)[-1] for J in raw])



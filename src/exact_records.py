"""Symbolic three-photon path contractions for the fixed U4 instrument.

For symmetric qubit sources and an axial helper, the six permutation
invariants are 1, tau, q/2, q/2, (tau+q-1)/2, (tau+q-1)/2.
"""
from itertools import permutations
from math import factorial
import json
import sympy as sp
from .fock import occupations


def build_exact_records(folder):
    tau,q=sp.symbols('tau q',real=True)
    def bs(i,j,c,s):
        U=sp.eye(5);U[i,i]=c;U[i,j]=s;U[j,i]=s;U[j,j]=-c
        return U
    h=sp.sqrt(2)/2
    U=bs(0,1,h,h)*bs(1,4,sp.sqrt(3)/3,sp.sqrt(6)/3)*bs(1,3,h,h)*bs(0,2,h,h)
    perms=list(permutations(range(3)))
    def contraction(pi,sigma):
        relative={pi[j]:sigma[j] for j in range(3)}
        changed=[i for i in range(3) if relative[i]!=i]
        if not changed:return sp.S.One
        if len(changed)==3:return (tau+q-1)/2
        return tau if set(changed)=={0,1} else q/2
    K=sp.Matrix([[contraction(pi,sigma) for sigma in perms] for pi in perms])
    Q=sp.Matrix([[1,sp.I,0,0],[0,0,sp.I,1],[0,0,sp.I,-1],[1,-sp.I,0,0]])/sp.sqrt(2)
    rows=[];groups={};total=sp.zeros(4)
    for record in occupations(5,3):
        slots=[i for i,n in enumerate(record) for _ in range(n)]
        Z=sp.zeros(4,6)
        for ab in range(4):
            inputs=[ab//2,2+ab%2,4]
            for k,pi in enumerate(perms):
                Z[ab,k]=sp.prod(U[slots[j],inputs[pi[j]]] for j in range(3))
        J=(Z*K*Z.T/(4*sp.prod(factorial(int(n)) for n in record))).applyfunc(sp.expand)
        M=sp.re(Q.conjugate().T*J*Q).applyfunc(sp.simplify)
        eig={sp.sstr(sp.simplify(k)):int(v) for k,v in M.eigenvals().items()}
        prob=sp.sstr(sp.simplify(sp.trace(J)))
        row={'record':record.tolist(),'P':prob,'magic_eigenvalues':eig,
             'J':[[sp.sstr(v) for v in J.row(i)] for i in range(4)]}
        rows.append(row);total+=J
        key=(prob,tuple(sorted(eig.items())))
        if key not in groups:groups[key]={'formulas':[prob,eig],'records':[],'multiplicity':0}
        groups[key]['records'].append(record.tolist());groups[key]['multiplicity']+=1
    if total.applyfunc(sp.simplify)!=sp.eye(4)/4:
        raise ArithmeticError('Symbolic conditional states do not form a complete instrument.')
    output={'variables':{'tau':'Tr(AB)=(1-p^2)/2+p^2*c^2',
                         'q':'Tr((A+B)H)=1+p*c*h_z; |h_z|<=1'},
            'method':'Exact permutation contraction from the specified four-mixing matrix.',
            'record_order':'Lexicographic nondecreasing output-slot lists.',
            'reference_basis':['00','01','10','11'], 'records':rows,
            'completeness':'sum_y J_y = I_4/4',
            'physical_parameter_domain':'0<=p,c<=1; H=(I+h_z Z)/2, |h_z|<=1'}
    (folder/'EXACT_RECORDS_SYMMETRIC_QUBIT.json').write_text(json.dumps(output,indent=2)+'\n')
    (folder/'RECORD_GROUPS_SYMMETRIC_QUBIT.json').write_text(json.dumps(list(groups.values()),indent=2)+'\n')
    return rows

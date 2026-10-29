"""Generate all datasets from source parameters and optical transformations.

All probabilities use the reference-pair
input |Phi+> |Phi+>; complete spatial records trace over internal labels.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.fock import four_unitary,three_unitary,two_helper_unitary,mixed_instrument
from src.models import source_densities,bare_probability,efficiency,gaussian_target,four_point
from src.policies import bell_recoveries,probability_frontier,acceptance_policy
from src.exact_records import build_exact_records


def write(folder,name,obj):
    (folder/name).write_text(json.dumps(obj,indent=2)+'\n')


def curve(p,n,start=.5,count=301):
    _,maximum,knots=probability_frontier(p,n,[])
    grid=np.unique(np.r_[np.linspace(start,maximum,count),
                         knots[(knots>=start)&(knots<=maximum)],maximum])
    values,_,_=probability_frontier(p,n,grid)
    return np.column_stack([grid,values]),maximum,knots


def main():
    start=time.perf_counter();folder=ROOT/'data';folder.mkdir(exist_ok=True)
    curves=[];policies=[];generated={}
    for p,c,tag in [(1.,.5,'p1_c0_5'),(.9,.5,'p0_9_c0_5'),(1.,.9,'p1_c0_9')]:
        U4,U3=four_unitary(),three_unitary()
        s4,s3=source_densities(p,c,'midpoint'),source_densities(p,c,'opposite')
        records,J=mixed_instrument(U4,s4);records3,J3=mixed_instrument(U3,s3)
        weights=np.trace(J,axis1=1,axis2=2).real;w3=np.trace(J3,axis1=1,axis2=2).real
        n,targets,W=bell_recoveries(J);n3,targets3,W3=bell_recoveries(J3)
        f4,max4,knots4=curve(weights,n);f3,max3,knots3=curve(w3,n3)
        metadata={'source_bloch_length':p,'pure_component_overlap':c,
                  'reference_basis':['00','01','10','11'],
                  'internal_basis':'midpoint basis: h+=(1,0), h-=(0,1)',
                  'helper_U4':'midpoint h+','helper_U3':'opposite h-',
                  'record_order':'Lexicographic nondecreasing output-slot lists, all 35 occupations.',
                  'probability':'per supplied trial, with two ideal reference-rail Bell pairs',
                  'recovery':'record-dependent local unitary W on first reference',
                  'degeneracy':'first nonzero column of the top real spectral projector; positive first nonzero component',
                  'optical_order':'rightmost matrix first; U3=B04 B13 B02, all balanced; U4=B01 B14 B13 B02'}
        np.savez(folder/f'POLICY_{tag}.npz',U=U4,source_states=s4,occupations=records,
                 J=J,p=weights,n=n,targets=targets,recoveries=W,
                 three_U=U3,three_sources=s3,three_raw_J=J3,three_p=w3,three_n=n3,
                 three_targets=targets3,three_recoveries=W3,
                 three_ceiling_acceptance=acceptance_policy(w3,n3,max3),
                 source_bloch_length=p,pure_component_overlap=c,metadata_json=json.dumps(metadata))
        grid=np.unique(np.r_[np.linspace(.5,1,101),knots4,knots3,.74,.75,.8,.875,max3,max4])
        grid=grid[(grid>=.5)&(grid<=1)]
        p4,_,_=probability_frontier(weights,n,grid);p3,_,_=probability_frontier(w3,n3,grid)
        entry=dict(metadata,p=p,c=c,tau=(1-p*p)/2+p*p*c*c,
                   three_ceiling=max3,three_attained_probability=float(f3[-1,1]),
                   four_ceiling=max4,four_endpoint_probability=float(f4[-1,1]),
                   policy_breakpoints={'U4':sorted(set(knots4.tolist())),
                                       'U3':sorted(set(knots3.tolist()))},
                   points=[{'target_F':float(t),'P_four':float(a),'P_three_attainer':float(b),
                            'P_bare':bare_probability((1-p*p)/2+p*p*c*c,t)}
                           for t,a,b in zip(grid,p4,p3)])
        curves.append(entry)
        for name,pw,nw in [('U4',weights,n),('U3',w3,n3)]:
            for F in [.74,.75,.8,.875]:
                acceptance=acceptance_policy(pw,nw,F)
                policies.append({'p':p,'c':c,'circuit':name,'target_F':F,
                                 'acceptance':acceptance.tolist(),
                                 'P':float(acceptance@pw),'N':float(acceptance@nw)})
        generated[tag]=(J,f4,f3)
        print(f'Built {tag}: U4 endpoint ({max4:.12g}, {f4[-1,1]:.12g}); '
              f'U3 h- endpoint ({max3:.12g}, {f3[-1,1]:.12g})',flush=True)
    write(folder,'FIXED_NETWORK_FRONTIERS.json',{'curves':curves,'policies':policies,
          'scope':'Complete acceptance policies of the specified fixed U4/h+ and U3/h- instruments; local-unitary recovery.'})

    c=.9;J,f4,f3=generated['p1_c0_9'];tau=c*c;boundary=(1+tau)/2
    ggrid=np.unique(np.r_[np.linspace(boundary,1,501),four_point(c)['F'],1.])
    common=[];independent=[]
    for F in ggrid:
        r=np.log(2*F-1)/(2*np.log(c))
        common.append([float(F),float(max(0.,r)*c**(2*(1-r))/2*efficiency(2,1))])
        val=.5 if F<=boundary+1e-14 else gaussian_target(c,F)['P']
        independent.append([float(F),float(val*efficiency(2,1))])
    densities=source_densities(1.,c,'opposite')
    rec2,J2=mixed_instrument(two_helper_unitary(),np.r_[densities,densities[2:3]])
    i2=np.flatnonzero(np.all(rec2==[1,1,0,0,1,1],axis=1))[0]
    n2,_,_=bell_recoveries(J2[i2:i2+1]);p2=float(np.trace(J2[i2]).real)
    np.savez(folder/'TWO_HELPER_p1_c0_9.npz',U=two_helper_unitary(),
             source_states=np.r_[densities,densities[2:3]],occupations=rec2,J=J2,
             selected_record=np.array([1,1,0,0,1,1]),
             metadata_json=json.dumps({'p':1.,'c':c,'helpers':'two independent h-',
                 'probability':'per supplied trial','reference_basis':['00','01','10','11']}))
    bare_grid=np.unique(np.r_[np.linspace(.5,boundary,151),.5+tau/4,boundary])
    fig4={'c':c,'probability_convention':'accepted probability per source attempt',
          'gaussian_domain':'Equal widths; common center or independently optimized centers; complete bare acceptance.',
          'efficiencies':{'sA':.8,'sB':.8,'sH_per_helper':.7,'path':.95,'detector':.9,'uniform_padded_layer':.98},
          'instrument_settings':{'U4':'four_unitary(), midpoint h+','U3':'three_unitary(), opposite h-'},
          'four_curve':(f4*[1,efficiency(3,3)]).tolist(),
          'three_attainer_curve':(f3*[1,efficiency(3,2)]).tolist(),
          'bare_curve':[[float(F),bare_probability(tau,F)*efficiency(2,1)] for F in bare_grid],
          'common_center_curve':common,'independent_center_curve':independent,
          'two_helper_point':{'F':float(n2[0]/p2),'P_attempt':p2*efficiency(4,2)},
          'rank_one_point':{'F':1.,'P_attempt':(1+c)**2/8*efficiency(2,1)},
          'source':'Model-generated instruments and analytical filters; Section 6 and Appendix F.'}
    write(folder,'FIGURE4_PLOT_INPUTS.json',fig4)
    print('Building exact record expressions from permutation contractions...',flush=True)
    rows=build_exact_records(folder)
    import sympy as sp
    symbols={'tau':sp.Rational(81,100),'q':sp.Rational(19,10)}
    exact=np.array([[[complex(sp.sympify(x).subs(symbols)) for x in row] for row in r['J']] for r in rows])
    if np.max(abs(exact-J))>1e-11:
        raise ArithmeticError('Exact table and creation-operator model disagree.')
    print(f'Wrote model-generated data to {folder}; elapsed {time.perf_counter()-start:.2f} s.',flush=True)


if __name__=='__main__':main()

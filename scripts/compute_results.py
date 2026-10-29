"""Evaluate representative fidelities, rates and thresholds from the paper."""
from pathlib import Path
import json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.models import four_point, gaussian_target, efficiency
from src.mixed_domain import coherence, selected, threshold, attaining_helper
from src.policies import probability_frontier, magic_n

def main():
    pure=[]
    for c in [.5,.9,.99]:
        point=four_point(c);epsilon=(1-c*c)/2
        pure.append({'c':c,'F_three_max':(1+c)/2,'F_four':point['F'],
                     'P_four_supplied':point['P'],'bare_error':epsilon,
                     'four_error_exact':1-point['F'],'quadratic_asymptote':epsilon**2/3})
    mixed=[]
    for p,c in [(.9,.3),(.9,.5),(.8,.7)]:
        prob,fidelity=selected(p,c)
        mixed.append({'p':p,'c':c,'F_three_max':float((1+coherence(p,c))/2),
                      'F_four':float(fidelity),'P_four_supplied':float(prob),
                      'three_attaining_helper_bloch':attaining_helper(p,c).tolist()})
    policies=[]
    for p,name in [(1.,'POLICY_p1_c0_5.npz'),(.9,'POLICY_p0_9_c0_5.npz')]:
        d=np.load(ROOT/'data'/name)
        for circuit,raw in [('U4',d['J']),('three_mixing_instrument',d['three_raw_J'])]:
            weights=np.trace(raw,axis1=1,axis2=2).real;bell=magic_n(raw)
            vals,fmax,_=probability_frontier(weights,bell,[.8])
            end,_,_=probability_frontier(weights,bell,[fmax])
            policies.append({'p':p,'c':.5,'circuit':circuit,'F_max':fmax,
                             'P_endpoint_supplied':float(end[0]),'target_F':.8,
                             'P_at_target_supplied':float(vals[0])})
    physical=[]
    for c in [.5,.9]:
        four=four_point(c)
        physical.append({'c':c,'four_target_F':four['F'],
            'four_P_attempt':four['P']*efficiency(3,3),
            'independent_center_Gaussian_P_attempt_at_four_target':gaussian_target(c,four['F'])['P']*efficiency(2,1),
            'independent_center_Gaussian_P_attempt_at_F1':gaussian_target(c,1.)['P']*efficiency(2,1),
            'two_helper_F':1.,'two_helper_P_attempt':(1+c)**2/128*efficiency(4,2),
            'rank_one_F':1.,'rank_one_P_attempt':(1+c)**2/8*efficiency(2,1)})
    out={'probability_conventions':{'supplied':'Required interface photons have been delivered.',
         'attempt':'Includes the source, path, detector and padded-layer factors in Section 6.'},
         'pure_source_selected_event':pure,'mixed_source_selected_event':mixed,
         'mixed_source_boundary':[{'c':c,'p_crit':float(threshold(c))} for c in [.1,.3,.5,.7,.9]],
         'fixed_instrument_policies':policies,'physical_comparisons':physical,
         'efficiencies':{'sA':.8,'sB':.8,'sH_per_helper':.7,'path':.95,'detector':.9,'uniform_padded_layer':.98},
         'calibration_floor_for_two_3mrad_offsets':(2+4/3)*.003**2,
         'formula_sources':['Sections 3-6','Appendices D-F']}
    folder=ROOT/'results';folder.mkdir(exist_ok=True)
    (folder/'numerical_results.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print('Numerical results written to',folder/'numerical_results.json')

if __name__=='__main__':main()

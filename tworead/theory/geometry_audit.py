#!/usr/bin/env python3
"""Exact finite checks of the geometric steps of Lemmas 3-5 (Section 2.2; audited in Section 3).

Fixed-center clipping, closed windows on half-open bins, irredundant
cross-scale arcs and the residual-cover scope. No continuum certificate.
"""
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import random


def main(output):
    checks=[]

    def record(name, passed, **details):
        checks.append({'name':name,'passed':bool(passed),'arithmetic':'exact_rational_or_integer',**details})
        assert passed,name

    # Clipping inequality independently exhausted on incidence-admissible rational
    # parameters, including clipped kappa, both signs and exact tangent endpoints.
    cases=0; clipped=0
    for denominator in range(2,25):
        for numerator in range(denominator):
            delta=F(numerator,denominator)
            for s in [1-delta,1/(1-delta)]:
                for kappa in {F(j,31)/s for j in range(-31,32)}|{F(-1),F(0),F(1)}:
                    if abs(s*kappa)>1: continue
                    c=max(F(-1),min(F(1),kappa))
                    assert abs(s*kappa-c)<=delta
                    cases+=1; clipped+=int(abs(kappa)>1)
    record('fixed_center_clipping_both_normalization_branches',True,cases=cases,clipped_cases=clipped)

    # Work on a circle of circumference 1; exact closed arcs vs half-open bins.
    # Count atom indices, with duplicate centers deliberately allowed.
    rng=random.Random(20261009); max_bin=0; max_charge=F(0); bin_cases=0
    for J in range(2,41):
        ell=F(1,J)
        for radius in [ell/2,ell*F(499,1000)]:
            for trial in range(25):
                S=2+trial%9
                centers=[(F(rng.randrange(2*J),2*J),F(rng.randrange(2*J),2*J)) for _ in range(S)]
                if trial%2==0: centers[-1]=centers[0]
                atom_hits=[]
                for pair in centers:
                    individual=[]
                    for c in pair:
                        hits={j for j in range(J) if any(c-radius+d<F(j+1,J) and c+radius+d>=F(j,J) for d in [-1,0,1])}
                        max_bin=max(max_bin,len(hits)); individual.append(hits)
                    atom_hits.append(individual[0]|individual[1])
                charge=sum(len(hits) for hits in atom_hits)
                assert charge<=4*S
                max_charge=max(max_charge,F(charge,S)); bin_cases+=1
    record('closed_windows_half_open_bins_index_charge',max_bin<=2,cases=bin_cases,
           max_bins_per_center=max_bin,max_total_charge_per_atom=str(max_charge))

    # Exhaust every family of 1--4 arcs on an eight-point circle with lengths
    # 1,2,3 < 8/2. Endpoints and every open arrangement cell are witnesses.
    # Bit masks encode exact sets (not a sampled angular grid).
    arcs=[(a,l) for a in range(8) for l in range(1,4)]
    masks=[sum(1<<x for x in range(16) if (x-2*a)%16<=2*l) for a,l in arcs]
    families=0; max_multiplicity=0; measure_cases=0
    for count in range(1,5):
        for chosen in combinations(range(len(arcs)),count):
            keep=list(chosen); original=0
            for i in chosen: original|=masks[i]
            while True:
                for i in keep:
                    other=0
                    for j in keep:
                        if i!=j: other|=masks[j]
                    if masks[i]&~other==0:
                        keep.remove(i); break
                else: break
            retained=0
            for i in keep: retained|=masks[i]
            assert retained==original
            mult=max(sum(bool(masks[i]&(1<<x)) for i in keep) for x in range(16))
            assert mult<=2
            max_multiplicity=max(max_multiplicity,mult); families+=1
            # Counting measures include arbitrary repeated center entries.
            masses=[1+x%3 for x in range(16)]
            total=sum(masses)
            charge=sum(sum(m for x,m in enumerate(masses) if masks[i]&(1<<x)) for i in keep)
            assert charge<=2*total; measure_cases+=1
    record('irredundant_cross_scale_union_and_repeated_center_measure',True,
           exhaustive_families=families,counting_measure_cases=measure_cases,
           max_multiplicity=max_multiplicity)

    # Scope attack: residual containment is insufficient for equality in l1.
    n=[F(1,2),F(3,5),F(4,5)]; w=[F(1),F(-144,145),F(17,145)]; e=F(1,450)
    residual=abs(sum(a*b for a,b in zip(n,w)))
    record('residual_union_is_only_an_upper_cover',residual<=e<residual/max(map(abs,n)),
           residual=str(residual),exact_l1_plane_distance=str(residual/max(map(abs,n))))

    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'status':'exact_finite_attacks',
         'formal_verification':False,'checks':checks,'failures':[]},indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)

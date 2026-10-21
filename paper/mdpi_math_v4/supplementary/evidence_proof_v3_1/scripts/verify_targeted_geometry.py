"""Targeted v3 review: exact finite geometry and labelled numerical diagnostics.
This does not constitute formal verification of the continuous lemmas.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,math,random
import mpmath as mp
mp.mp.dps=80
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manuscript',type=Path,default=Path(__file__).resolve().parents[3]/'manuscript.tex')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'targeted_geometry.json')
args=parser.parse_args()
rng=random.Random(20261008)
checks=[]
def record(name,ok,kind,**data):
 checks.append(dict(name=name,passed=bool(ok),kind=kind,**data))
 assert ok,name

# Explicit witnesses to wording defects in the current text.
h=1+mp.log(2)
e1=1/(2000*4*h);e2=1/(160*4*h*h)
record('max_tolerance_not_first_tolerance_S2', e2>e1, 'high_precision_counterexample', ratio_e2_e1=float(e2/e1))
# Rational normalized normal n=(1/2,3/5,4/5), and exact unit-circle point.
# cos(theta)=-144/145, sin(theta)=17/145; n.w=-3/1450.
n=[F(1,2),F(3,5),F(4,5)];w=[F(1),F(-144,145),F(17,145)];e=F(1,450)
res=abs(sum(a*b for a,b in zip(n,w)));dist=res/max(map(abs,n))
record('U_is_subset_not_equal_to_residual_union',res<=e and dist>e,'exact_rational_counterexample',normal=list(map(str,n)),point=list(map(str,w)),tolerance=str(e),residual=str(res),l1_distance=str(dist))
# The geometric cap does not imply heaviness for arbitrary tau.
alpha=lambda y:2*mp.asin(mp.sqrt(y/2))
rho=alpha(mp.mpf('.05'))+alpha(mp.mpf('.04'))
J=int(mp.floor(mp.pi/rho));ell=2*mp.pi/J
Gamma=2*(alpha(mp.mpf('.03'))-alpha(mp.mpf('.01')))
record('active_cap_does_not_imply_heavy', 8*Gamma>=ell and not 4>100*(ell+2*rho) and ell+2*rho<mp.pi,'high_precision_counterexample',e=.01,beta0=.02,beta1=.04,J=J,m=4,tau=100)

# Exhaust closed-arc/bin endpoint and wrap-around cases with exact fractions.
max_bin_count=0;bin_cases=0
for J in range(1,65):
 ell=F(1,J)
 for rad in [ell/2, (F(1,2*(J+1))+ell/2)/2]:
  assert int(F(1,2)/rad)==J
  for c in [F(k,2*J) for k in range(2*J)]+[F(1,100003),F(1)-F(1,100003)]:
   count=sum(any(c-rad+shift<F(j+1,J) and c+rad+shift>=F(j,J) for shift in [-1,0,1]) for j in range(J))
   max_bin_count=max(max_bin_count,count);bin_cases+=1
record('closed_arc_hits_at_most_two_half_open_bins',max_bin_count<=2,'exact_rational_finite_cases',cases=bin_cases,max_count=max_bin_count)

# Exact minimal covers: endpoint and interval-arrangement witnesses, no grid.
max_mult=0;families=3000;removed_total=0
for trial in range(families):
 arcs=[(rng.randrange(2000),rng.randrange(1,1000)) for _ in range(rng.randrange(2,31))]
 if trial%3==0: arcs.append(arcs[0])
 if trial%5==0: arcs.extend([(1900,200),(0,100),(1950,150)])
 pts=sorted({a%2000 for a,l in arcs}|{(a+l)%2000 for a,l in arcs})
 witnesses=[2*x for x in pts]+[(a+(pts[(i+1)%len(pts)]+(2000 if i+1==len(pts) else 0)))%4000 for i,a in enumerate(pts)]
 masks=[]
 for a,l in arcs:
  masks.append(sum((1<<i) for i,t in enumerate(witnesses) if (t-2*a)%4000<=2*l))
 keep=list(range(len(arcs)));rng.shuffle(keep)
 while True:
  for j in keep:
   other=0
   for i in keep:
    if i!=j: other|=masks[i]
   if masks[j]&~other==0:
    keep.remove(j);removed_total+=1;break
  else: break
 mult=max(sum(bool(masks[j]&(1<<i)) for j in keep) for i in range(len(witnesses)))
 max_mult=max(max_mult,mult)
 assert mult<=2
record('minimal_arc_subcover_multiplicity',max_mult<=2,'exact_integer_finite_cases',families=families,max_multiplicity=max_mult,removed_arcs=removed_total)

# Exact rank-one containment with zero, repeated, signed and parallel rows.
def cross(a,b):
 return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def nz(a):return any(a)
rank_cases=0
special=[[(0,0,0)]*5,[(1,2,3),(-1,-2,-3),(2,4,6),(0,0,0)],[(1,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,0,0)]]
for trial in range(500):
 rows=[tuple(rng.randrange(-2,3) for _ in range(3)) for j in range(5)]
 rows.extend([(0,0,0),rows[0],tuple(-v for v in rows[0])]);special.append(rows)
for rows in special:
 independent=any(nz(cross(a,b)) for a in rows for b in rows)
 if not independent:continue
 for i,a in enumerate(rows):
  for b in rows[i+1:]:
   if nz(cross(a,b)) or not (nz(a) or nz(b)):continue
   u=a if nz(a) else b
   assert any(nz(cross(u,v)) for v in rows)
   rank_cases+=1
record('rank_one_spans_embed_in_rank_two_dictionary_spans',True,'exact_integer_finite_cases',dictionaries=len(special),rank_one_pairs=rank_cases,all_zero_and_all_parallel='separate common-plane branch')

# Localization, including both normal branches, signs, clipping, r_u=0,
# repeated contact centers, tangency and vanishing parameters. Numerical only.
max_ratio=mp.mpf(0);loc_cases=0;clipped=0
for delta in [mp.mpf(0),mp.mpf('1e-30'),mp.mpf('1e-12'),mp.mpf('.01'),mp.mpf('.5'),mp.mpf('.99')]:
 for e in [mp.mpf('1e-30'),mp.mpf('1e-12'),mp.mpf('.001'),mp.mpf('.1'),mp.mpf(1)]:
  for branch in range(2):
   a,r=(1-delta,mp.mpf(1)) if branch==0 else (mp.mpf(1),1-delta)
   s=a/r
   for sign in [-1,1]:
    for frac in [-1,-.999999,-.5,0,.5,.999999,1]:
     # Set phi=0. Incidence forces cos(psi)=-a*kappa/r.
     kappa=mp.mpf(str(frac))/s
     A=sign*a
     psi=mp.acos(-A*kappa/r)
     center=mp.acos(max(mp.mpf(-1),min(mp.mpf(1),kappa)))
     Cu=[psi-center,psi+center]
     lo=max(mp.mpf(-1),(-A-e)/r);hi=min(mp.mpf(1),(-A+e)/r)
     if lo>hi:continue
     bound=alpha(delta+e)+alpha(delta)
     for z in [lo,hi,(lo+hi)/2]:
      t=mp.acos(z)
      for theta in [t,-t]:
       actual=min(abs(mp.atan2(mp.sin(theta-c),mp.cos(theta-c))) for c in Cu)
       max_ratio=max(max_ratio,actual/bound)
       assert actual<=bound+mp.mpf('1e-35')
       loc_cases+=1
     if abs(kappa)>1:clipped+=1
record('fixed_center_localization_targeted',max_ratio<=1+mp.mpf('1e-20'),'80_digit_numerical_diagnostics',cases=loc_cases,clipped_atom_cases=clipped,max_ratio=float(max_ratio),r_u_zero_case='analytic: n.u=0 forces a=0, r=1, Delta=1')

tex=args.manuscript
result={'scope':'v3.1 finite geometry and correction witnesses','manuscript_sha256':hashlib.sha256(tex.read_bytes()).hexdigest(),'checks':checks,'all_checks_passed':all(x['passed'] for x in checks),'limit':'Finite exact cases and numerical diagnostics do not prove continuous lemmas.'}
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

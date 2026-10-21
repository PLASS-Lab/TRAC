# V3.4 internal proof audit — 2026-10-09

Scope: the current normalized-circle proof, fixed contact centers,
two-atom localization, summation over angular bins, and the single heavy
cover across scales. This is a fresh definition-level internal review and
an additional implementation, not independent specialist review or a
formal proof assistant certificate. The original theorem constants and
numbered equations are preserved.

## Fixed contact centers: all incidence branches

Write r_u=sqrt(u_1^2+u_2^2), kappa=u_0/r_u and
C_u={psi +/- arccos(clip(kappa,-1,1))}. Centers depend on the atom only.
If r_u=0 and u is nonzero, n.u=0 forces a=0; the normalized normal
then has r=1 and Delta=1. Such atoms cannot enter bands below beta_L<1.
Zero atoms are excluded from the center definition and handled as spans.

For r_u>0 and Delta<1, both |a| and r are positive. The signed contact
angle gives cos(theta_0-psi)=s*kappa with s=|a|/r and |s*kappa|<=1.
When r=1, s=1-Delta. For |kappa|<=1, the clipping difference is
|kappa|Delta<=Delta. For |kappa|>1 it is 1-s|kappa|<=1-s=Delta.
When |a|=1, s=1/r>=1 and |kappa|<=1/s; clipping is inactive and
|kappa|(s-1)<=1-1/s=Delta. The equality |a|=r=1 satisfies both.
The arccos modulus and triangle inequality then give rho(Delta,e).
The argument is per incident atom, so BOTH endpoints of a qualifying
pair are localized. Rescaling or negating an atom leaves its center set
unchanged modulo swapping the two entries.

Endpoint clarification: the integral defining the arccos window at
q=1 is improper but finite. The maximizing-window argument is applied
away from +/-1 and extended by continuity. No derivative is evaluated
at a singular endpoint. The lemma includes nu=0 and nu>=2q.

## Per-bin count: union cap, ranks and endpoint conventions

The residual union V contains U because ||n||_infinity<=1; equality is
unnecessary and in general false. If all nonzero atoms are parallel,
one plane covers all nonzero spans. Otherwise every rank-one span,
including zero or signed/repeated partners, lies in some rank-two
pair span. The all-zero dictionary gives an empty approximating set.

J=floor(pi/rho_bar) gives ell=2pi/J>=2rho_bar. A closed center window
of length <=ell meets at most two HALF-OPEN bins, also at equality
and across angle zero. Each atom supplies at most two windows, so
sum_I m_I<=4S. m_I counts atom indices, not distinct centers; coincident
centers and duplicate atoms do not invalidate this charge.

Both endpoints of a qualifying pair are counted. Its total residual
length is <=Gamma, hence the union inside a bin has measure
<=min(ell, binom(m_I,2)Gamma)<=min(ell,m_I^2 Gamma/2)
<=m_I sqrt(ell Gamma/2). This is a union estimate: summing pair
multiplicities without the length cap would change the argument.

## Cross-scale heavy cover: one measure for every band

A_I is the CLOSED rho_bar-neighborhood of the CLOSURE of bin I.
This only enlarges the half-open bin neighborhood and has length
ell+2rho_bar<pi. The same counting measure mu of contact-center entries
is used in every band, with coincident entries retained. mu(A_I)>=m_I
and mu(circle)<=2S. Heavy means m_I>tau|A_I|, not that the geometric
length cap is active.

The heavy family is finite. Delete an arc contained in the union of
the others; this preserves the union and terminates. If three retained
arcs contain x, cut the circle at the antipode of x. Because every arc
has length <pi, they all become intervals about the same representative
of x. The intervals with extreme left and right endpoints contain the
third; if the extremes belong to one interval it contains both others.
Either case contradicts irredundance. Closed endpoints are included.

The retained multiplicity is <=2 POINTWISE, including atomic center
locations. Consequently sum mu(A)<=2mu(circle), not merely an
almost-everywhere Lebesgue assertion. Thus sum |A|<=4S/tau. Light
bins cost <=2S tau(ell+2rho_bar)Gamma per band. Summing light bands
and charging the heavy union once gives Psi. The empty heavy family
has zero cost. All beta/rho/J and short-arc hypotheses are verified
separately in the exact constant program for each small/large regime.

## Additional finite attacks and remaining limits

`scripts/audit_math_v3_4.py` independently exhausts rational clipping
branches, closed-window/half-open-bin cases, and all families of up to
four short arcs on an eight-point circle. For the last check, endpoints
and all open arrangement cells represent the exact set arrangement;
this is not a sampled-angle approximation. It checks preservation of
the union and multiplicity <=2, including repeated center measures.
The retained 44 constant checks and seven targeted geometry checks
are rerun against V3.4. JSON ties the audit to reviewed source hashes.

No mathematical counterexample was found. Two endpoint conventions
are made explicit in the text; no lower-bound constant or theorem
statement changes. Finite exhaustion complements the general proofs
above but does not establish the continuum quantifiers. External
specialist review, formal verification and removal of the logarithmic
gap remain outstanding.

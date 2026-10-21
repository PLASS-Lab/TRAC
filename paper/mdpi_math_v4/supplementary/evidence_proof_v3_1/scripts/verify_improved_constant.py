#!/usr/bin/env python3
"""Exact certificate for the improved multiscale constants.

PRIMARY (Part D, single log, on top of lem:crossscale): circle 2000, curve 12000.
VARIANT (Parts A/B, squared log): circle 160, curve 960.

Part A (proof inputs, exact): rational interval arithmetic (Fractions with
outward-rounded sqrt/log/atan/pi) checks every numerical inequality used in
Proposition prop:multiscale-circle (new version) and Theorem thm:main:
  * arc length Lambda = 2 arctan(3/4) and transfer factor 2(1+x) <= 6;
  * crude regime 2 <= S <= 38 (each S, and the monotone endpoint form);
  * analytic regime S >= 39 (slope and endpoint conditions, side conditions).
Part B (cross-check, exact): the general band bound Phi_S of Lemma lem:multiscale
evaluated with the actual L(S) for S=39..400 and a few huge S.
Part C (float, EVIDENCE ONLY): random tests of the plane lemmas, snapshot
upper constructions on the circle arc and on the four-key curve, crossovers.
Writes work_dirs/upgrade_20261006/constant/verify.json; exits 1 on failure.
"""
import argparse
import json
import math
import sys
from fractions import Fraction as Q
from math import isqrt
from pathlib import Path

import numpy as np
import mpmath as mp
import sympy as sp

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'constant_verification.json')
OUT = parser.parse_args().output
checks, failures = [], []


def check(name, ok, **info):
    checks.append({'name': name, 'pass': bool(ok),
                   **{k: (float(v) if isinstance(v, Q) else v) for k, v in info.items()}})
    if not ok:
        failures.append(name)


# ---------- outward-rounded rational intervals: (lo, hi) ----------
def iv(x):
    x = Q(x)
    return (x, x)


def add(a, b): return (a[0]+b[0], a[1]+b[1])
def sub(a, b): return (a[0]-b[1], a[1]-b[0])


def mul(a, b):
    p = [a[i]*b[j] for i in (0, 1) for j in (0, 1)]
    return (min(p), max(p))


def div(a, b):
    assert b[0] > 0 or b[1] < 0, 'division by interval containing 0'
    return mul(a, (1/b[1], 1/b[0]))


def sqrt(a):
    assert a[0] >= 0

    def scale(v):                          # 2^t with v*4^t >= 2^300 (relative precision)
        if v == 0:
            return 1
        t = max(0, (300-(v.numerator.bit_length()-v.denominator.bit_length()))//2+1)
        return 2**t
    d0, d1 = scale(a[0]), scale(a[1])
    lo = Q(isqrt(math.floor(a[0]*d0*d0)), d0)
    hi = Q(isqrt(math.ceil(a[1]*d1*d1))+1, d1)
    assert lo*lo <= a[0] and hi*hi >= a[1]
    return (lo, hi)


def atanh_series(z, n=60):
    """2*atanh(z) for rational |z|<=1/2, enclosed."""
    s = sum(z**(2*j+1)/(2*j+1) for j in range(n))
    tail = abs(z)**(2*n+1)/((2*n+1)*(1-z*z))
    return (2*(s-tail), 2*(s+tail))


LOG2 = atanh_series(Q(1, 3))


def log(x):
    x = Q(x)
    assert x > 0
    k = 0
    while x > Q(3, 2):
        x /= 2
        k += 1
    while x < Q(3, 4):
        x *= 2
        k -= 1
    return add(mul(iv(k), LOG2), atanh_series((x-1)/(x+1)))


def atan(y, n=80):
    """arctan(y), 0<y<=1 rational: alternating series with decreasing terms."""
    y = Q(y)
    s = sum((-1)**j*y**(2*j+1)/(2*j+1) for j in range(n))
    t = y**(2*n+1)/(2*n+1)
    return (s-t, s+t)


PI = sub(mul(iv(16), atan(Q(1, 5))), mul(iv(4), atan(Q(1, 239))))
LAM = mul(iv(2), atan(Q(3, 4)))           # |Theta_0| = 2 arctan 2 - 2 arctan(1/2)
assert Q(314159, 10**5) < PI[0] and PI[1] < Q(314160, 10**5)
SQ2 = sqrt(iv(2))


def h(S):                                   # 1 + log S
    return add(iv(1), log(S))


def F_up(y, ymax):
    """upper bound of arccos(1-y) <= sqrt(2y)/sqrt(1-y/2) for 0<=y<=ymax<2."""
    return div(sqrt(mul(iv(2), y)), sqrt(sub(iv(1), div(ymax, iv(2)))))


def lt(a, b): return a[1] < b[0]
def le(a, b): return a[1] <= b[0]


# ---------------- Part A: proof inputs ----------------
K = Q(160)
SQK = sqrt(iv(K))
CURVE_FACTOR = 6
S_CRUDE = 38
PAR = dict(mu=Q(3), q=Q(18), sigma=Q(1, 200), m=1)   # analytic regime parameters
S1 = S_CRUDE+1

# transfer identities (symbolic)
x = sp.symbols('x', positive=True)
D = 1+x+x**2+x**3
Ta = sp.Matrix([1+x**2, 1-x**2, 2*x])/D
w = sp.Matrix([(1-x**2)/(1+x**2), 2*x/(1+x**2)])   # cos, sin of 2 arctan x
check('transfer_identity_Ta=w/(1+x)',
      sp.simplify(Ta - sp.Matrix([1, w[0], w[1]])/(1+x)) == sp.zeros(3, 1))
check('T_l1_norm_is_2', max(sum(abs(c) for c in col) for col in
                             ([1, 1, 0], [0, 0, 2], [1, -1, 0], [0, 0, 0])) == 2)
check('arc_identity_2atan2-2atan(1/2)=2atan(3/4)',
      sp.simplify(sp.tan(sp.atan(2)-sp.atan(sp.Rational(1, 2))) - sp.Rational(3, 4)) == 0)
check('transfer_factor_2(1+x)<=6_on_[1/2,2]', 2*(1+2) == CURVE_FACTOR)
check('Lambda_lower_bound', LAM[0] > Q(1287, 1000), Lambda_lo=LAM[0])

# crude regime: binom(S,2)*2F(2e) <= 2(S-1)/(h sqrt(K) sqrt(1-e)) < Lambda, e=e_S
crude_rows = []
for S in range(2, S_CRUDE+1):
    hS = h(S)
    e = div(iv(1), mul(iv(K*S*S), mul(hS, hS)))
    # F(2e) <= sqrt(4e)/sqrt(1-e); binom(S,2)*2*F(2e)
    cov = mul(iv(Q(S*(S-1), 2)*2), F_up(mul(iv(2), e), mul(iv(2), e)))
    crude_rows.append(float(cov[1]))
    if not lt(cov, LAM):
        check(f'crude_S={S}', False, cover=cov[1])
check('crude_regime_each_S_2..38', all(c < float(LAM[0]) for c in crude_rows),
      worst=max(crude_rows), Lambda=LAM[0])
# monotone form: (S-1)/(1+log S) increasing, so the S=38 instance covers all S<=38
e_max_crude = Q(1, 4)/K
endpoint = div(iv(2*(S_CRUDE-1)), mul(mul(h(S_CRUDE), SQK), sqrt(iv(1-e_max_crude))))
check('crude_regime_endpoint_S=38', lt(endpoint, LAM), bound=endpoint[1], Lambda=LAM[0])


def tail_constants(K, S1, mu, q, sigma, m):
    h1 = h(S1)
    e1 = div(iv(1), mul(iv(K*S1*S1), mul(h1, h1)))
    bbar = div(iv(q*(sigma+Q(1, S1*S1))), mul(iv(K), mul(h1, h1)))
    ymax = add(bbar, e1)
    cF = div(iv(1), sqrt(sub(iv(1), div(ymax, iv(2)))))
    Rmax = mul(mul(SQ2, cF), add(sqrt(ymax), sqrt(bbar)))
    cB = sqrt(div(mul(SQ2, cF), mul(sqrt(sub(iv(2), bbar)), sub(iv(1), div(Rmax, PI)))))

    def g(k):
        X = mu*q**k
        return div(add(sqrt(iv(q*X+1)), sqrt(iv(q*X))), sqrt(iv(X-1)))
    lq = log(q)
    A = add(mul(mul(iv(4), mul(SQ2, cF)), add(sqrt(iv(mu+1)), sqrt(iv(mu)))),
            div(iv(2), sqrt(mul(iv(sigma), sub(iv(2), bbar)))))
    lead = iv(0)
    for k in range(m):
        lead = add(lead, sqrt(g(k)))
    sgm = sqrt(g(m))
    A = add(A, mul(mul(iv(8), cB), add(lead, mul(sgm, sub(div(log((sigma+Q(1, S1*S1))/mu), lq),
                                                           iv(m-1))))))
    B = div(mul(mul(iv(16), cB), sgm), lq)
    side = {
        'mu>1': mu > 1,
        'mu<=sigma*S1^2+1': mu <= sigma*S1*S1+1,
        'log_q((sigma S1^2+1)/mu)>=m-1': le(iv(m-1), div(log((sigma*S1*S1+1)/mu), lq)),
        'bbar+e1<=1': le(ymax, iv(1)),
        'Rmax<pi': lt(Rmax, PI),
        'g_k decreasing (algebraic, see note)': True,
    }
    bottom = mul(mul(iv(4), mul(SQ2, cF)), add(sqrt(iv(mu+1)), sqrt(iv(mu))))
    top = div(iv(2), sqrt(mul(iv(sigma), sub(iv(2), bbar))))
    return dict(A=A, B=B, e1=e1, bbar=bbar, cF=cF, cB=cB, Rmax=Rmax, side=side, h1=h1,
                sqrt_g=[sqrt(g(k)) for k in range(m+1)], bottom=bottom, top=top, logq=lq,
                logratio=log((sigma+Q(1, S1*S1))/mu))


T = tail_constants(K, S1, **PAR)
C = mul(LAM, SQK)
for k, v in T['side'].items():
    check('tail_side_'+k, v)
check('tail_slope_B<=Lambda*sqrt(K)', le(T['B'], C), B=T['B'][1], c=C[0])
lhs = add(T['A'], mul(T['B'], log(S1)))
rhs = mul(C, h(S1))
check('tail_endpoint_A+B*logS1<Lambda*sqrt(K)*(1+logS1)', lt(lhs, rhs),
      lhs=lhs[1], rhs=rhs[0], A=T['A'][1], cB=T['cB'][1], cF=T['cF'][1],
      Rmax=T['Rmax'][1], bbar=T['bbar'][1])
check('tail_A<=Lambda*sqrt(K)_(so_all_S>=S1_without_endpoint)', le(T['A'], C), A=T['A'][1], c=C[0])
TAIL_REPORT = {k: [float(T[k][0]), float(T[k][1])] for k in
               ('A', 'B', 'e1', 'bbar', 'cF', 'cB', 'Rmax', 'bottom', 'top', 'logq', 'logratio')}
TAIL_REPORT['sqrt_g'] = [[float(a), float(b)] for a, b in T['sqrt_g']]
TAIL_REPORT['Lambda_sqrtK'] = [float(C[0]), float(C[1])]
check('degenerate_single_line_2F(2e)<Lambda',
      lt(F_up(iv(2*Q(1, 4)/K), iv(2*Q(1, 4)/K)), LAM))


# ---------------- Part B: direct evaluation of Phi_S (exact) ----------------
def phi_direct(S, mu, q, sigma, m=None):
    """Bound of Lemma lem:multiscale with beta_k = mu q^k e, L minimal with mu q^L -1 >= sigma S^2,
    using exact-interval upper bounds: 4S R(beta0) + sum 4S sqrt(pi G_k/J_k) + C(S,2) G(beta_L)."""
    hS = h(S)
    e = div(iv(1), mul(iv(K*S*S), mul(hS, hS)))
    e = (e[1], e[1])                       # use upper end as tolerance (covering is monotone)
    L = 1
    while mu*q**L-1 < sigma*S*S:
        L += 1
    beta = [mul(iv(mu*q**k), e) for k in range(L+1)]
    ymax = add(beta[L], e)
    assert ymax[1] < 1
    R = lambda d: add(F_up(add(d, e), ymax), F_up(d, ymax))
    G = lambda b: div(mul(iv(4), e), sqrt(mul(sub(b, e), sub(add(iv(2), e), b))))
    tot = mul(iv(4*S), R(beta[0]))
    for k in range(L):
        Rk = R(beta[k+1])[1]
        J = math.floor(PI[0]/Rk)
        assert J >= 1
        tot = add(tot, mul(iv(4*S), sqrt(div(mul(PI, G(beta[k])), iv(J)))))
    tot = add(tot, mul(iv(Q(S*(S-1), 2)), G(beta[L])))
    return tot


direct = {}
for S in list(range(S1, 401)) + [10**3, 10**4, 10**6, 10**9, 10**15, 10**30, 10**60]:
    val = phi_direct(S, **PAR)
    direct[S] = float(val[1])
    if not lt(val, LAM):
        check(f'direct_Phi_S={S}', False, Phi=val[1])
check('direct_Phi_below_Lambda_S=39..400_and_huge', all(v < float(LAM[0]) for v in direct.values()),
      worst=max(direct.values()), worst_S=max(direct, key=direct.get))


# ---------------- Part C: float evidence ----------------
rng = np.random.default_rng(20261006)
Fa = lambda y: 2*np.arcsin(np.sqrt(np.clip(y, 0, 2)/2))
cdist = lambda a, b: np.abs((a-b+np.pi) % (2*np.pi)-np.pi)


def zset_measure(case, sgn, Delta, e):
    """measure of {theta: |a + r cos(theta-phi)| <= e} at 60 digits (no cancellation)."""
    mp.mp.dps = 60
    D, E = mp.mpf(Delta), mp.mpf(e)
    a, r = (sgn*(1-D), mp.mpf(1)) if case == 0 else (mp.mpf(sgn), 1-D)
    lo, hi = max(mp.mpf(-1), (-a-E)/r), min(mp.mpf(1), (-a+E)/r)
    return 0. if lo > hi else float(2*(mp.acos(lo)-mp.acos(hi)))


FA = lambda y: float(2*mp.asin(mp.sqrt(mp.mpf(y)/2)))


worst_loc = worst_G = worst_single = 0.
for trial in range(20000):
    Delta = 10**rng.uniform(-7, -0.5)
    e = Delta*10**rng.uniform(-3, 0.3)
    case, sgn = int(rng.random() < .5), int(rng.choice([-1, 1]))
    A, r = ((1-Delta)*sgn, 1.) if case == 0 else (float(sgn), 1-Delta)
    phi = rng.uniform(0, 2*np.pi)
    n = np.array([A, r*np.cos(phi), r*np.sin(phi)])
    # random atom in the plane n.u = 0
    basis = np.linalg.svd(n[None])[2][1:]
    u = rng.normal(size=2) @ basis
    ru = math.hypot(u[1], u[2])
    psi = math.atan2(u[2], u[1])
    c = max(-1, min(1, u[0]/ru))
    Cu = np.array([psi-math.acos(c), psi+math.acos(c)])
    th = np.linspace(0, 2*np.pi, 4001)
    th = np.concatenate([th, phi+np.pi*(A > 0)+np.linspace(-1, 1, 4001)*min(np.pi, 3*math.sqrt(Delta+e))])
    good = th[np.abs(n @ np.stack([np.ones_like(th), np.cos(th), np.sin(th)])) <= e]
    if good.size:
        d = np.min(cdist(good[:, None], Cu[None]), 1).max()
        worst_loc = max(worst_loc, d/(Fa(Delta+e)+Fa(Delta)))
    meas = zset_measure(case, sgn, Delta, e)
    worst_single = max(worst_single, meas/FA(2*e)/2)
    if e < Delta:
        worst_G = max(worst_G, meas/(2*(FA(Delta+e)-FA(Delta-e))))
check('float_lemma_centers_ratio<=1', worst_loc <= 1+1e-9, worst=worst_loc)
check('float_lemma_plane_transverse_ratio<=1', worst_G <= 1+1e-9, worst=worst_G)
check('float_lemma_plane_single_ratio<=1', worst_single <= 1+1e-9, worst=worst_single)

TH0, TH1 = 2*math.atan(.5), 2*math.atan(2.)


def circle_sup(U, G=1 << 15):
    """grid max over Theta_0 of min over pair planes of exact l1 distance |n.w|/|n|_inf."""
    i, j = np.triu_indices(len(U), 1)
    N = np.cross(U[i], U[j])
    N /= np.abs(N).max(1, keepdims=True)
    th = np.linspace(TH0, TH1, G)
    best = 0.
    for k in range(0, G, 4096):
        t = th[k:k+4096]
        W = np.stack([np.ones_like(t), np.cos(t), np.sin(t)], 1)
        best = max(best, np.abs(W @ N.T).min(1).max())
    return best


snap_rows = []
for S in (4, 8, 16, 32, 64, 128):
    lb = 1/(160*S*S*(1+math.log(S))**2)
    a = np.linspace(TH0, TH1, S)
    snap = circle_sup(np.stack([np.ones(S), np.cos(a), np.sin(a)], 1))
    best = snap
    for rho in (1.0005, 1.002, 1.01):      # slightly outside the cone, padded ends
        pad = (TH1-TH0)/(S-1)/2
        a2 = np.linspace(TH0-pad, TH1+pad, S)
        best = min(best, circle_sup(np.stack([np.ones(S), rho*np.cos(a2), rho*np.sin(a2)], 1)))
    snap_rows.append({'S': S, 'circle_lower_bound': lb, 'snapshot_grid_max': snap,
                      'best_ring_grid_max': best, 'ratio_best/lower': best/lb})
check('float_circle_upper_constructions_above_new_lower',
      all(r['best_ring_grid_max'] > r['circle_lower_bound'] for r in snap_rows),
      min_ratio=min(r['ratio_best/lower'] for r in snap_rows))


def l1_dist_plane(a, P):
    """exact l1 distance from a in R^4 to span of the 2 columns of P (vertex enumeration)."""
    best = np.inf
    for p in range(4):
        for q in range(p+1, 4):
            M = P[[p, q]]
            if abs(np.linalg.det(M)) < 1e-14:
                continue
            c = np.linalg.solve(M, a[[p, q]])
            best = min(best, np.abs(a-P @ c).sum())
    return best


arow = lambda x: np.array([1, x, x*x, x**3])/(1+x+x*x+x**3)
curve_rows = []
for S in (2, 4, 8, 16, 32, 64):
    nodes = np.exp(np.linspace(-math.log(2), math.log(2), S))
    xs = np.exp(np.linspace(-math.log(2), math.log(2), 4097))
    err = 0.
    for xq in xs:
        k = min(np.searchsorted(nodes, xq), S-1)
        k = max(k, 1)
        err = max(err, l1_dist_plane(arow(xq), np.stack([arow(nodes[k-1]), arow(nodes[k])], 1)))
    lb = 1/(960*S*S*(1+math.log(S))**2)
    curve_rows.append({'S': S, 'theorem_lower_bound': lb, 'snapshot_two_read_grid_max': err,
                       'upper_bound_9log2^2/(4(S-1)^2)': 9*math.log(2)**2/(4*(S-1)**2),
                       'ratio': err/lb})
check('float_curve_snapshot_above_new_lower', all(r['ratio'] > 1 for r in curve_rows),
      min_ratio=min(r['ratio'] for r in curve_rows))


def first_S(pred, lo=2, hi=10**12):
    """smallest integer S>=lo with pred(S) true, pred monotone on [lo,hi]."""
    if pred(lo):
        return lo
    while hi-lo > 1:
        mid = (lo+hi)//2
        lo, hi = (lo, mid) if pred(mid) else (mid, hi)
    return hi


hf = lambda S: 1+math.log(S)
cross = {
    'old_theorem_beats_1/(2304S^3)_from_S': first_S(lambda S: 2304*S > 6e8*hf(S)**2, 3),
    'new_theorem_beats_1/(2304S^3)_from_S': first_S(lambda S: 2304*S > 960*hf(S)**2),
    'old_theorem_beats_1/(480C(S,2)^2)_from_S': first_S(lambda S: 120*(S-1)**2 > 6e8*hf(S)**2),
    'new_theorem_beats_1/(480C(S,2)^2)_from_S': first_S(lambda S: 120*(S-1)**2 > 960*hf(S)**2),
    'old_relaxation_ratio>1_from_S': first_S(lambda S: (S-1)**2 > 5.4e9*math.log(2)**2*hf(S)**2),
    'new_relaxation_ratio>1_from_S': first_S(lambda S: (S-1)**2 > 9*960*math.log(2)**2*hf(S)**2),
}
# exact confirmation of the two new-bound claims used in the text
ok = all(lt(mul(iv(960), mul(h(S), h(S))), iv(2304*S)) for S in range(2, 200))
check('exact_new_beats_2304S^3_for_S=2..199_(monotone_beyond)', ok)
S11 = cross['new_theorem_beats_1/(480C(S,2)^2)_from_S']
check('exact_new_beats_norming_at_first_S',
      lt(mul(iv(960), mul(h(S11), h(S11))), iv(120*(S11-1)**2)) and
      not lt(mul(iv(960), mul(h(S11-1), h(S11-1))), iv(120*(S11-2)**2)), S=S11)

# relaxation ratio (S-1)^2/(9*960*(log2)^2*(1+log S)^2) > 1 exactly from S=461 (monotone in S)
rel = lambda S: lt(mul(iv(9*960), mul(mul(LOG2, LOG2), mul(h(S), h(S)))), iv((S-1)**2))
check('exact_relaxation_threshold_S=461', rel(461) and not rel(460))

# ======================================================================
# Part D (PRIMARY, single log): cross-scale chain, circle c'=2000, curve 12000
#   Psi = 4S R(beta0,e) + C(S,2) G(betaL,e) + 4S/tau + 2S tau sum_k (l_k+2Rbar_k) G(beta_k,e)
# ======================================================================
KX = Q(2000)
SQKX = sqrt(iv(KX))
SX_CRUDE = 66
PARX = dict(mu=Q(5, 2), q=Q(7), sigma=Q(1, 25))
SX1 = SX_CRUDE+1

# crude regime S<=66: C(S,2)*2*alpha(2e) <= 2(S-1)/(sqrt(KX h) sqrt(1-e)), e=1/(KX S^2 h)
rowsX = []
for S in range(2, SX_CRUDE+1):
    e = div(iv(1), mul(iv(KX*S*S), h(S)))
    cov = mul(iv(Q(S*(S-1), 2)*2), F_up(mul(iv(2), e), mul(iv(2), e)))
    rowsX.append(float(cov[1]))
check('X_crude_regime_each_S_2..66', all(c < float(LAM[0]) for c in rowsX), worst=max(rowsX))
eX0 = Q(1, 4)/KX
endX = div(iv(2*(SX_CRUDE-1)), mul(sqrt(mul(iv(KX), h(SX_CRUDE))), sqrt(iv(1-eX0))))
check('X_crude_endpoint_S=66_(monotone)', lt(endX, LAM), bound=endX[1])


def tailX(K, S1, mu, q, sigma):
    h1 = h(S1)
    e1 = div(iv(1), mul(iv(K*S1*S1), h1))
    bbar = div(iv(q*(sigma+Q(1, S1*S1))), mul(iv(K), h1))
    ymax = add(bbar, e1)
    cF = div(iv(1), sqrt(sub(iv(1), div(ymax, iv(2)))))
    Rb = mul(mul(SQ2, cF), add(sqrt(ymax), sqrt(bbar)))
    kR = add(iv(1), div(iv(1), sub(iv(1), div(Rb, PI))))
    gam = div(mul(iv(4), mul(SQ2, cF)), sqrt(sub(iv(2), bbar)))

    def g(k):
        X = mu*q**k
        return div(add(sqrt(iv(q*X+1)), sqrt(iv(q*X))), sqrt(iv(X-1)))
    lq = log(q)
    l0 = log((sigma+Q(1, S1*S1))/mu)
    a = add(g(0), div(mul(g(1), l0), lq))
    b = div(mul(iv(2), g(1)), lq)
    d = sub(b, a)
    Cc = add(mul(mul(iv(4), mul(SQ2, cF)), add(sqrt(iv(mu+1)), sqrt(iv(mu)))),
             div(iv(2), sqrt(mul(iv(sigma), sub(iv(2), bbar)))))
    D = mul(iv(8), sqrt(mul(kR, gam)))
    final = add(mul(D, sqrt(b)), div(mul(mul(Cc, Cc), sqrt(b)), mul(iv(2), mul(D, d))))
    side = {'mu>1': mu > 1, 'mu<=sigma*S1^2+1': mu <= sigma*S1*S1+1,
            'bbar+e1<=1': le(ymax, iv(1)), 'Rbar<pi': lt(Rb, PI),
            'arc_length_2Rbar*kR<pi': lt(mul(iv(2), mul(Rb, kR)), PI),
            'd=b-a>0': d[0] > 0}
    enc = dict(e1=e1, bbar=bbar, cF=cF, Rbar=Rb, kappaR=kR, gamma=gam, g0=g(0), g1=g(1),
               a=a, b=b, d=d, C=Cc, D=D, final=final, logq=lq, l0=l0)
    return final, side, enc


finX, sideX, encX = tailX(KX, SX1, **PARX)
TX = mul(LAM, SQKX)
for k, v in sideX.items():
    check('X_tail_side_'+k, v)
check('X_tail_closed_form_Dsqrt(b)+C^2sqrt(b)/(2Dd)<Lambda*sqrt(2000)', lt(finX, TX),
      lhs=finX[1], rhs=TX[0])


def psi_direct(S, mu, q, sigma):
    """Psi of Lemma lem:crossscale with the bands of the proof and the best tau on a grid."""
    hS = h(S)
    e = div(iv(1), mul(iv(KX*S*S), hS))
    e = (e[1], e[1])
    L = 1
    while mu*q**L-1 < sigma*S*S:
        L += 1
    beta = [mul(iv(mu*q**k), e) for k in range(L+1)]
    ymax = add(beta[L], e)
    R = lambda dd: add(F_up(add(dd, e), ymax), F_up(dd, ymax))
    G = lambda bb: div(mul(iv(4), e), sqrt(mul(sub(bb, e), sub(add(iv(2), e), bb))))
    base = add(mul(iv(4*S), R(beta[0])), mul(iv(Q(S*(S-1), 2)), G(beta[L])))
    W = iv(0)                              # sum_k (l_k + 2 Rbar_k) G_k
    for k in range(L):
        Rk = R(beta[k+1])[1]
        J = math.floor(PI[0]/Rk)
        assert J >= 1 and 2*PI[1]/J+2*Rk < PI[0]
        W = add(W, mul(add(iv(2*PI[1]/J), iv(2*Rk)), G(beta[k])))
    tau = Q(round(1/math.sqrt(float(W[1])/2)*10**6), 10**6)   # optimum of 4S/tau + 2S tau W
    return add(base, add(iv(4*S/tau), mul(iv(2*S*tau), W)))


directX = {}
for S in list(range(SX1, 401)) + [10**3, 10**4, 10**6, 10**9, 10**15, 10**30, 10**60]:
    directX[S] = float(psi_direct(S, **PARX)[1])
check('X_direct_Psi_below_Lambda_S=67..400_and_huge', all(v < float(LAM[0]) for v in directX.values()),
      worst=max(directX.values()), worst_S=str(max(directX, key=directX.get)))

# ---- float evidence for the cross-scale chain ----
# (i) minimal subcover of short closed arcs has multiplicity <= 2
worst_mult = 0
for trial in range(300):
    n = int(rng.integers(2, 40))
    c0 = rng.uniform(0, 2*np.pi, n)
    ln = rng.uniform(0.01, 1.5, n)
    arcs = list(zip(c0-ln/2, c0+ln/2))
    grid = np.linspace(0, 2*np.pi, 20001)
    inside = lambda A: cdist(grid, (A[0]+A[1])/2) <= (A[1]-A[0])/2
    keep = list(range(n))
    changed = True
    while changed:
        changed = False
        for j in list(keep):
            others = np.zeros_like(grid, bool)
            for i in keep:
                if i != j:
                    others |= inside(arcs[i])
            if not (inside(arcs[j]) & ~others).any():
                keep.remove(j)
                changed = True
                break
    worst_mult = max(worst_mult, int(sum(inside(arcs[i]).astype(int) for i in keep).max()))
check('float_minimal_subcover_multiplicity<=2', worst_mult <= 2, worst=worst_mult)


# (ii) actual coverage of explicit dictionaries <= Psi (any e, any S)
def coverage(U, e, G_=1 << 16):
    i, j = np.triu_indices(len(U), 1)
    N = np.cross(U[i], U[j])
    keep = np.linalg.norm(N, axis=1) > 1e-12
    N = N[keep]/np.abs(N[keep]).max(1, keepdims=True)
    th = np.linspace(0, 2*np.pi, G_, endpoint=False)
    W = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    return 2*np.pi*(np.abs(W @ N.T).min(1) <= e).mean()


def psi_float(S, e, mu=2.5, q=7., sigma=.04):
    L = 1
    while mu*q**L-1 < sigma*S*S:
        L += 1
    b = [mu*e*q**k for k in range(L+1)]
    if b[-1]+e > .5:
        return np.inf
    Fa_ = lambda y: 2*math.asin(math.sqrt(y/2))
    R = lambda dd: Fa_(dd+e)+Fa_(dd)
    G = lambda bb: 2*(Fa_(bb+e)-Fa_(bb-e))
    W = sum((2*math.pi/math.floor(math.pi/R(b[k+1]))+2*R(b[k+1]))*G(b[k]) for k in range(L))
    return 4*S*R(b[0])+S*(S-1)/2*G(b[L])+4*S*math.sqrt(2*W)


worst_cov = 0.
for S in (4, 8, 16, 24):
    for e in (1e-3, 1e-4, 1e-5):
        dicts = [np.stack([np.ones(S), np.cos(a), np.sin(a)], 1) for a in
                 (np.linspace(TH0, TH1, S), rng.uniform(0, 2*np.pi, S))]
        dicts.append(np.column_stack([rng.normal(size=S), rng.normal(size=(S, 2))]))
        tight = rng.uniform(1., 1.0+3*e, S)     # near-tangent cluster
        a = rng.uniform(1, 1.2, S)
        dicts.append(np.stack([np.ones(S), tight*np.cos(a), tight*np.sin(a)], 1))
        for U in dicts:
            worst_cov = max(worst_cov, coverage(U, e)/psi_float(S, e))
check('float_actual_coverage<=Psi', worst_cov <= 1, worst_ratio=worst_cov)

# (iii) constructions vs the single-log bounds
for r in snap_rows:
    S = r['S']; r['circle_lower_bound_single_log'] = 1/(2000*S*S*(1+math.log(S)))
for r in curve_rows:
    S = r['S']; r['theorem_lower_bound_single_log'] = 1/(12000*S*S*(1+math.log(S)))
check('float_constructions_above_single_log_bounds',
      all(r['best_ring_grid_max'] > r['circle_lower_bound_single_log'] for r in snap_rows) and
      all(r['snapshot_two_read_grid_max'] > r['theorem_lower_bound_single_log'] for r in curve_rows),
      min_ratio_circle=min(r['best_ring_grid_max']/r['circle_lower_bound_single_log'] for r in snap_rows),
      min_ratio_curve=min(r['snapshot_two_read_grid_max']/r['theorem_lower_bound_single_log']
                          for r in curve_rows))

crossX = {
    'refereed_6e7_beats_1/(2304S^3)_from_S': first_S(lambda S: 2304*S > 6e7*hf(S)),
    'new_12000_beats_1/(2304S^3)_from_S': first_S(lambda S: 2304*S > 12000*hf(S)),
    'new_12000_beats_1/(480C(S,2)^2)_from_S': first_S(lambda S: 120*(S-1)**2 > 12000*hf(S)),
    'new_12000_relaxation_ratio>1_from_S': first_S(lambda S: (S-1)**2 > 9*12000*math.log(2)**2*hf(S)),
    'squared_log_960_stronger_than_single_log_12000_below_S':
        first_S(lambda S: 960*hf(S)**2 > 12000*hf(S)),
}
ex = lambda S: lt(mul(iv(12000), h(S)), iv(2304*S))      # S/h increasing for S>1
c22 = crossX['new_12000_beats_1/(2304S^3)_from_S']
check('X_exact_beats_2304S^3_from_S=%d' % c22, ex(c22) and not ex(c22-1))
en = lambda S: lt(mul(iv(12000), h(S)), iv(120*(S-1)**2))  # (S-1)^2/h increasing
cn = crossX['new_12000_beats_1/(480C(S,2)^2)_from_S']
check('X_exact_beats_norming_from_S=%d' % cn, en(cn) and not en(cn-1))
er = lambda S: lt(mul(iv(9*12000), mul(mul(LOG2, LOG2), h(S))), iv((S-1)**2))
cr = crossX['new_12000_relaxation_ratio>1_from_S']
check('X_exact_relaxation_threshold_S=%d' % cr, er(cr) and not er(cr-1))
check('X_family_81920S^3_dominated_S=2_(S/h increasing)', lt(mul(iv(20000), h(2)), iv(81920*2)))
check('X_transfer_constants_6*2000=12000_10*2000=20000', 6*KX == 12000 and 10*KX == 20000)
PRIMARY = {'circle_constant': 2000, 'curve_constant': 12000, 'family_constant': 20000,
           'crude_regime': {'S_range': [2, SX_CRUDE], 'worst_cover': max(rowsX)},
           'tail_regime': {'S_from': SX1, 'params': {k: str(v) for k, v in PARX.items()},
                           'enclosures': {k: [float(v[0]), float(v[1])] for k, v in encX.items()},
                           'Lambda_sqrtK': [float(TX[0]), float(TX[1])]},
           'direct_Psi_samples': {str(k): v for k, v in directX.items()
                                  if k in (67, 100, 200, 400) or k >= 1000},
           'crossovers': crossX}


res = {
    'PRIMARY_single_log': PRIMARY,
    'VARIANT_squared_log': 'fields below',
    'label': 'Part A/B exact rational intervals; Part C float EVIDENCE ONLY',
    'circle_constant_K': 160, 'curve_constant': 960, 'family_constant': 1600,
    'Lambda_enclosure': [float(LAM[0]), float(LAM[1])],
    'crude_regime': {'S_range': [2, S_CRUDE], 'worst_cover': max(crude_rows)},
    'analytic_regime': {'S_from': S1, 'params': {k: str(v) for k, v in PAR.items()},
                        'A_upper': float(T['A'][1]), 'B_upper': float(T['B'][1]),
                        'Lambda_sqrtK_lower': float(C[0]), 'enclosures': TAIL_REPORT},
    'direct_Phi_samples': {str(k): v for k, v in direct.items()
                           if k in (39, 50, 100, 200, 400) or k >= 1000},
    'circle_upper_constructions': snap_rows,
    'curve_upper_constructions': curve_rows,
    'crossovers': cross,
    'checks': checks, 'failures': failures,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ('PRIMARY_single_log', 'crossovers', 'failures')}, indent=1))
print(f'{len(checks)-len(failures)}/{len(checks)} checks passed')
sys.exit(1 if failures else 0)

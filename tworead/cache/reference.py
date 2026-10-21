"""NumPy reference implementations used for replay and complete-cost timing.

The first group replays the certified scalar cache without PyTorch
(Section 5.1). The second group is the finite-window joint bilateral
oracle, the uniform two-read cache and the joint Gaussian-polynomial
comparator of Section 5.3 and Supplementary Section S4, all with a
radius-6, sigma_s=2 spatial kernel and replicated borders.
"""
import math
import time

import numpy as np


# ----- scalar softmax readouts -------------------------------------------
def nrows(q, k, b):
    z = np.asarray(q)[..., None]*k + b
    z -= z.max(axis=-1, keepdims=True)
    z = np.exp(z)
    return z / z.sum(axis=-1, keepdims=True)


def nenvelope(k, b, left, right):
    width = right-left
    K = np.ptp(k)
    if K == 0:
        return np.zeros_like(width)
    p = nrows((left+right)/2, k, b)
    centre = np.sum(p*k, axis=-1)
    moment = np.sum(p*(k-centre[..., None])**2, axis=-1)
    with np.errstate(divide='ignore'):
        logvar = np.minimum(2*np.log(K)-np.log(4), np.log(moment)+K*width/2)
    return width**2*np.exp(logvar)/4*(1+1e-12)


def nnodes(k, b, eps):
    nodes = np.array([0., 1.])
    while True:
        env = nenvelope(k, b, nodes[:-1], nodes[1:])
        split = env > eps
        if not split.any():
            return nodes, env
        assert len(nodes)+split.sum() < 65537
        nodes = np.sort(np.r_[nodes, ((nodes[:-1]+nodes[1:])/2)[split]])


def metadata(q, nodes, uniform=False):
    if uniform:
        lo = np.minimum(np.floor(q*(len(nodes)-1)).astype(np.int64), len(nodes)-2)
        f = q*(len(nodes)-1)-lo
    else:
        lo = np.clip(np.searchsorted(nodes, q, side='right')-1, 0, len(nodes)-2)
        f = (q-nodes[lo])/(nodes[lo+1]-nodes[lo])
    return lo, f


def ninterp(q, nodes, snapshots, uniform=False):
    lo, f = metadata(q, nodes, uniform)
    if snapshots.ndim > 1:
        f = f[..., None]
    return snapshots[lo]+f*(snapshots[lo+1]-snapshots[lo])


# ----- whole-image range-kernel filters ----------------------------------
def kernel():
    a = np.exp(-np.arange(-6, 7, dtype=np.float64)**2/8)
    return a/a.sum()


def blur(a):
    from scipy.ndimage import correlate1d
    k = kernel()
    return correlate1d(correlate1d(a, k, axis=-1, mode='nearest'),
                       k, axis=-2, mode='nearest')


def direct(v, g, sr, polynomial_order=None):
    """Independent explicit finite-window oracle, including replicated borders."""
    from numpy.lib.stride_tricks import sliding_window_view
    h, w = g.shape
    vg = sliding_window_view(np.pad(v, 6, mode='edge'), (13, 13))
    gg = sliding_window_view(np.pad(g, 6, mode='edge'), (13, 13))
    spatial = kernel()[:, None]*kernel()[None, :]
    result = np.empty_like(v)
    for y in range(0, h, 8):
        a = gg[y:y+8]; q = g[y:y+8, :, None, None]
        if polynomial_order is None:
            weights = np.exp(-(a-q)**2/(2*sr**2))
        else:
            x = (a-.5)*(q-.5)/sr**2
            term = np.ones_like(x); series = term.copy()
            for n in range(1, polynomial_order):
                term *= x/n; series += term
            weights = np.exp(-((a-.5)**2+(q-.5)**2)/(2*sr**2))*series
        weights *= spatial
        den = weights.sum(axis=(-2, -1))
        result[y:y+8] = np.einsum('hwab,hwab->hw', weights, vg[y:y+8])/den
    return result


def uniform(v, g, sr, S, stages=False):
    start = time.perf_counter_ns()
    snapshots = []
    for level in np.linspace(0, 1, S):
        z = -(g-level)**2/(2*sr**2)
        z -= z.max()
        a = np.exp(z)
        sums = blur(np.stack((a*v, a)))
        assert sums[1].min() > 0
        snapshots.append(sums[0]/sums[1])
    cache = np.stack(snapshots)
    built = time.perf_counter_ns()
    pos = g*(S-1)
    low = np.minimum(pos.astype(np.int64), S-2)
    fraction = pos-low
    a = np.take_along_axis(cache, low[None], axis=0)[0]
    b = np.take_along_axis(cache, (low+1)[None], axis=0)[0]
    result = a+fraction*(b-a)
    end = time.perf_counter_ns()
    return (result, {'construction_ms':(built-start)/1e6,
                     'readout_ms':(end-built)/1e6}) if stages else result


def gpa_order(sr, S):
    """Smallest Chernoff-certified GPA order matching the uniform-cache bound."""
    eps = 1/(16*sr**4*(S-1)**2)
    lam = (.5/sr)**2
    w0 = float(kernel()[6]**2)
    target = w0*eps/(2+eps)
    N = math.floor(lam)+1
    while -lam+N*(1+math.log(lam)-math.log(N)) > math.log(target):
        N += 1
    delta = math.exp(-lam+N*(1+math.log(lam)-math.log(N)))
    bound = 2*delta/(w0-delta)
    assert bound <= eps
    return N, delta, bound


def gpa(v, g, sr, N):
    x = (g-.5)/sr
    gaussian = np.exp(-x*x/2)
    power = np.ones_like(g)
    numerator = np.zeros_like(g); denominator = np.zeros_like(g)
    for n in range(N):
        f = gaussian*power
        sums = blur(np.stack((f*v, f)))
        numerator += power*sums[0]
        denominator += power*sums[1]
        if n+1 < N:
            power *= x/math.sqrt(n+1)
    assert denominator.min() > 0
    return numerator/denominator

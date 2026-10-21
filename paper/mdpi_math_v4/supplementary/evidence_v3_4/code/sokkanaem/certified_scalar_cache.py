"""Variance envelopes for two-read scalar softmax caches (Mathematics v3.2).

The envelope is an analytic guarantee in real arithmetic. This implementation
evaluates it in FP64 with a numerical guard; it is not interval arithmetic.
Node selection depends on keys, biases and a declared normalized tolerance,
never on the value vector or on the sampled evaluation queries.
"""
import math

import torch


def rows(queries, keys, biases):
    return (queries[..., None] * keys + biases).softmax(-1)


def interval_envelope(keys, biases, left, right):
    """Upper envelope on row interpolation error for each [left, right].

    The returned error is normalized by a supplied value radius R. The
    second returned quantity bounds Var_t(keys) on the entire interval.
    """
    width = right - left
    key_range = keys.max() - keys.min()
    if float(key_range) == 0:
        return torch.zeros_like(width), torch.zeros_like(width)
    mid = (left + right) / 2
    logits = mid[..., None] * keys + biases
    logp = logits.log_softmax(-1)
    centre = (logp.exp() * keys).sum(-1)
    # Any centre gives E[(key-centre)^2] >= Var. Keeping this moment in
    # log space avoids losing a tiny midpoint variance before tilting it.
    logmoment = torch.logsumexp(logp + 2 * (keys - centre[..., None]).abs().log(), -1)
    logcap = 2 * key_range.log() - math.log(4)
    logvariance = torch.minimum(logcap, logmoment + key_range * width / 2)
    variance = logvariance.exp()
    error = width.square() * variance / 4
    return error * (1 + 1e-12), variance * (1 + 1e-12)


def adaptive_nodes(keys, biases, start, stop, tolerance, max_cells=65536):
    """Bisect all uncertified cells; stop only when every envelope passes."""
    if tolerance <= 0 or stop <= start:
        raise ValueError('positive tolerance and an increasing interval are required')
    nodes = torch.tensor([start, stop], dtype=keys.dtype, device=keys.device)
    while True:
        bounds, _ = interval_envelope(keys, biases, nodes[:-1], nodes[1:])
        split = bounds > tolerance
        if not bool(split.any()):
            return nodes, bounds
        if len(nodes) - 1 + int(split.sum()) > max_cells:
            raise RuntimeError('requested tolerance exceeds the cell budget')
        nodes = torch.cat((nodes, ((nodes[:-1] + nodes[1:]) / 2)[split])).sort().values


def interpolate(queries, nodes, snapshots):
    if bool((queries < nodes[0]).any() or (queries > nodes[-1]).any()):
        raise ValueError('query outside the stored interval')
    low = (torch.searchsorted(nodes, queries.contiguous(), right=True) - 1).clamp(0, len(nodes) - 2)
    fraction = (queries - nodes[low]) / (nodes[low + 1] - nodes[low])
    if snapshots.ndim > 1:
        fraction = fraction[..., None]
    return snapshots[low] + fraction * (snapshots[low + 1] - snapshots[low])

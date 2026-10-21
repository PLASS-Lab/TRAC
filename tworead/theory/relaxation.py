#!/usr/bin/env python3
"""Independent-plane relaxation of the four-key curve (Proposition 2, Figure 4).

Checks the constructive tangent-plane library behind Proposition 2: M planes
span{a(t_k), a'(t_k)} at the midpoints of M equal log-query cells.  The l1 distance to a plane is an l1 regression whose
optimum is attained where two residual coordinates vanish, so it is computed
exactly (up to floating point) by enumerating the six coordinate pairs.
A grid cannot certify a continuous supremum; the bound itself is proved.
"""
import argparse
import json
import math
from itertools import combinations
from pathlib import Path

LOG2 = math.log(2.0)


def row(t):
    w = [math.exp(j * t) for j in range(4)]
    s = sum(w)
    return [x / s for x in w]


def drow(t):
    a = row(t)
    mu = sum(j * a[j] for j in range(4))
    return [a[j] * (j - mu) for j in range(4)]


def theorem_lower(s):
    """Lower bound of Theorem 1 on E_{S,2}/R."""
    h = 1 + math.log(s)
    return max(1 / (12000 * s * s * h), 1 / (960 * s * s * h * h))


def l1_dist_to_plane(z, p, q):
    best = sum(abs(x) for x in z)  # zero element of the plane
    for i, j in combinations(range(4), 2):
        det = p[i] * q[j] - p[j] * q[i]
        if abs(det) < 1e-300:
            continue
        al = (z[i] * q[j] - z[j] * q[i]) / det
        be = (p[i] * z[j] - p[j] * z[i]) / det
        best = min(best, sum(abs(z[k] - al * p[k] - be * q[k]) for k in range(4)))
    return best


def library_error(m, per_cell=9):
    """Largest l1 distance on per_cell equally spaced points of every cell."""
    h = 2 * LOG2 / m
    worst = 0.0
    for k in range(m):
        c = -LOG2 + (k + 0.5) * h
        p, q = row(c), drow(c)
        for j in range(per_cell):
            t = c - h / 2 + h * j / (per_cell - 1)
            worst = max(worst, l1_dist_to_plane(row(t), p, q))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-cell', type=int, default=9)
    args = parser.parse_args()
    rows = []
    for m in [1, 2, 4, 8, 16, 32, 64, 128]:
        err = library_error(m, args.per_cell)
        upper = 9 * LOG2 ** 2 / (4 * m * m)
        lower = 1 / (480 * m * m)
        assert err <= upper + 1e-12, (m, err, upper)
        rows.append({'M': m, 'grid_max': err, 'proved_upper': upper,
                     'proved_lower': lower})
    slope = (math.log(rows[-1]['grid_max']) - math.log(rows[-4]['grid_max'])) / math.log(128 / 16)
    pair_rows = []
    for s in [4, 8, 16, 32, 64]:
        m = s * (s - 1) // 2
        err = library_error(m, args.per_cell)
        upper = 9 * LOG2 ** 2 / (4 * m * m)
        assert err <= upper + 1e-12, (s, err, upper)
        pair_rows.append({'S': s, 'M': m, 'grid_max': err, 'proved_upper': upper})
    pair_slope = (math.log(pair_rows[-1]['grid_max']) - math.log(pair_rows[2]['grid_max'])) / math.log(64 / 16)
    sep = []
    for s in [2, 4, 16, 64, 256, 1024, 4096, 10 ** 6]:
        m = s * (s - 1) // 2
        relax_upper = 9 * LOG2 ** 2 / (4 * m * m)
        shared_lower = theorem_lower(s)
        sep.append({'S': s, 'pairs': m, 'relaxation_upper': relax_upper,
                    'shared_dictionary_lower': shared_lower,
                    'ratio_lower_over_relaxation': shared_lower / relax_upper})
    result = {'status': 'passed', 'per_cell': args.per_cell, 'library': rows,
              'pair_count_library': pair_rows, 'fitted_slope_in_S_16_to_64': pair_slope,
              'fitted_slope_M16_to_128': slope, 'separation': sep,
              'scope': 'floating grid diagnostic; continuous bounds are proved in the manuscript'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'slope_M': slope, 'slope_S': pair_slope, 'rows': rows, 'pairs': pair_rows}, indent=1))


if __name__ == '__main__':
    main()

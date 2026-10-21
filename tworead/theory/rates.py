#!/usr/bin/env python3
"""Upper construction of Theorem 1 on a dense query grid (Section 3, Figure 4 left).

Equally spaced snapshot rows of the four-key family, two adjacent scalar reads
per query. A grid checks the implementation and its analytic upper bound; it
cannot certify the minimax lower bound over all caches.
"""
import argparse
import json
import math
from pathlib import Path


def attention(t):
    x = math.exp(t)
    z = (1.0, x, x * x, x * x * x)
    total = sum(z)
    return tuple(v / total for v in z)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def evaluate(s, points):
    left, right = -math.log(2), math.log(2)
    h = (right - left) / (s - 1)
    rows = [attention(left + j * h) for j in range(s)]
    best_error, best_t, best_residual, best_pair, best_coefficients = -1.0, None, None, None, None
    for sample in range(points):
        t = left + (right - left) * sample / (points - 1)
        position = min(max((t - left) / h, 0.0), s - 1.0)
        j = min(int(position), s - 2)
        alpha = position - j
        a = attention(t)
        estimate = tuple((1 - alpha) * rows[j][i] + alpha * rows[j + 1][i]
                         for i in range(4))
        residual = tuple(a[i] - estimate[i] for i in range(4))
        error = sum(abs(v) for v in residual)
        if error > best_error:
            best_error, best_t, best_residual = error, t, residual
            best_pair, best_coefficients = (j, j + 1), (1 - alpha, alpha)
    values = tuple(1.0 if r >= 0 else -1.0 for r in best_residual)
    cells = [dot(row, values) for row in rows]
    reconstructed = sum(c * cells[j] for c, j in zip(best_coefficients, best_pair))
    actual = dot(attention(best_t), values)
    witness_error = abs(actual - reconstructed)
    upper = 9 * math.log(2) ** 2 / (4 * (s - 1) ** 2)
    assert abs(witness_error - best_error) < 1e-12
    assert best_error <= upper + 1e-12
    return {'S': s, 'grid_points': points, 'grid_max_error': best_error,
            'analytic_upper': upper, 'upper_holds_on_grid': True,
            'two_scalar_read_witness': {'x': math.exp(best_t),
                                        'cell_indices': best_pair,
                                        'values': values,
                                        'absolute_error': witness_error}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--points', type=int, default=16385)
    args = parser.parse_args()
    if args.points < 3 or args.output.exists():
        parser.error('points >= 3 and a new output path are required')
    rows = [evaluate(s, args.points) for s in (4, 8, 16, 32, 64, 128)]
    xs = [math.log(row['S']) for row in rows[2:]]
    ys = [math.log(row['grid_max_error']) for row in rows[2:]]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    result = {'scope': 'Four keys 0,1,2,3; t=log x in [-log2,log2]; R=1; equally spaced snapshot rows; two scalar cells read per query.',
              'qualification': 'Dense-grid numerical diagnostic of constructive upper policy, not a continuous theorem certificate or a test of arbitrary-cache minimax lower bounds.',
              'rows': rows, 'log_log_slope_S_16_to_128': slope}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows': [(r['S'], r['grid_max_error'], r['analytic_upper']) for r in rows],
                      'slope': slope}, indent=2))


if __name__ == '__main__':
    main()

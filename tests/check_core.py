"""Fast self-check of the core claims; run from the repository root: python tests/check_core.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from tworead.cache.bilateral import bilateral_exact, bilateral_two_read, error_bound
from tworead.cache.reference import direct, gpa, gpa_order, uniform
from tworead.cache.scalar import adaptive_nodes, interpolate, rows
from tworead.experiments.variance import embedding, stability
from tworead.theory.rates import evaluate
from tworead.theory.relaxation import library_error, theorem_lower

# Theorem 1 upper construction and Proposition 2 tangent library.
assert evaluate(16, 1025)['upper_holds_on_grid']
for m in (1, 4, 16):
    assert 1 / (480 * m * m) <= library_error(m) <= 9 * math.log(2) ** 2 / (4 * m * m)
ratio = lambda s: theorem_lower(s) / (9 * math.log(2) ** 2 / (s * s * (s - 1) ** 2))
assert ratio(460) < 1 < ratio(461)          # separation threshold quoted in Section 2.4

# Proposition 4: certified adaptive nodes never exceed their envelope.
torch.manual_seed(0)
keys, bias = torch.rand(40, dtype=torch.float64) * 6, torch.randn(40, dtype=torch.float64)
nodes, bounds = adaptive_nodes(keys, bias, 0, 1, 1e-4)
q = torch.linspace(0, 1, 2049, dtype=torch.float64)
err = (rows(q, keys, bias) - interpolate(q, nodes, rows(nodes, keys, bias))).abs().sum(-1)
cell = (torch.searchsorted(nodes, q, right=True) - 1).clamp(0, len(nodes) - 2)
assert bool((err <= bounds[cell] + 1e-12).all())

# Corollary 1 and the NumPy/PyTorch filter pair; GPA meets its matched bound.
rng = np.random.default_rng(0)
g, v = rng.uniform(0, 1, (12, 14)), rng.uniform(-1, 1, (12, 14))
tg, tv = torch.from_numpy(g)[None, None], torch.from_numpy(v)[None, None]
for sr, S in ((0.2, 8), (0.4, 32)):
    exact = direct(v, g, sr)
    assert abs(exact - bilateral_exact(tv, tg, 2., sr, chunk=4).numpy()[0, 0]).max() < 2e-12
    cached = uniform(v, g, sr, S)
    assert abs(cached - bilateral_two_read(tv, tg, 2., sr, S).numpy()[0, 0]).max() < 2e-12
    assert abs(cached - exact).max() <= error_bound(sr, S, 1.0)
    N, _, bound = gpa_order(sr, S)
    assert abs(gpa(v, g, sr, N) - exact).max() <= bound + 1e-10

# Proposition 5 embedding and stable normalization.
assert embedding()['symbolic_identity'] and embedding()['grid_max_abs'] < 1e-12
assert all(r['stable_constant_error'] < 1e-6 for r in stability())
print('core checks passed')

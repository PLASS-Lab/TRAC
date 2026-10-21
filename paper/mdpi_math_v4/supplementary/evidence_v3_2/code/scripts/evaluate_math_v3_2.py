#!/usr/bin/env python3
"""Frozen variance-envelope diagnostics for Mathematics v3.2; no model inference.

Run --freeze before --run. Data are reused NYUv2 diagnostic images. A family
holds its 169 guides/spatial weights fixed while an independent scalar query
sweeps [0,1]. These are range-query caches, not a timed full-image algorithm.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import sympy as sp
import torch

from sokkanaem.certified_scalar_cache import adaptive_nodes, interpolate, rows
from sokkanaem.range_cache_filter import bilateral_two_read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dirs/mdpi_math_v3_2'
PAPER = ROOT / 'paper/mdpi_math_v3_2'
SIGMAS = (0.05, 0.1, 0.2, 0.4)
TOLERANCES = (0.01, 0.001)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze():
    dest = OUT / 'freeze.json'
    if dest.exists():
        raise RuntimeError('freeze already exists; use an isolated directory for another experiment')
    manifest = ROOT / 'work_dirs/practical_upgrade_20261001/nyu_test_manifest.json'
    images = json.loads(manifest.read_text())['rows'][:622:27]
    assert len(images) == 24
    for row in images:
        assert sha(row['rgb']) == row['rgb_sha256']
        assert sha(row['depth']) == row['depth_sha256']
    sources = [Path(__file__), ROOT / 'sokkanaem/certified_scalar_cache.py',
               ROOT / 'sokkanaem/range_cache_filter.py', OUT / 'PLAN.md']
    write(dest, {'date_seoul': '2026-10-08', 'role': 'reused_diagnostic_not_holdout',
                 'sigma_s': 2, 'radius': 6, 'probes_per_image': 16,
                 'sigma_r': SIGMAS, 'normalized_tolerances': TOLERANCES,
                 'query_grid': 257, 'images': images,
                 'sources': {str(p.relative_to(ROOT)): sha(p) for p in sources},
                 'source_manifest_sha256': sha(manifest)})
    print('freeze written before diagnostic scoring', flush=True)


def stability():
    path = PAPER / 'supplementary/evidence_v3_2/range_cache_filter_legacy.py'
    spec = importlib.util.spec_from_file_location('legacy_range_cache', path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    result = []
    for dtype in (torch.float32, torch.float64):
        for layout, sr in (('constant', 0.05), ('two_regions', 0.01)):
            g = torch.full((1, 1, 8, 16), 0.5, dtype=dtype)
            if layout == 'two_regions':
                g[..., :8], g[..., 8:] = 0, 1
            v = torch.ones_like(g)
            old = legacy.bilateral_two_read(v, g, 1, sr, 2)
            new = bilateral_two_read(v, g, 1, sr, 2)
            shifted = bilateral_two_read(v + 7, g, 1, sr, 2)
            result.append({'dtype': str(dtype), 'layout': layout, 'sigma_r': sr,
                           'legacy_constant_error': float((old - 1).abs().max()),
                           'stable_constant_error': float((new - 1).abs().max()),
                           'stable_offset_error': float((shifted - new - 7).abs().max())})
    return result


def embedding():
    z, sigma, key = sp.symbols('z sigma key', real=True, positive=True)
    log_weight = (key ** 2 - key) / (2 * sigma ** 2)
    logits = -((z-key) ** 2) / (2*sigma ** 2) + log_weight
    expected = (z-sp.Rational(1, 2))*key/sigma**2 - z**2/(2*sigma**2)
    identity = sp.simplify(logits - expected) == 0
    sr = 0.4
    k = torch.arange(4, dtype=torch.float64) * (2*sr**2*math.log(2))
    logw = (k.square()-k)/(2*sr**2)
    q = torch.linspace(0, 1, 1025, dtype=torch.float64)
    actual = (logw - (q[:, None]-k).square()/(2*sr**2)).softmax(-1)
    x = 2**(2*q-1)
    target = torch.stack([x**i for i in range(4)], -1)
    target /= target.sum(-1, keepdim=True)
    return {'symbolic_identity': identity, 'grid_max_abs': float((actual-target).abs().max()),
            'sigma_r': sr, 'keys': k.tolist(), 'positive_weights': logw.exp().tolist(),
            'scope': 'arbitrary_positive_sample_weights_independent_query; no_fixed_Gaussian_lattice_lower_bound'}


def run():
    dest = OUT / 'results.json'
    if dest.exists():
        raise RuntimeError('completed scoring is not overwritten')
    frozen = json.loads((OUT/'freeze.json').read_text())
    for path, digest in frozen['sources'].items():
        assert sha(ROOT/path) == digest, path
    torch.set_num_threads(4)
    torch.manual_seed(20261008)
    dtype = torch.float64
    query = torch.linspace(0, 1, frozen['query_grid'], dtype=dtype)
    offset = torch.arange(-6, 7, dtype=dtype)
    logspatial = -(offset[:, None].square()+offset[None, :].square()).flatten()/8
    records, arrays = [], {}
    start = time.perf_counter()
    for image_no, image in enumerate(frozen['images']):
        assert sha(image['rgb']) == image['rgb_sha256']
        assert sha(image['depth']) == image['depth_sha256']
        rgb = np.asarray(Image.open(image['rgb']).convert('RGB'), dtype=np.float64)/255
        guide = torch.from_numpy(rgb @ np.array([0.299, 0.587, 0.114]))
        depth = torch.from_numpy(np.load(image['depth']).astype(np.float64))
        ys = np.linspace(6, guide.shape[0]-7, 4).round().astype(int)
        xs = np.linspace(6, guide.shape[1]-7, 4).round().astype(int)
        for probe, (yy, xx) in enumerate((y, x) for y in ys for x in xs):
            k0 = guide[yy-6:yy+7, xx-6:xx+7].flatten()
            v = depth[yy-6:yy+7, xx-6:xx+7].flatten()
            arrays[f'keys_{image_no}_{probe}'] = k0.numpy()
            arrays[f'values_{image_no}_{probe}'] = v.numpy()
            for sr in SIGMAS:
                keys, bias = k0/sr**2, logspatial-k0.square()/(2*sr**2)
                exact = rows(query, keys, bias)
                for tolerance in TOLERANCES:
                    nodes, certificates = adaptive_nodes(keys, bias, 0, 1, tolerance)
                    cached_rows = interpolate(query, nodes, rows(nodes, keys, bias))
                    errors = (exact-cached_rows).abs().sum(-1)
                    cells = (torch.searchsorted(nodes, query, right=True)-1).clamp(0, len(nodes)-2)
                    assert bool((errors <= certificates[cells]+1e-12).all())
                    uniform = torch.linspace(0, 1, len(nodes), dtype=dtype)
                    uniform_error = (exact-interpolate(query, uniform, rows(uniform, keys, bias))).abs().sum(-1)
                    radius = float((v.max()-v.min())/2)
                    vcentred = v-(v.max()+v.min())/2
                    actual_error = float(((exact-cached_rows) @ vcentred).abs().max())
                    at = int(errors.argmax())
                    corner = (exact[at]-cached_rows[at]).sign()
                    witness_error = float(((exact[at]-cached_rows[at])*corner).sum())
                    record = {'image': image['index'], 'probe': probe, 'yy': int(yy), 'xx': int(xx),
                              'sigma_r': sr, 'tolerance': tolerance, 'adaptive_S': len(nodes),
                              'local_uniform_S': max(2, 1+math.ceil(float(keys.max()-keys.min())/(4*math.sqrt(tolerance)))),
                              'global_uniform_S': 1+math.ceil(1/(4*sr**2*math.sqrt(tolerance))),
                              'certificate_max': float(certificates.max()),
                              'sampled_row_l1_error': float(errors.max()),
                              'same_S_uniform_sampled_error': float(uniform_error.max()),
                              'actual_value_radius': radius, 'actual_depth_query_error': actual_error,
                              'witness_error': witness_error, 'certificate_violations': 0}
                    records.append(record)
                    if image_no == 0 and probe == 0:
                        suffix = f'{sr}_{tolerance}'
                        arrays[f'nodes_{suffix}'] = nodes.numpy()
                        arrays[f'bounds_{suffix}'] = certificates.numpy()
        print(f'images {image_no+1}/24, schemes {len(records)}', flush=True)
    np.savez_compressed(OUT/'sampled_families.npz', **arrays)
    old = json.loads((ROOT/'work_dirs/bilateral_20261006/results.json').read_text())['summary']['q5']
    old = [r for r in old if r['channel'] != 'hard']
    write(dest, {'role': frozen['role'], 'arithmetic': 'FP64_analytic_envelope_evaluation_not_interval_certificate',
                 'records': records, 'stability': stability(), 'embedding': embedding(),
                 'legacy_quantiles': {'groups': len(old), 'individual_cases': sum(r['images'] for r in old),
                                      'quantile_better_individual_max': sum(r['images_quantile_better_max'] for r in old),
                                      'uniform_better_group_max': sum(r['quantile_worst_max']>r['uniform_worst_max'] for r in old)},
                 'families_sha256': sha(OUT/'sampled_families.npz'),
                 'elapsed_seconds': time.perf_counter()-start})
    print('diagnostics completed', flush=True)


def summarize():
    data = json.loads((OUT/'results.json').read_text())
    summaries, tex = [], []
    for sr in SIGMAS:
        for tolerance in TOLERANCES:
            subset = [r for r in data['records'] if r['sigma_r']==sr and r['tolerance']==tolerance]
            ratios = [r['local_uniform_S']/r['adaptive_S'] for r in subset]
            r = {'sigma_r': sr, 'tolerance': tolerance, 'schemes': len(subset),
                 'adaptive_S_median': float(np.median([r['adaptive_S'] for r in subset])),
                 'adaptive_S_min': min(r['adaptive_S'] for r in subset), 'adaptive_S_max': max(r['adaptive_S'] for r in subset),
                 'local_uniform_S_median': float(np.median([r['local_uniform_S'] for r in subset])),
                 'global_uniform_S': subset[0]['global_uniform_S'],
                 'local_uniform_over_adaptive_median': float(np.median(ratios)),
                 'local_uniform_over_adaptive_min': min(ratios), 'local_uniform_over_adaptive_max': max(ratios),
                 'max_sampled_error_over_tolerance': max(r['sampled_row_l1_error']/tolerance for r in subset),
                 'adaptive_better_same_S_count': sum(r['sampled_row_l1_error']<r['same_S_uniform_sampled_error'] for r in subset)}
            summaries.append(r)
            tex.append(f"{sr:g} & ${tolerance:g}$ & {r['global_uniform_S']} & {r['local_uniform_S_median']:g} & "
                       f"{r['adaptive_S_median']:g} & {r['local_uniform_over_adaptive_median']:.2f} & {r['max_sampled_error_over_tolerance']:.3f}\\\\")
    write(OUT/'summary.json', {'groups': summaries, 'schemes':len(data['records']),
                             'family_count': len(data['records'])//len(TOLERANCES), 'images':24,
                             'violations': sum(r['certificate_violations'] for r in data['records']),
                             'stability':data['stability'], 'embedding':data['embedding'], 'legacy_quantiles':data['legacy_quantiles']})
    (OUT/'adaptive_table_rows.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        freeze()
    if args.run:
        run()
    if args.summarize:
        summarize()

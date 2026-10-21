#!/usr/bin/env python3
"""Two-read bilateral cache against its a priori bound (plan: work_dirs/bilateral_20261006/PLAN.md).

Q1-Q5 accuracy questions in FP64 on the GPU; Q6 timing with --timing.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sokkanaem.range_cache_filter import bilateral_exact, bilateral_two_read, error_bound

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dirs/bilateral_20261006'
MANIFEST = ROOT / 'work_dirs/practical_upgrade_20261001/nyu_test_manifest.json'
SIGMA_S = (2, 4, 8)
SIGMA_R = (0.05, 0.1, 0.2, 0.4)
GRID_S = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
FINE_S = tuple(range(2, 33)) + tuple(sorted({int(round(32 * 1.08 ** k)) for k in range(1, 37)} - set(range(2, 33))))
EPS = (1e-2, 1e-3, 1e-4)


def load_images(device):
    rows = json.loads(MANIFEST.read_text())['rows'][:622:27]
    assert len(rows) == 24, len(rows)
    out = []
    for r in rows:
        rgb = np.asarray(Image.open(r['rgb']).convert('RGB'), dtype=np.float64) / 255
        guide = rgb @ np.array([0.299, 0.587, 0.114])
        depth = np.load(r['depth']).astype(np.float64)
        t = lambda a: torch.from_numpy(a)[None, None].to(device)
        out.append({'index': r['index'], 'guide': t(guide), 'depth': t(depth)})
    return out


def hard_image(device, h=480, w=640, block=4):
    """Two balanced key levels 0/1 (values -1/+1) plus sparse probe pixels whose guide values
    sweep [0,1] (value 0): the two-key extreme of Lemma interp, queried inside the range."""
    yy, xx = np.mgrid[:h, :w]
    g = (((yy // block) + (xx // block)) % 2).astype(np.float64)
    v = 2 * g - 1
    probe = (yy % 8 == 4) & (xx % 8 == 4)
    g[probe] = np.linspace(0, 1, int(probe.sum()))
    v[probe] = 0
    t = lambda a: torch.from_numpy(a)[None, None].to(device)
    return t(g), t(v)


def centred(v):
    lo, hi = float(v.min()), float(v.max())
    return v - (lo + hi) / 2, max((hi - lo) / 2, 1e-12)


def quantile_levels(guide, s):
    q = torch.quantile(guide.flatten()[::7], torch.linspace(0, 1, s, dtype=guide.dtype, device=guide.device))
    q[0], q[-1] = 0.0, 1.0
    return torch.cummax(q, 0).values + torch.arange(s, device=q.device, dtype=q.dtype) * 1e-9


def errors(exact, approx, r):
    d = (exact - approx).abs() / r
    return float(d.max()), float(d.mean())


def accuracy(device):
    images = load_images(device)
    hard_g, hard_v = hard_image(device)
    rows = []
    for ss in SIGMA_S:
        for sr in SIGMA_R:
            cases = [('self', im['index'], im['guide'], im['guide']) for im in images]
            cases += [('joint', im['index'], im['guide'], im['depth']) for im in images]
            cases += [('hard', -1, hard_g, hard_v)]
            for channel, idx, g, v in cases:
                v, r = centred(v)
                ref = bilateral_exact(v, g, ss, sr, chunk=32)
                row = {'channel': channel, 'image': idx, 'sigma_s': ss, 'sigma_r': sr, 'R': r,
                       'uniform': {}, 'quantile': {}}
                for s in sorted(set(GRID_S) | set(FINE_S)):
                    row['uniform'][s] = errors(ref, bilateral_two_read(v, g, ss, sr, s), r)
                    if s in GRID_S:
                        row['quantile'][s] = errors(ref, bilateral_two_read(v, g, ss, sr, s, quantile_levels(g, s)), r)
                rows.append(row)
            print(f'sigma_s={ss} sigma_r={sr} done', flush=True)
    return rows


def summarize(rows):
    out = {'q1_violations': [], 'q2_slopes': [], 'q3': [], 'q4': [], 'q5': []}
    for r in rows:
        for s, (mx, _) in r['uniform'].items():
            if mx > error_bound(r['sigma_r'], int(s), 1.0) * (1 + 1e-9):
                out['q1_violations'].append({**{k: r[k] for k in ('channel', 'image', 'sigma_s', 'sigma_r')}, 'S': s, 'err': mx})
    keys = sorted({(r['channel'], r['sigma_s'], r['sigma_r']) for r in rows})
    for ch, ss, sr in keys:
        group = [r for r in rows if (r['channel'], r['sigma_s'], r['sigma_r']) == (ch, ss, sr)]
        worst = {s: max(r['uniform'][s][0] for r in group) for s in GRID_S}
        xs = [s for s in GRID_S if s >= 16 and worst[s] > 1e-13]
        slope = float(np.polyfit(np.log([s - 1 for s in xs]), np.log([worst[s] for s in xs]), 1)[0]) if len(xs) >= 3 else None
        out['q2_slopes'].append({'channel': ch, 'sigma_s': ss, 'sigma_r': sr, 'slope': slope,
                                 'worst_over_bound_S16': worst[16] / error_bound(sr, 16)})
        for eps in EPS:
            s_bound = next((s for s in range(2, 10 ** 7) if error_bound(sr, s) <= eps), None)
            fine = sorted(group[0]['uniform'])
            ok = [s for s in fine if all(r['uniform'][t][0] <= eps for r in group for t in fine if t >= s)]
            s_needed = ok[0] if ok else None
            out['q3'].append({'channel': ch, 'sigma_s': ss, 'sigma_r': sr, 'eps': eps, 'S_bound': s_bound,
                              'S_needed': s_needed, 'ratio': s_bound / s_needed if s_needed else None})
        out['q4'].append({'channel': ch, 'sigma_s': ss, 'sigma_r': sr,
                          'max_err_over_bound': max(r['uniform'][s][0] / error_bound(sr, s) for r in group for s in GRID_S)})
        for s in (8, 32):
            uni = [r['uniform'][s] for r in group]
            qua = [r['quantile'][s] for r in group]
            out['q5'].append({'channel': ch, 'sigma_s': ss, 'sigma_r': sr, 'S': s,
                              'uniform_worst_max': max(u[0] for u in uni), 'quantile_worst_max': max(q[0] for q in qua),
                              'uniform_mean_mean': float(np.mean([u[1] for u in uni])),
                              'quantile_mean_mean': float(np.mean([q[1] for q in qua])),
                              'images_quantile_better_max': sum(q[0] < u[0] for u, q in zip(uni, qua)),
                              'images': len(group)})
    return out


def timing():
    out = []
    g = torch.rand(1, 1, 480, 640)
    v = torch.rand(1, 1, 480, 640)
    for device in ('cuda', 'cpu'):
        torch.set_num_threads(4)
        gd, vd = g.to(device), v.to(device)
        for ss in (2, 4, 8, 16):
            for name, fn in [('exact', lambda: bilateral_exact(vd, gd, ss, 0.1, chunk=16 if ss == 16 else 32))] + \
                            [(f'two_read_S{s}', (lambda s=s: bilateral_two_read(vd, gd, ss, 0.1, s))) for s in (8, 32)]:
                reps = 20 if device == 'cuda' else (3 if name == 'exact' and ss >= 8 else 5)
                fn()
                ts = []
                for _ in range(reps):
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    ts.append(1e3 * (time.perf_counter() - t0))
                out.append({'device': device, 'sigma_s': ss, 'method': name, 'ms': float(np.median(ts)), 'reps': reps})
                print(out[-1], flush=True)
    return {'image': '480x640 random, sigma_r 0.1, FP32', 'gpu': torch.cuda.get_device_name(0), 'cpu_threads': 4,
            'torch': torch.__version__, 'rows': out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.timing:
        (OUT / 'timing.json').write_text(json.dumps(timing(), indent=1))
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = accuracy('cuda')
    summary = summarize(rows)
    (OUT / 'results.json').write_text(json.dumps({'plan': 'PLAN.md', 'summary': summary, 'rows': rows}))
    print(json.dumps({k: summary[k] for k in ('q1_violations',)}, indent=1))


if __name__ == '__main__':
    main()

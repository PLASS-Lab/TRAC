#!/usr/bin/env python3
"""Replay all v3.2 scalar-cache schemes, source hashes and legacy preservation.

Requires only the bundled sampled families, not NYUv2 files or model weights.
The numerical replay supports the analytic theorem; it is not a formal proof.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from sokkanaem.certified_scalar_cache import adaptive_nodes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    default = root/'work_dirs/mdpi_math_v3_2'
    if not default.exists():
        default = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=default)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    result = json.loads((evidence/'results.json').read_text())
    freeze = json.loads((evidence/'freeze.json').read_text())
    summary = json.loads((evidence/'summary.json').read_text())
    failures, checks = [], 0

    def check(name, ok):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(name)

    code = evidence/'code'
    for path, digest in freeze['sources'].items():
        actual = (evidence/'PLAN.md') if path.endswith('/PLAN.md') else ((code if code.exists() else root)/path)
        check('frozen_source:'+path, actual.exists() and sha(actual)==digest)
    check('sampled_family_hash', sha(evidence/'sampled_families.npz')==result['families_sha256'])
    check('scheme_count', len(result['records'])==3072 and summary['schemes']==3072)
    check('embedding_symbolic', result['embedding']['symbolic_identity'])
    check('embedding_numeric', result['embedding']['grid_max_abs']<1e-14)
    check('quantile_individual_counts', result['legacy_quantiles']['individual_cases']==1152 and
          result['legacy_quantiles']['quantile_better_individual_max']==27)
    for row in result['stability']:
        check('constant_and_offset:'+row['dtype']+row['layout'],
              row['stable_constant_error']<1e-6 and row['stable_offset_error']<1e-6)

    torch.set_num_threads(4)
    raw = np.load(evidence/'sampled_families.npz')
    image_order = {r['index']: i for i,r in enumerate(freeze['images'])}
    q = torch.linspace(0,1,257,dtype=torch.float64)
    offset = torch.arange(-6,7,dtype=torch.float64)
    logspatial = -(offset[:,None].square()+offset[None,:].square()).flatten()/8
    max_replay_gap, max_ratio = 0., 0.
    for ordinal, record in enumerate(result['records']):
        image, probe = image_order[record['image']], record['probe']
        k0 = torch.from_numpy(raw[f'keys_{image}_{probe}'])
        value = torch.from_numpy(raw[f'values_{image}_{probe}'])
        sr, tolerance = record['sigma_r'], record['tolerance']
        keys, bias = k0/sr**2, logspatial-k0.square()/(2*sr**2)
        nodes, certificate = adaptive_nodes(keys,bias,0,1,tolerance)
        exact = (q[:,None]*keys+bias).softmax(-1)
        snapshots = (nodes[:,None]*keys+bias).softmax(-1)
        index = (torch.searchsorted(nodes,q,right=True)-1).clamp(0,len(nodes)-2)
        weight = ((q-nodes[index])/(nodes[index+1]-nodes[index]))[:,None]
        approximate = (1-weight)*snapshots[index]+weight*snapshots[index+1]
        error = (exact-approximate).abs().sum(-1)
        actual = float(error.max())
        gap = abs(actual-record['sampled_row_l1_error'])
        max_replay_gap = max(max_replay_gap,gap)
        max_ratio = max(max_ratio,actual/tolerance)
        check(f'scheme_{ordinal}_budget_and_certificate', len(nodes)==record['adaptive_S'] and
              float(certificate.max())<=tolerance and bool((error<=certificate[index]+1e-12).all()))
        check(f'scheme_{ordinal}_error_replay', gap<1e-12 and
              abs(actual-record['witness_error'])<1e-12)
        centred = value-(value.max()+value.min())/2
        check(f'scheme_{ordinal}_actual_values',
              abs(float(((exact-approximate)@centred).abs().max())-record['actual_depth_query_error'])<1e-10)
        expected_uniform = max(2,1+math.ceil(float(keys.max()-keys.min())/(4*math.sqrt(tolerance))))
        check(f'scheme_{ordinal}_uniform_comparator', expected_uniform==record['local_uniform_S'])
    for group in summary['groups']:
        records=[r for r in result['records'] if r['sigma_r']==group['sigma_r'] and r['tolerance']==group['tolerance']]
        check('group_medians:'+str((group['sigma_r'],group['tolerance'])),
              len(records)==384 and group['adaptive_S_median']==float(np.median([r['adaptive_S'] for r in records])) and
              group['local_uniform_S_median']==float(np.median([r['local_uniform_S'] for r in records])) and
              abs(group['local_uniform_over_adaptive_median']-float(np.median([r['local_uniform_S']/r['adaptive_S'] for r in records])))<1e-12)
    # A packaged replay need not have the old checkout beside it. When it is
    # available, verify every original file, including historical build files.
    baseline=root/'paper/mdpi_math_v3_1'
    inventory=evidence/'v3_1_baseline_sha256.json'
    if not inventory.exists():
        inventory=root/'paper/mdpi_math_v3_2/supplementary/evidence_v3_2/v3_1_baseline_sha256.json'
    preserved=0
    if baseline.exists():
        for path,digest in json.loads(inventory.read_text()).items():
            check('v3.1_preserved:'+path, sha(baseline/path)==digest)
            preserved += 1
    output={'passed':not failures,'checks':checks,'failures':failures,
            'schemes_replayed':3072,'query_row_checks':3072*257,
            'max_replay_abs_gap':max_replay_gap,'max_sampled_error_over_tolerance':max_ratio,
            'v3_1_files_preserved':preserved,
            'scope':'finite_FP64_replay_not_formal_continuous_certificate'}
    print(json.dumps(output,indent=2))
    if args.output:
        args.output.write_text(json.dumps(output,indent=2)+'\n')
    raise SystemExit(0 if output['passed'] else 1)


if __name__=='__main__':
    main()

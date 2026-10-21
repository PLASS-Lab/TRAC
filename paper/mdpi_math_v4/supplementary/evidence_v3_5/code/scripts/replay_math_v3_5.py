#!/usr/bin/env python3
"""Replay V3.5 without original datasets, model weights, GPU or network access.

The baselines environment supplies NumPy, SciPy, PyTorch and Pillow.
Run this copied script inside supplementary/evidence_v3_5/code/scripts/,
or pass --evidence /path/to/evidence_v3_5.
"""
import argparse
import json
import os
from pathlib import Path
import sys
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import torch

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from scripts.evaluate_math_v3_5 import (direct, uniform, gpa, gpa_order,
                                       masks, calibration, metrics, sha)
from sokkanaem.range_cache_filter import bilateral_exact,bilateral_two_read

def replay(evidence):
    f=json.loads((evidence/'freeze.json').read_text())
    result=json.loads((evidence/'results.json').read_text())['records']
    summary=json.loads((evidence/'summary.json').read_text())
    predictions=json.loads((evidence/'predictions.json').read_text())['records']
    torch.set_num_threads(1)
    assert len(result)==len(predictions)==72
    max_baseline_gap=0.;baseline_checks=0
    for r in result:
        frame=next(x for x in f['frames'] if x['id']==r['frame'])
        pp=evidence/'arrays'/f'{r["frame"]}_{r["model"]}.npz'
        pinfo=next(x for x in predictions if x['frame']==r['frame'] and x['model']==r['model'])
        assert sha(pp)==pinfo['array_sha256']
        v=np.load(pp)['prediction'].astype(np.float64)
        gt=np.load(evidence/'arrays'/f'{r["frame"]}_inputs.npz')['gt'].astype(np.float64)
        valid,edge=masks(gt,frame['max_depth'])
        cal=calibration(v,gt,valid,r['model'],frame['max_depth'])
        assert cal==r['calibration']
        mm=metrics(v,gt,valid,edge,cal)
        gap=max(abs(mm[k]-r['baseline'][k]) for k in mm)
        max_baseline_gap=max(gap,max_baseline_gap);assert gap<2e-11
        assert int(valid.sum())==r['valid_pixels'] and int(edge.sum())==r['edge_pixels']
        assert len(r['records'])==6
        R=float(np.ptp(v)/2)
        for z in r['records']:
            N,delta,pb=gpa_order(z['sigma_r'],z['S'])
            bound=R/(16*z['sigma_r']**4*(z['S']-1)**2)
            assert z['R']==R and z['GPA_N']==N and abs(z['bound']-bound)<1e-12
            assert z['cache_max_error']<=bound+z['tolerance']
            assert z['GPA_max_error']<=R*pb+z['tolerance']
            assert abs(z['cache_error_over_bound']-z['cache_max_error']/bound)<1e-12
        baseline_checks+=1
    # Native-image replay against separately implemented archived PyTorch filters.
    fid=f['frames'][0]['id'];tag='da2_small'
    data=np.load(evidence/'arrays'/f'{fid}_inputs.npz')
    pred=np.load(evidence/'arrays'/f'{fid}_{tag}.npz')['prediction'].astype(np.float64)
    guide=(data['rgb'].astype(np.float64)/255)@np.array([.299,.587,.114])
    mid=(pred.max()+pred.min())/2;v=pred-mid
    tv=torch.from_numpy(v)[None,None];tg=torch.from_numpy(guide)[None,None]
    row=next(r for r in result if r['frame']==fid and r['model']==tag)
    max_exact_gap=0.;max_cache_gap=0.;max_GPA_gap=0.;max_score_gap=0.
    frame=f['frames'][0];gt=data['gt'].astype(np.float64)
    valid,edge=masks(gt,frame['max_depth'])
    for sr in (.2,.4):
        exact=direct(v,guide,sr)
        te=bilateral_exact(tv,tg,2.,sr,chunk=8).numpy()[0,0]
        max_exact_gap=max(max_exact_gap,float(abs(exact-te).max()))
        for S in (8,32,64):
            cached=uniform(v,guide,sr,S)
            tc=bilateral_two_read(tv,tg,2.,sr,S).numpy()[0,0]
            max_cache_gap=max(max_cache_gap,float(abs(cached-tc).max()))
            N,_,_=gpa_order(sr,S);poly=gpa(v,guide,sr,N)
            explicit=direct(v,guide,sr,polynomial_order=N)
            max_GPA_gap=max(max_GPA_gap,float(abs(poly-explicit).max()))
            z=next(z for z in row['records'] if z['sigma_r']==sr and z['S']==S)
            for arm,a in [('exact',exact),('cached',cached),('GPA',poly)]:
                mm=metrics(a+mid,gt,valid,edge,row['calibration'])
                max_score_gap=max(max_score_gap,max(abs(mm[k]-z[arm][k]) for k in mm))
            assert abs(float(abs(cached-exact).max())-z['cache_max_error'])<1e-10
    assert max_exact_gap<2e-10 and max_cache_gap<2e-10 and max_GPA_gap<2e-9 and max_score_gap<1e-10
    # Independently reproduce paired cluster means and every descriptive interval.
    max_summary_gap=0.
    for g in summary['groups']:
        selected=[r for r in result if r['dataset']==g['dataset'] and r['model']==g['model']]
        names=sorted({r['group'] for r in selected})
        clusters={}
        for arm in ('baseline','exact','cached','GPA'):
            clusters[arm]={}
            for metric in ('AbsRel','RMSE','delta1','edge_AbsRel'):
                values=[]
                for name in names:
                    pairs=[]
                    for r in selected:
                        if r['group']!=name:continue
                        z=next(z for z in r['records'] if z['sigma_r']==g['sigma_r'] and z['S']==g['S'])
                        pairs.append((r['baseline'] if arm=='baseline' else z[arm])[metric])
                    values.append(float(np.mean(pairs)))
                clusters[arm][metric]=np.array(values)
                max_summary_gap=max(max_summary_gap,abs(float(np.mean(values))-g['metrics'][arm][metric]))
        ix=np.random.default_rng(20261009).integers(0,len(names),(3000,len(names)))
        for change,a,b in [('exact_minus_baseline','exact','baseline'),('cache_minus_exact','cached','exact'),
                           ('GPA_minus_exact','GPA','exact')]:
            for metric in clusters[a]:
                diff=clusters[a][metric]-clusters[b][metric]
                ci=np.quantile(diff[ix].mean(1),[.025,.975])
                z=g['changes'][change][metric]
                max_summary_gap=max(max_summary_gap,abs(float(diff.mean())-z['mean']),float(abs(ci-z['CI95']).max()))
    assert max_summary_gap<1e-12
    times=json.loads((evidence/'timings.json').read_text())['records']
    assert len(times)==24
    for row in times:
        assert all(len(t)==7 and min(t)>0 for t in row['durations_ms'].values())
        assert len(row['uniform_stages_ms'])==7
        assert row['logical_snapshot_bytes']==row['S']*np.prod(row['shape'])*8
    for g in summary['costs']:
        rr=[r for r in times if r['sigma_r']==g['sigma_r'] and r['S']==g['S']]
        for arm in ('direct','uniform','GPA'):
            actual=float(np.median([np.median(r['durations_ms'][arm]) for r in rr]))
            assert abs(actual-g['median_ms'][arm])<1e-10
    return {'passed':True,'baseline_scores_replayed':baseline_checks,'summary_groups_replayed':len(summary['groups']),
        'paired_intervals_replayed':len(summary['groups'])*12,'timing_configurations':24,
        'native_reference_shape':list(pred.shape),'native_filter_configurations_replayed':6,
        'max_baseline_metric_gap':max_baseline_gap,'max_archived_direct_gap':max_exact_gap,
        'max_archived_uniform_gap':max_cache_gap,'max_explicit_GPA_gap':max_GPA_gap,
        'max_metric_replay_gap':max_score_gap,'max_group_bootstrap_gap':max_summary_gap}

if __name__=='__main__':
    default=CODE.parent if CODE.parent.name=='evidence_v3_5' else CODE/'work_dirs/mdpi_math_v3_5'
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,default=default)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();output=replay(args.evidence)
    if args.output:args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

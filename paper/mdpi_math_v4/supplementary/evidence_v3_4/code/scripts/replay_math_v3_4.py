#!/usr/bin/env python3
"""Portable NumPy-only replay: python replay_math_v3_4.py EVIDENCE_DIRECTORY."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def probabilities(q,k,b):
    logits=q[:,None]*k+b
    weights=np.exp(logits-logits.max(axis=1,keepdims=True))
    return weights/weights.sum(axis=1,keepdims=True)

def approximation(q,nodes,p):
    j=np.minimum(np.maximum(np.searchsorted(nodes,q,side='right')-1,0),len(nodes)-2)
    f=((q-nodes[j])/(nodes[j+1]-nodes[j]))[:,None]
    return (1-f)*p[j]+f*p[j+1],j

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.evidence
    frozen=json.loads((root/'freeze.json').read_text());results=json.loads((root/'results.json').read_text())
    data=np.load(root/'families.npz');summary=json.loads((root/'summary.json').read_text())
    assert hashlib.sha256((root/'families.npz').read_bytes()).hexdigest()==results['families_sha256']
    assert hashlib.sha256((root/'freeze.json').read_bytes()).hexdigest()==results['freeze_sha256']
    q=np.linspace(0,1,frozen['query_points']);d=np.arange(-6,7);spatial=-(d[:,None]**2+d[None,:]**2).ravel()/8
    count=0;max_difference=0.;groups=[]
    for r in results['records']:
        prefix=f'i{r["image"]:02d}_p{r["probe"]:02d}';suffix=prefix+f'_s{r["sigma_r"]:g}_e{r["tolerance"]:g}'
        g=data[prefix+'_guides'];v=data[prefix+'_values'];nodes=data[suffix+'_nodes'];bounds=data[suffix+'_bounds']
        keys=g/r['sigma_r']**2;bias=spatial-g*g/(2*r['sigma_r']**2)
        p=probabilities(q,keys,bias);ap,index=approximation(q,nodes,probabilities(nodes,keys,bias))
        err=np.abs(p-ap).sum(axis=1);radius=(v.max()-v.min())/2
        actual=np.max(np.abs((p-ap)@(v-(v.max()+v.min())/2)))
        assert np.all(err<=bounds[index]+3e-12) and actual<=r['tolerance']*radius+3e-12
        diff=max(abs(err.max()-r['row_l1_error']),abs(actual-r['actual_red_error']))
        assert diff<3e-12
        max_difference=max(max_difference,float(diff));count+=len(q)
    rng=np.random.default_rng(20261009)
    for group in summary['groups']:
        rr=[r for r in results['records'] if r['sigma_r']==group['sigma_r'] and r['tolerance']==group['tolerance']]
        image_ratios=np.array([np.mean([r['local_uniform_S']/r['adaptive_S'] for r in rr if r['image']==i]) for i in range(1,25)])
        intervals=np.quantile(np.median(image_ratios[rng.integers(0,24,(2000,24))],axis=1),[.025,.975])
        assert abs(np.median(image_ratios)-group['image_mean_ratio_median'])<1e-12
        assert np.max(abs(intervals-group['image_bootstrap95']))<1e-12
        assert sum(r['adaptive_S']>r['local_uniform_S'] for r in rr)==group['adaptive_loses_storage']
        groups.append({'sigma_r':group['sigma_r'],'tolerance':group['tolerance'],'images':len(image_ratios)})
    output={'passed':True,'backend':'standalone_NumPy_no_Torch_no_source_photographs',
            'numpy':np.__version__,'caches':len(results['records']),'queries':count,
            'max_numeric_difference':max_difference,'image_statistic_groups_replayed':len(groups)}
    if args.output:args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()

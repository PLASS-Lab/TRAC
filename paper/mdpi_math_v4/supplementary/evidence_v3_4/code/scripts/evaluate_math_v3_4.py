#!/usr/bin/env python3
"""Frozen cross-dataset diagnostics and complete CPU cache lifetime costs."""
import argparse
import hashlib
import json
import math
import platform
import os
from contextlib import nullcontext
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from PIL import Image
import torch
from sokkanaem.certified_scalar_cache import adaptive_nodes, rows, interpolate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dirs/mdpi_math_v3_4'
SIGMAS = (.05, .1, .2, .4)
TOLERANCES = (.01, .001)

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, data):
    p.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

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

def download():
    dest = OUT/'kodak'; dest.mkdir(exist_ok=True)
    # Obtain an immutable public mirror revision; preserve canonical provenance.
    repo = 'MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite'
    with urllib.request.urlopen(f'https://api.github.com/repos/{repo}/commits/master', timeout=30) as f:
        commit = json.load(f)['sha']
    def fetch(i):
        name = f'kodim{i:02d}.png'; p = dest/name
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/PhotoCD_PCD0992/{i:02d}.png'
        if not p.exists():
            with urllib.request.urlopen(url, timeout=60) as f:
                body = f.read()
            Image.open(__import__('io').BytesIO(body)).verify()
            p.write_bytes(body)
        return {'index': i, 'path': str(p), 'source_url': url, 'sha256': sha(p),
                'size': list(Image.open(p).size)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        images = list(pool.map(fetch, range(1,25)))
    write(OUT/'download_manifest.json', {'canonical_url':'https://r0k.us/graphics/kodak/',
                                        'mirror':repo, 'commit':commit, 'images':images})
    print('Downloaded all 24 source photographs', flush=True)

def freeze():
    assert not (OUT/'freeze.json').exists(), 'freeze already exists'
    source = json.loads((OUT/'download_manifest.json').read_text())
    for image in source['images']:
        assert sha(image['path']) == image['sha256']
    paths = [Path(__file__), ROOT/'sokkanaem/certified_scalar_cache.py', OUT/'PLAN.md']
    write(OUT/'freeze.json', {'date_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'role':'new_cross_dataset_cache_validation_no_training_or_depth_accuracy',
          'sigma_r':SIGMAS, 'tolerances':TOLERANCES, 'query_points':1025,
          'probes_per_image':16, 'images':source['images'],
          'canonical_url':source['canonical_url'], 'mirror':source['mirror'], 'commit':source['commit'],
          'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths},
          'timing':{'probe':0,'sigma_r':.1,'tolerance':.001,'Q':[1,16,256,4096],
                    'warmup':3,'repeats':15,'seed':20261009,'threads':1}})
    print('Input and source hashes frozen before scoring', flush=True)

def elapsed(fun):
    start = time.perf_counter_ns(); value = fun()
    return (time.perf_counter_ns()-start)/1e6, value

def benchmark(k, b, v, rng):
    eps=.001; uniform_S=max(2,1+math.ceil(np.ptp(k)/(4*math.sqrt(eps))))
    def unodes():
        nodes=np.linspace(0,1,uniform_S)
        return nodes, np.full(uniform_S-1, (np.ptp(k)/(4*(uniform_S-1)))**2)
    builders={'adaptive':lambda:nnodes(k,b,eps), 'uniform':unodes}
    grids={arm:builder() for arm,builder in builders.items()}
    snapshots={arm:nrows(nodes,k,b)@v for arm,(nodes,_) in grids.items()}
    construction={arm:[] for arm in builders}; snapshot_times={arm:[] for arm in builders}
    for repeat in range(18):
        arms=list(builders) if repeat%2 else list(reversed(builders))
        for arm in arms:
            t,_=elapsed(builders[arm]); ts,_=elapsed(lambda:nrows(grids[arm][0],k,b)@v)
            if repeat>=3:
                construction[arm].append(t); snapshot_times[arm].append(ts)
    result={'uniform_S':uniform_S,'arms':{},'batches':[]}
    for arm,(nodes,env) in grids.items():
        result['arms'][arm]={'S':len(nodes),'node_selection_ms':construction[arm],
             'snapshot_ms':snapshot_times[arm], 'snapshot_bytes':snapshots[arm].nbytes,
             'node_bytes':nodes.nbytes if arm=='adaptive' else 16,
             'certificate_bytes':env.nbytes if arm=='adaptive' else 8}
    for Q in [1,16,256,4096]:
        q=rng.uniform(0,1,Q)
        functions={'direct':lambda:np.sum(nrows(q,k,b)*v,axis=-1)}
        for arm,(nodes,_) in grids.items():
            functions[arm]=lambda arm=arm,nodes=nodes:ninterp(q,nodes,snapshots[arm],arm=='uniform')
        durations={arm:[] for arm in functions}; meta={arm:[] for arm in grids}
        assert np.max(np.abs(functions['direct']()-functions['adaptive']())) <= eps*np.ptp(v)/2+1e-12
        for repeat in range(18):
            order=list(functions); order=order[repeat%3:]+order[:repeat%3]
            for arm in order:
                t,_=elapsed(functions[arm])
                if repeat>=3: durations[arm].append(t)
            for arm,(nodes,_) in grids.items():
                t,_=elapsed(lambda:metadata(q,nodes,arm=='uniform'))
                if repeat>=3: meta[arm].append(t)
        result['batches'].append({'Q':Q,'query_ms':durations,'metadata_ms':meta,
                                 'queries_sha256':hashlib.sha256(q.tobytes()).hexdigest()})
    return result

def run():
    assert not (OUT/'results.json').exists(), 'completed results are immutable'
    frozen=json.loads((OUT/'freeze.json').read_text())
    for p,h in frozen['sources'].items(): assert sha(ROOT/p)==h,p
    torch.set_num_threads(1)
    q=np.linspace(0,1,1025); qt=torch.from_numpy(q)
    offsets=np.arange(-6,7); logsp=-(offsets[:,None]**2+offsets[None,:]**2).ravel()/8
    arrays={}; records=[]; timings=[]; rng=np.random.default_rng(20261009)
    max_row_replay=0.; max_env_replay=0.; start=time.perf_counter()
    with nullcontext():
        for image in frozen['images']:
            assert sha(image['path'])==image['sha256']
            rgb=np.asarray(Image.open(image['path']).convert('RGB'),dtype=np.float64)/255
            guide=rgb@np.array([.299,.587,.114]); values=rgb[:,:,0]
            ys=np.linspace(6,guide.shape[0]-7,4).round().astype(int)
            xs=np.linspace(6,guide.shape[1]-7,4).round().astype(int)
            for probe,(y,x) in enumerate((y,x) for y in ys for x in xs):
                g=guide[y-6:y+7,x-6:x+7].ravel(); v=values[y-6:y+7,x-6:x+7].ravel()
                prefix=f'i{image["index"]:02d}_p{probe:02d}'
                arrays[prefix+'_guides']=g; arrays[prefix+'_values']=v
                for sr in SIGMAS:
                    k=g/sr**2; b=logsp-g**2/(2*sr**2)
                    kt,bt=torch.from_numpy(k),torch.from_numpy(b)
                    exact=rows(qt,kt,bt).numpy(); reference=nrows(q,k,b)
                    max_row_replay=max(max_row_replay,float(np.max(abs(exact-reference))))
                    for eps in TOLERANCES:
                        nt,et=adaptive_nodes(kt,bt,0,1,eps); nodes,env=nt.numpy(),et.numpy()
                        cached=interpolate(qt,nt,rows(nt,kt,bt)).numpy()
                        replay=ninterp(q,nodes,nrows(nodes,k,b))
                        assert np.max(abs(cached-replay))<2e-12
                        env_np=nenvelope(k,b,nodes[:-1],nodes[1:])
                        max_env_replay=max(max_env_replay,float(np.max(abs(env_np-env))))
                        lo,_=metadata(q,nodes); err=np.sum(abs(reference-replay),axis=-1)
                        violations=int(np.sum(err>env[lo]+2e-12))
                        assert violations==0
                        uniform=np.linspace(0,1,len(nodes)); ue=np.sum(abs(reference-ninterp(q,uniform,nrows(uniform,k,b),True)),axis=-1)
                        R=np.ptp(v)/2; v0=v-(v.max()+v.min())/2
                        actual=float(np.max(abs((reference-replay)@v0)))
                        assert actual<=eps*R+2e-12
                        suffix=prefix+f'_s{sr:g}_e{eps:g}'
                        arrays[suffix+'_nodes']=nodes; arrays[suffix+'_bounds']=env
                        records.append({'image':image['index'],'probe':probe,'y':int(y),'x':int(x),
                            'sigma_r':sr,'tolerance':eps,'adaptive_S':len(nodes),
                            'local_uniform_S':max(2,1+math.ceil(np.ptp(k)/(4*math.sqrt(eps)))),
                            'global_uniform_S':1+math.ceil(1/(4*sr**2*math.sqrt(eps))),
                            'row_l1_error':float(err.max()), 'same_S_uniform_error':float(ue.max()),
                            'envelope_max':float(env.max()),'value_radius':float(R),
                            'actual_red_error':actual,'violations':violations})
                    if probe==0 and sr==.1:
                        t=benchmark(k,b,v,rng); t['image']=image['index']; timings.append(t)
            print(f'Kodak {image["index"]:02d}/24: {len(records)} caches',flush=True)
        environment={'platform':platform.platform(),'processor':platform.processor(),
            'torch':torch.__version__,'numpy':np.__version__,
            'thread_environment':{k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            'torch_threads':torch.get_num_threads(),
            'cpu_model':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), 'unknown')}
    np.savez_compressed(OUT/'families.npz',**arrays)
    write(OUT/'results.json',{'freeze_sha256':sha(OUT/'freeze.json'),'records':records,'timings':timings,
        'environment':environment,'row_replay_max':max_row_replay,'envelope_replay_max':max_env_replay,
        'families_sha256':sha(OUT/'families.npz'),'elapsed_seconds':time.perf_counter()-start})
    print('Completed new diagnostics, independent replay and lifetime timing',flush=True)

def summarize():
    data=json.loads((OUT/'results.json').read_text()); rng=np.random.default_rng(20261009)
    groups=[]
    for sr in SIGMAS:
        for eps in TOLERANCES:
            rr=[r for r in data['records'] if r['sigma_r']==sr and r['tolerance']==eps]
            image_ratio=np.array([np.mean([r['local_uniform_S']/r['adaptive_S'] for r in rr if r['image']==i]) for i in range(1,25)])
            boot=np.median(image_ratio[rng.integers(0,24,(2000,24))],axis=1)
            groups.append({'sigma_r':sr,'tolerance':eps,'caches':len(rr),
                'adaptive_S_median':float(np.median([r['adaptive_S'] for r in rr])),
                'local_uniform_S_median':float(np.median([r['local_uniform_S'] for r in rr])),
                'image_mean_ratio_median':float(np.median(image_ratio)),
                'image_bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                'adaptive_loses_storage':sum(r['adaptive_S']>r['local_uniform_S'] for r in rr),
                'adaptive_better_same_S':sum(r['row_l1_error']<r['same_S_uniform_error'] for r in rr),
                'max_error_over_tolerance':max(r['row_l1_error']/eps for r in rr)})
    lifetime=[]
    for Q in [1,16,256,4096]:
        perimage=[]
        for t in data['timings']:
            batch=next(b for b in t['batches'] if b['Q']==Q)
            direct=float(np.median(batch['query_ms']['direct']))
            row={'image':t['image'],'direct_ms':direct}
            for arm,a in t['arms'].items():
                build=float(np.median(a['node_selection_ms']))+float(np.median(a['snapshot_ms']))
                total=build+float(np.median(batch['query_ms'][arm]))
                row[arm+'_total_ms']=total; row[arm+'_speedup']=direct/total
                row[arm+'_metadata_ms']=float(np.median(batch['metadata_ms'][arm]))
            perimage.append(row)
        lifetime.append({'Q':Q,'per_image':perimage,**{name:float(np.median([r[name] for r in perimage])) for name in perimage[0] if name!='image'},
            'adaptive_faster_images':sum(r['adaptive_speedup']>1 for r in perimage),
            'uniform_faster_images':sum(r['uniform_speedup']>1 for r in perimage)})
    costs={}
    for arm in ['adaptive','uniform']:
        aa=[t['arms'][arm] for t in data['timings']]
        costs[arm]={'S_median':float(np.median([a['S'] for a in aa])),
            'selection_ms_median':float(np.median([np.median(a['node_selection_ms']) for a in aa])),
            'snapshot_ms_median':float(np.median([np.median(a['snapshot_ms']) for a in aa])),
            'persistent_bytes_median':float(np.median([a['snapshot_bytes']+a['node_bytes']+a['certificate_bytes'] for a in aa]))}
    write(OUT/'summary.json',{'images':24,'families':1536,'caches':3072,'queries':3072*1025,
          'violations':sum(r['violations'] for r in data['records']),'groups':groups,
          'lifetime':lifetime,'costs':costs,'row_replay_max':data['row_replay_max'],
          'envelope_replay_max':data['envelope_replay_max']})
    print(json.dumps({'groups':groups,'costs':costs,'lifetime':[{k:v for k,v in r.items() if k!='per_image'} for r in lifetime]},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['download','freeze','run','summarize'])
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    globals()[args.action]()

#!/usr/bin/env python3
"""Frozen fresh inference, native-resolution joint-filter accuracy and matched costs.

freeze/predict: sokkanaem environment. test/audit/time/summary: baselines environment.
Dependencies are deliberately lazy so inference does not need scipy/matplotlib.
"""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'work_dirs/mdpi_math_v3_5'
SR = (.2, .4)
SIZES = (8, 32, 64)
MODEL_IDS = ('da2_small', 'zoedepth')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def kernel():
    a = np.exp(-np.arange(-6, 7, dtype=np.float64)**2/8)
    return a/a.sum()

def blur(a):
    from scipy.ndimage import correlate1d
    k = kernel()
    return correlate1d(correlate1d(a, k, axis=-1, mode='nearest'),
                       k, axis=-2, mode='nearest')

def direct(v, g, sr, polynomial_order=None):
    """Independent explicit finite-window oracle, including replicated borders."""
    from numpy.lib.stride_tricks import sliding_window_view
    h, w = g.shape
    vg = sliding_window_view(np.pad(v, 6, mode='edge'), (13, 13))
    gg = sliding_window_view(np.pad(g, 6, mode='edge'), (13, 13))
    spatial = kernel()[:, None]*kernel()[None, :]
    result = np.empty_like(v)
    for y in range(0, h, 8):
        a = gg[y:y+8]; q = g[y:y+8, :, None, None]
        if polynomial_order is None:
            weights = np.exp(-(a-q)**2/(2*sr**2))
        else:
            x = (a-.5)*(q-.5)/sr**2
            term = np.ones_like(x); series = term.copy()
            for n in range(1, polynomial_order):
                term *= x/n; series += term
            weights = np.exp(-((a-.5)**2+(q-.5)**2)/(2*sr**2))*series
        weights *= spatial
        den = weights.sum(axis=(-2, -1))
        result[y:y+8] = np.einsum('hwab,hwab->hw', weights, vg[y:y+8])/den
    return result

def uniform(v, g, sr, S, stages=False):
    start = time.perf_counter_ns()
    snapshots = []
    for level in np.linspace(0, 1, S):
        z = -(g-level)**2/(2*sr**2)
        z -= z.max()
        a = np.exp(z)
        sums = blur(np.stack((a*v, a)))
        assert sums[1].min() > 0
        snapshots.append(sums[0]/sums[1])
    cache = np.stack(snapshots)
    built = time.perf_counter_ns()
    pos = g*(S-1)
    low = np.minimum(pos.astype(np.int64), S-2)
    fraction = pos-low
    a = np.take_along_axis(cache, low[None], axis=0)[0]
    b = np.take_along_axis(cache, (low+1)[None], axis=0)[0]
    result = a+fraction*(b-a)
    end = time.perf_counter_ns()
    return (result, {'construction_ms':(built-start)/1e6,
                     'readout_ms':(end-built)/1e6}) if stages else result

def gpa_order(sr, S):
    eps = 1/(16*sr**4*(S-1)**2)
    lam = (.5/sr)**2
    w0 = float(kernel()[6]**2)
    target = w0*eps/(2+eps)
    N = math.floor(lam)+1
    while -lam+N*(1+math.log(lam)-math.log(N)) > math.log(target):
        N += 1
    delta = math.exp(-lam+N*(1+math.log(lam)-math.log(N)))
    bound = 2*delta/(w0-delta)
    assert bound <= eps
    return N, delta, bound

def gpa(v, g, sr, N):
    x = (g-.5)/sr
    gaussian = np.exp(-x*x/2)
    power = np.ones_like(g)
    numerator = np.zeros_like(g); denominator = np.zeros_like(g)
    for n in range(N):
        f = gaussian*power
        sums = blur(np.stack((f*v, f)))
        numerator += power*sums[0]
        denominator += power*sums[1]
        if n+1 < N:
            power *= x/math.sqrt(n+1)
    assert denominator.min() > 0
    return numerator/denominator

def tests():
    import torch
    from sokkanaem.range_cache_filter import bilateral_exact, bilateral_two_read
    torch.set_num_threads(1)
    rng = np.random.default_rng(20261009)
    records = []
    for case in ('random', 'binary', 'constant'):
        g = rng.uniform(0, 1, (9, 11)) if case!='binary' else rng.integers(0, 2, (9,11)).astype(float)
        v = rng.uniform(-2, 3, g.shape) if case!='constant' else np.full(g.shape, 2.5)
        for sr in SR:
            a = direct(v, g, sr)
            tv = torch.from_numpy(v)[None, None]; tg = torch.from_numpy(g)[None, None]
            torch_exact = bilateral_exact(tv, tg, 2., sr, chunk=4).numpy()[0,0]
            exact_gap = float(abs(a-torch_exact).max())
            assert exact_gap < 2e-12
            for S in SIZES:
                u = uniform(v, g, sr, S)
                tu = bilateral_two_read(tv, tg, 2., sr, S).numpy()[0,0]
                N, delta, bound = gpa_order(sr,S)
                p = gpa(v, g, sr, N)
                po = direct(v, g, sr, polynomial_order=N)
                gap = float(abs(p-po).max()); ugap=float(abs(u-tu).max())
                assert gap < 2e-11 and ugap < 2e-12
                R = float(np.ptp(v)/2)
                assert abs(p-a).max() <= R*bound+1e-10
                if case=='constant':
                    assert abs(p-2.5).max()<1e-12 and abs(u-2.5).max()<1e-12
                records.append({'case':case,'sigma_r':sr,'S':S,'N':N,
                                'explicit_GPA_gap':gap,'archived_uniform_gap':ugap,
                                'archived_direct_gap':exact_gap})
    dump(OUT/'implementation_tests.json', {'cases':records,'passed':True})
    print('Passed',len(records),'explicit/archived filter oracle comparisons',flush=True)

def timestamp_rows(path):
    return [(float(s[0]),s[1]) for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')
            for s in [line.split()]]

def inputs():
    rows=[]
    bonn = Path('/home/hyunsu/dataset_ssd/bonn/rgbd_bonn_dataset')
    for seq in sorted(bonn.glob('rgbd_bonn_*'))[:8]:
        dr=timestamp_rows(seq/'depth.txt'); dt=np.array([t for t,p in dr])
        pairs=[]
        for t,p in timestamp_rows(seq/'rgb.txt'):
            j=int(np.argmin(abs(dt-t)))
            if abs(dt[j]-t)<=.02:
                pairs.append((seq/p,seq/dr[j][1],abs(dt[j]-t)))
        for fraction in (1/3,2/3):
            i=round((len(pairs)-1)*fraction); rgb,dep,gap=pairs[i]
            rows.append({'dataset':'Bonn','group':seq.name,'frame':rgb.stem,
                         'rgb':str(rgb),'depth':str(dep),'pair_gap_s':gap,
                         'fraction':fraction,'ordinal':i,'population':len(pairs),'max_depth':10.})
    tartan=Path('/home/hyunsu/dataset_ssd/tartanair_v2')
    for env in sorted(tartan.iterdir()):
        camera=env/'Data_easy/P002/image_lcam_front'
        if not camera.is_dir(): continue
        pairs=[(p,env/'Data_easy/P002/depth_lcam_front'/f'{p.stem}_depth.png') for p in sorted(camera.glob('*.png'))]
        pairs=[(p,d) for p,d in pairs if d.is_file()]
        for fraction in (1/3,2/3):
            i=round((len(pairs)-1)*fraction); rgb,dep=pairs[i]
            rows.append({'dataset':'TartanAirV2','group':env.name,'frame':rgb.stem,
                         'rgb':str(rgb),'depth':str(dep),'pair_gap_s':0.,
                         'fraction':fraction,'ordinal':i,'population':len(pairs),'max_depth':80.})
    assert len(rows)==36 and len({r['group'] for r in rows if r['dataset']=='TartanAirV2'})==10
    for i,r in enumerate(rows):
        r['id']=f'f{i:02d}'
        r['rgb_sha256']=sha(r['rgb']);r['depth_sha256']=sha(r['depth'])
        r['native_size']=list(Image.open(r['rgb']).size)[::-1]
    return rows

def decoded(r):
    rgb=np.asarray(Image.open(r['rgb']).convert('RGB'))
    raw=np.asarray(Image.open(r['depth']))
    if r['dataset']=='Bonn':
        gt=raw.astype(np.float32)/5000
    else:
        assert raw.ndim==3 and raw.shape[2]==4 and raw.dtype==np.uint8
        gt=raw[..., [2,1,0,3]].copy().view('<f4').reshape(raw.shape[:2])
    assert gt.shape==rgb.shape[:2]
    return rgb,gt

def freeze():
    assert not (OUT/'freeze.json').exists()
    assert json.loads((OUT/'implementation_tests.json').read_text())['passed']
    old=json.loads((ROOT/'work_dirs/mdpi_math_v2/freeze.json').read_text())
    models={k:old['models'][k] for k in MODEL_IDS}
    for m in models.values():
        for p,h in m['files'].items(): assert sha(Path(m['local_path'])/p)==h,p
    frames=inputs()
    import cv2
    r=next(r for r in frames if r['dataset']=='TartanAirV2')
    _,gt=decoded(r); reference=cv2.imread(r['depth'],cv2.IMREAD_UNCHANGED).view('<f4').squeeze(-1)
    assert np.array_equal(gt,reference,equal_nan=True)
    paths=[Path(__file__),OUT/'PLAN.md',ROOT/'sokkanaem/range_cache_filter.py']
    dump(OUT/'freeze.json',{'status':'frozen_before_prediction_and_scoring',
        'date_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'data_role':'new_output_filter_replication_on_reused_project_datasets',
        'models':models,'frames':frames,'sigma_r':SR,'S':SIZES,
        'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths},
        'official_depth_decoder_matches_OpenCV':True,
        'timing':{'frames':[frames[0]['id'],r['id']],'warmup':2,'repeats':7,'threads':1},
        'bootstrap':{'resamples':3000,'seed':20261009}})
    print('Frozen',len(frames),'frames and',len(models),'public checkpoints before scores',flush=True)

def frozen():
    f=json.loads((OUT/'freeze.json').read_text())
    for p,h in f['sources'].items(): assert sha(ROOT/p)==h,p
    return f

def predict():
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation
    f=frozen(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    dest=OUT/'arrays';dest.mkdir(exist_ok=True)
    for r in f['frames']:
        for key in ('rgb','depth'): assert sha(r[key])==r[key+'_sha256']
        rgb,gt=decoded(r)
        path=dest/f'{r["id"]}_inputs.npz'
        if not path.exists(): np.savez_compressed(path,rgb=rgb,gt=gt)
    records=[]
    for tag,m in f['models'].items():
        for p,h in m['files'].items(): assert sha(Path(m['local_path'])/p)==h,p
        processor=AutoImageProcessor.from_pretrained(m['local_path'],local_files_only=True)
        model=AutoModelForDepthEstimation.from_pretrained(m['local_path'],local_files_only=True,
                                                        dtype=torch.float32).eval().to('cuda')
        for r in f['frames']:
            path=dest/f'{r["id"]}_{tag}.npz'
            if path.exists():
                meta=json.loads(path.with_suffix('.json').read_text());records.append(meta);continue
            img=Image.open(r['rgb']).convert('RGB');shape=r['native_size']
            batch=processor(images=img,return_tensors='pt')
            batch={k:v.to('cuda') for k,v in batch.items()}
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                output=model(**batch)
                if tag=='zoedepth':
                    result=processor.post_process_depth_estimation(output,source_sizes=[tuple(shape)],
                                     target_sizes=[tuple(shape)])[0]['predicted_depth']
                else:
                    result=F.interpolate(output.predicted_depth.unsqueeze(1),size=tuple(shape),
                                        mode='bilinear',align_corners=False)[0,0]
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            prediction=result.detach().float().cpu().numpy()
            assert prediction.shape==tuple(shape) and np.isfinite(prediction).all()
            np.savez_compressed(path,prediction=prediction)
            meta={'frame':r['id'],'model':tag,'array_sha256':sha(path),'forward_wall_s':elapsed,
                  'min':float(prediction.min()),'max':float(prediction.max()),
                  'processor_input_shape':list(batch['pixel_values'].shape)}
            dump(path.with_suffix('.json'),meta);records.append(meta)
            print('Predicted',r['id'],r['dataset'],tag,round(elapsed,3),'s',flush=True)
        del model
        torch.cuda.empty_cache()
    dump(OUT/'predictions.json',{'runtime':{'python':platform.python_version(),'numpy':np.__version__,
        'torch':torch.__version__,'transformers':transformers.__version__,
        'device':torch.cuda.get_device_name(0),'precision':'FP32','TF32':False},'records':records})

def masks(gt,maximum):
    from scipy.ndimage import maximum_filter
    valid=np.isfinite(gt)&(gt>.1)&(gt<maximum)
    log=np.log(np.maximum(gt,.1))
    jumps=np.zeros_like(valid)
    z=valid[:,1:]&valid[:,:-1]&(abs(log[:,1:]-log[:,:-1])>.15)
    jumps[:,1:]|=z;jumps[:,:-1]|=z
    z=valid[1:,:]&valid[:-1,:]&(abs(log[1:,:]-log[:-1,:])>.15)
    jumps[1:,:]|=z;jumps[:-1,:]|=z
    edge=maximum_filter(jumps,size=7,mode='nearest')&valid
    assert valid.sum()>1000 and edge.sum()>10
    return valid,edge

def calibration(pred,gt,valid,tag,maximum):
    if tag=='zoedepth': return {'mode':'raw_metres','floor':.001}
    design=np.column_stack((pred[valid],np.ones(valid.sum())))
    a,b=np.linalg.lstsq(design,1/gt[valid],rcond=None)[0]
    return {'mode':'baseline_affine_inverse_depth_GT_assisted','a':float(a),'b':float(b),
            'inverse_floor':1/maximum}

def metric_depth(pred,cal):
    if cal['mode']=='raw_metres':
        return np.maximum(pred,cal['floor']),float(np.mean(pred<cal['floor']))
    inv=cal['a']*pred+cal['b'];floor=cal['inverse_floor']
    return 1/np.maximum(inv,floor),float(np.mean(inv<floor))

def metrics(pred,gt,valid,edge,cal):
    dep,clip=metric_depth(pred,cal)
    p=dep[valid];t=gt[valid]
    ratio=np.maximum(p/t,t/p)
    return {'AbsRel':float(np.mean(abs(p-t)/t)),
            'RMSE':float(np.sqrt(np.mean((p-t)**2))),
            'delta1':float(np.mean(ratio<1.25)),
            'edge_AbsRel':float(np.mean(abs(dep[edge]-gt[edge])/gt[edge])),
            'floor_fraction_full_image':clip}

def audit():
    f=frozen();dest=OUT/'frame_results';dest.mkdir(exist_ok=True)
    first={dataset:next(r['id'] for r in f['frames'] if r['dataset']==dataset)
           for dataset in ('Bonn','TartanAirV2')}
    for tag in MODEL_IDS:
        for r in f['frames']:
            path=dest/f'{r["id"]}_{tag}.json'
            if path.exists(): continue
            arrays=np.load(OUT/'arrays'/f'{r["id"]}_inputs.npz')
            rgb,gt=arrays['rgb'],arrays['gt'].astype(np.float64)
            pred=np.load(OUT/'arrays'/f'{r["id"]}_{tag}.npz')['prediction'].astype(np.float64)
            guide=(rgb.astype(np.float64)/255)@np.array([.299,.587,.114])
            valid,edge=masks(gt,r['max_depth'])
            cal=calibration(pred,gt,valid,tag,r['max_depth'])
            mid=float((pred.max()+pred.min())/2);R=float(np.ptp(pred)/2);v=pred-mid
            base=metrics(pred,gt,valid,edge,cal);records=[]
            visuals={'rgb':rgb,'gt':gt,'valid':valid,'baseline':pred}
            for sr in SR:
                exact=direct(v,guide,sr)
                em=metrics(exact+mid,gt,valid,edge,cal)
                for S in SIZES:
                    cache=uniform(v,guide,sr,S)
                    N,delta,pb=gpa_order(sr,S);poly=gpa(v,guide,sr,N)
                    bound=R/(16*sr**4*(S-1)**2);tol=1e-10*max(1,R)
                    err=float(abs(cache-exact).max()); pe=float(abs(poly-exact).max())
                    assert err<=bound+tol and pe<=R*pb+tol
                    cm=metrics(cache+mid,gt,valid,edge,cal)
                    pm=metrics(poly+mid,gt,valid,edge,cal)
                    records.append({'sigma_r':sr,'S':S,'R':R,'bound':bound,'tolerance':tol,
                        'cache_max_error':err,'cache_error_over_bound':err/bound if bound else 0.,
                        'GPA_N':N,'GPA_kernel_bound':delta,'GPA_output_bound':R*pb,
                        'GPA_max_error':pe,'GPA_error_over_bound':pe/(R*pb) if R else 0.,
                        'exact':em,'cached':cm,'GPA':pm,'violations':0})
                    if r['id']==first[r['dataset']] and sr==.2 and S==32:
                        visuals.update(exact=exact+mid,cached=cache+mid,GPA=poly+mid)
            result={'frame':r['id'],'model':tag,'dataset':r['dataset'],'group':r['group'],
                'shape':list(pred.shape),'valid_pixels':int(valid.sum()),'edge_pixels':int(edge.sum()),
                'calibration':cal,'baseline':base,'records':records}
            dump(path,result)
            if r['id']==first[r['dataset']]:
                np.savez_compressed(OUT/'arrays'/f'{r["id"]}_{tag}_qualitative.npz',**visuals)
            print('Audited',r['id'],r['dataset'],tag,'max ratio',
                  round(max(z['cache_error_over_bound'] for z in records),4),flush=True)
    records=[json.loads(p.read_text()) for p in sorted(dest.glob('*.json'))]
    assert len(records)==72
    dump(OUT/'results.json',{'runtime':{'numpy':np.__version__,'python':platform.python_version(),
                                      'platform':platform.platform(),'precision':'FP64'},'records':records})

def timing():
    f=frozen();import scipy
    timing_rows=[]
    for fid in f['timing']['frames']:
        rgb=np.load(OUT/'arrays'/f'{fid}_inputs.npz')['rgb']
        guide=(rgb.astype(np.float64)/255)@np.array([.299,.587,.114])
        r=next(r for r in f['frames'] if r['id']==fid)
        for tag in MODEL_IDS:
            v=np.load(OUT/'arrays'/f'{fid}_{tag}.npz')['prediction'].astype(np.float64)
            v-= (v.max()+v.min())/2
            for sr in SR:
                for S in SIZES:
                    N,delta,pb=gpa_order(sr,S)
                    functions={'direct':lambda:direct(v,guide,sr),
                               'uniform':lambda:uniform(v,guide,sr,S,stages=True),
                               'GPA':lambda:gpa(v,guide,sr,N)}
                    durations={k:[] for k in functions};stages=[]
                    for rep in range(9):
                        arms=list(functions);arms=arms[rep%3:]+arms[:rep%3]
                        for arm in arms:
                            start=time.perf_counter_ns();result=functions[arm]()
                            dt=(time.perf_counter_ns()-start)/1e6
                            if rep>=2:
                                durations[arm].append(dt)
                                if arm=='uniform': stages.append(result[1])
                    timing_rows.append({'frame':fid,'dataset':r['dataset'],'model':tag,
                        'shape':list(guide.shape),'sigma_r':sr,'S':S,'GPA_N':N,
                        'uniform_scalar_filterings':2*S,'GPA_scalar_filterings':2*N,
                        'logical_snapshot_bytes':int(S*guide.size*8),'durations_ms':durations,
                        'uniform_stages_ms':stages})
                    dump(OUT/'timings_partial.json',timing_rows)
                    print('Timed',fid,tag,sr,S,{k:round(float(np.median(a)),1) for k,a in durations.items()},flush=True)
    dump(OUT/'timings.json',{'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,
        'python':platform.python_version(),'platform':platform.platform(),'thread_env':
        {k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'CPU': next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                     if s.startswith('model name'))),'precision':'FP64'},'records':timing_rows})

def summary():
    f=frozen();results=json.loads((OUT/'results.json').read_text())['records']
    allrows=[x for r in results for x in r['records']]
    groups=[]
    for dataset in ('Bonn','TartanAirV2'):
        for model in MODEL_IDS:
            selected=[r for r in results if r['dataset']==dataset and r['model']==model]
            for sr in SR:
                for S in SIZES:
                    clusters=sorted({r['group'] for r in selected})
                    values={arm:{metric:[] for metric in ('AbsRel','RMSE','delta1','edge_AbsRel')}
                            for arm in ('baseline','exact','cached','GPA')}
                    for group in clusters:
                        rr=[r for r in selected if r['group']==group]
                        for arm in values:
                            for metric in values[arm]:
                                vals=[]
                                for r in rr:
                                    z=next(z for z in r['records'] if z['sigma_r']==sr and z['S']==S)
                                    vals.append((r['baseline'] if arm=='baseline' else z[arm])[metric])
                                values[arm][metric].append(float(np.mean(vals)))
                    rng=np.random.default_rng(f['bootstrap']['seed'])
                    ix=rng.integers(0,len(clusters),(3000,len(clusters)))
                    changes={}
                    for name,a,b in [('exact_minus_baseline','exact','baseline'),
                                     ('cache_minus_exact','cached','exact'),('GPA_minus_exact','GPA','exact')]:
                        changes[name]={}
                        for metric in values[a]:
                            diff=np.array(values[a][metric])-values[b][metric]
                            ci=np.quantile(diff[ix].mean(1),[.025,.975])
                            changes[name][metric]={'mean':float(diff.mean()),'CI95':ci.tolist(),
                                'negative_groups':int((diff<0).sum()),'positive_groups':int((diff>0).sum())}
                    zz=[z for r in selected for z in r['records'] if z['sigma_r']==sr and z['S']==S]
                    groups.append({'dataset':dataset,'model':model,'sigma_r':sr,'S':S,
                        'frames':len(selected),'groups':len(clusters),'group_names':clusters,
                        'metrics':{arm:{m:float(np.mean(v)) for m,v in d.items()} for arm,d in values.items()},
                        'changes':changes,'cluster_metrics':values,
                        'max_cache_error_over_bound':max(z['cache_error_over_bound'] for z in zz),
                        'max_GPA_error_over_bound':max(z['GPA_error_over_bound'] for z in zz),
                        'GPA_N':zz[0]['GPA_N']})
    timings=json.loads((OUT/'timings.json').read_text())['records']
    costs=[]
    for sr in SR:
        for S in SIZES:
            rr=[r for r in timings if r['sigma_r']==sr and r['S']==S]
            costs.append({'sigma_r':sr,'S':S,'GPA_N':rr[0]['GPA_N'],'images':len(rr),
                'median_ms':{arm:float(np.median([np.median(r['durations_ms'][arm]) for r in rr]))
                             for arm in ('direct','uniform','GPA')},
                'median_direct_over_uniform':float(np.median([np.median(r['durations_ms']['direct'])/
                                                   np.median(r['durations_ms']['uniform']) for r in rr])),
                'median_uniform_over_GPA':float(np.median([np.median(r['durations_ms']['uniform'])/
                                                   np.median(r['durations_ms']['GPA']) for r in rr])),
                'snapshot_MiB_range':[min(r['logical_snapshot_bytes'] for r in rr)/2**20,
                                      max(r['logical_snapshot_bytes'] for r in rr)/2**20]})
    total={'frames':36,'predictions':72,'uniform_configurations':len(allrows),
           'GPA_configurations':len(allrows),'pixels_per_method':int(sum(np.prod(r['shape'])*6 for r in results)),
           'violations':sum(z['violations'] for z in allrows),
           'max_cache_error_over_bound':max(z['cache_error_over_bound'] for z in allrows),
           'max_GPA_error_over_bound':max(z['GPA_error_over_bound'] for z in allrows),
           'groups':groups,'costs':costs}
    dump(OUT/'summary.json',total)
    print('Summarized',total['uniform_configurations'],'uniform and GPA configurations;',
          total['violations'],'violations',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('test','freeze','predict','audit','time','summary'))
    args=parser.parse_args()
    globals()[{'test':'tests','time':'timing'}.get(args.stage,args.stage)]()

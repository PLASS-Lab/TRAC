"""Held-out-frame depth inference, joint-filter fidelity and matched costs (Sections 5.2-5.3).

Stages: test (oracle agreement), freeze (36 Bonn/TartanAir V2 frames and two
public checkpoints, hashed before scoring), predict (GPU FP32, needs
transformers), audit (FP64 filters and metrics), time (one-thread CPU
references), summary (grouped descriptive intervals). replay re-checks an
archived evidence directory without datasets, weights, GPU or network.
"""
from ..common import single_thread
single_thread()
import argparse
import json
import math
import os
import platform
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ..cache import bilateral, reference
from ..cache.reference import direct, gpa, gpa_order, uniform
from ..common import dump, sha, source_hashes

SR = (.2, .4)
SIZES = (8, 32, 64)
MODEL_IDS = ('da2_small', 'zoedepth')


def sources():
    import sys
    return source_hashes(sys.modules[__name__], bilateral, reference)


def tests(out):
    import torch
    from ..cache.bilateral import bilateral_exact, bilateral_two_read
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
    dump(out/'implementation_tests.json', {'cases':records,'passed':True})
    print('Passed',len(records),'explicit/archived filter oracle comparisons',flush=True)

def timestamp_rows(path):
    return [(float(s[0]),s[1]) for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')
            for s in [line.split()]]

def inputs(bonn,tartan):
    rows=[]
    bonn=Path(bonn)
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
    tartan=Path(tartan)
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

def freeze(out,bonn,tartan,models_freeze):
    assert not (out/'freeze.json').exists()
    assert json.loads((out/'implementation_tests.json').read_text())['passed']
    old=json.loads(Path(models_freeze).read_text())
    models={k:old['models'][k] for k in MODEL_IDS}
    for m in models.values():
        for p,h in m['files'].items(): assert sha(Path(m['local_path'])/p)==h,p
    frames=inputs(bonn,tartan)
    import cv2
    r=next(r for r in frames if r['dataset']=='TartanAirV2')
    _,gt=decoded(r); reference=cv2.imread(r['depth'],cv2.IMREAD_UNCHANGED).view('<f4').squeeze(-1)
    assert np.array_equal(gt,reference,equal_nan=True)
    dump(out/'freeze.json',{'status':'frozen_before_prediction_and_scoring',
        'date_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'data_role':'new_output_filter_replication_on_reused_project_datasets',
        'models':models,'frames':frames,'sigma_r':SR,'S':SIZES,
        'sources':sources(),
        'official_depth_decoder_matches_OpenCV':True,
        'timing':{'frames':[frames[0]['id'],r['id']],'warmup':2,'repeats':7,'threads':1},
        'bootstrap':{'resamples':3000,'seed':20261009}})
    print('Frozen',len(frames),'frames and',len(models),'public checkpoints before scores',flush=True)

def frozen(out):
    f=json.loads((out/'freeze.json').read_text())
    assert f['sources']==sources(),'source changed after freeze'
    return f

def predict(out):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoImageProcessor,AutoModelForDepthEstimation
    f=frozen(out); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    dest=out/'arrays';dest.mkdir(exist_ok=True)
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
    dump(out/'predictions.json',{'runtime':{'python':platform.python_version(),'numpy':np.__version__,
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

def audit(out):
    f=frozen(out);dest=out/'frame_results';dest.mkdir(exist_ok=True)
    first={dataset:next(r['id'] for r in f['frames'] if r['dataset']==dataset)
           for dataset in ('Bonn','TartanAirV2')}
    for tag in MODEL_IDS:
        for r in f['frames']:
            path=dest/f'{r["id"]}_{tag}.json'
            if path.exists(): continue
            arrays=np.load(out/'arrays'/f'{r["id"]}_inputs.npz')
            rgb,gt=arrays['rgb'],arrays['gt'].astype(np.float64)
            pred=np.load(out/'arrays'/f'{r["id"]}_{tag}.npz')['prediction'].astype(np.float64)
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
                np.savez_compressed(out/'arrays'/f'{r["id"]}_{tag}_qualitative.npz',**visuals)
            print('Audited',r['id'],r['dataset'],tag,'max ratio',
                  round(max(z['cache_error_over_bound'] for z in records),4),flush=True)
    records=[json.loads(p.read_text()) for p in sorted(dest.glob('*.json'))]
    assert len(records)==72
    dump(out/'results.json',{'runtime':{'numpy':np.__version__,'python':platform.python_version(),
                                      'platform':platform.platform(),'precision':'FP64'},'records':records})

def timing(out):
    f=frozen(out);import scipy
    timing_rows=[]
    for fid in f['timing']['frames']:
        rgb=np.load(out/'arrays'/f'{fid}_inputs.npz')['rgb']
        guide=(rgb.astype(np.float64)/255)@np.array([.299,.587,.114])
        r=next(r for r in f['frames'] if r['id']==fid)
        for tag in MODEL_IDS:
            v=np.load(out/'arrays'/f'{fid}_{tag}.npz')['prediction'].astype(np.float64)
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
                    dump(out/'timings_partial.json',timing_rows)
                    print('Timed',fid,tag,sr,S,{k:round(float(np.median(a)),1) for k,a in durations.items()},flush=True)
    dump(out/'timings.json',{'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,
        'python':platform.python_version(),'platform':platform.platform(),'thread_env':
        {k:os.environ[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        'CPU': next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                     if s.startswith('model name'))),'precision':'FP64'},'records':timing_rows})

def summary(out):
    f=frozen(out);results=json.loads((out/'results.json').read_text())['records']
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
    timings=json.loads((out/'timings.json').read_text())['records']
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
    dump(out/'summary.json',total)
    print('Summarized',total['uniform_configurations'],'uniform and GPA configurations;',
          total['violations'],'violations',flush=True)


# ----- data-free replay ---------------------------------------------------
def replay(evidence):
    """Replay scores, bounds, intervals and timings from archived arrays."""
    import torch
    from ..cache.bilateral import bilateral_exact,bilateral_two_read
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('test','freeze','predict','audit','time','summary','replay'))
    parser.add_argument('--out',type=Path,help='work directory')
    parser.add_argument('--bonn',type=Path,help='rgbd_bonn_dataset root (freeze)')
    parser.add_argument('--tartan',type=Path,help='TartanAir V2 root (freeze)')
    parser.add_argument('--models-freeze',type=Path,help='JSON with checkpoint paths and hashes (freeze)')
    parser.add_argument('--evidence',type=Path,help='archived evidence directory (replay)')
    parser.add_argument('--output',type=Path,help='replay JSON record')
    args=parser.parse_args()
    if args.stage=='replay':
        if args.evidence is None: parser.error('replay needs --evidence')
        output=replay(args.evidence)
        if args.output: dump(args.output,output)
        print(json.dumps(output,indent=2))
        return
    if args.out is None: parser.error(args.stage+' needs --out')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=='freeze':
        if None in (args.bonn,args.tartan,args.models_freeze):
            parser.error('freeze needs --bonn, --tartan and --models-freeze')
        freeze(args.out,args.bonn,args.tartan,args.models_freeze)
    else:
        {'test':tests,'predict':predict,'audit':audit,'time':timing,'summary':summary}[args.stage](args.out)


if __name__=='__main__':
    main()

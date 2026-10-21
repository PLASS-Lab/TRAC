#!/usr/bin/env python3
"""Render all predetermined V3.5 views from frozen scores, without selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scripts.build_math_v3_2_explanatory_visuals as style
from scripts.evaluate_math_v3_5 import metric_depth

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work_dirs/mdpi_math_v3_5'
PAPER=ROOT/'paper/mdpi_math_v3_5'
BLUE,ORANGE,MUTED='#176783','#BA552F','#60727D'
LABELS={'da2_small':'DA-V2 Small','zoedepth':'ZoeDepth'}

def table(name,columns,header,rows):
    (PAPER/(name+'.tex')).write_text('\\begin{tabularx}{\\textwidth}{'+columns+'}\n'
        '\\toprule\n'+header+'\\\\\n\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\n\\end{tabularx}\n')

def save(fig,name):
    for suffix in ('pdf','png'):
        fig.savefig(PAPER/'figures'/f'{name}.{suffix}',dpi=210)
    plt.close(fig)

def main():
    style.OUT=WORK/'visuals';style.figure_style()
    plt.rcParams.update({'font.size':10})
    f=json.loads((WORK/'freeze.json').read_text())
    s=json.loads((WORK/'summary.json').read_text())
    r=json.loads((WORK/'results.json').read_text())['records']
    groups=[g for g in s['groups'] if g['sigma_r']==.2 and g['S']==32]
    rows=[]
    for g in groups:
        m=g['metrics'];d=g['changes']['exact_minus_baseline']['AbsRel'];lo,hi=d['CI95']
        rows.append(f"{g['dataset'].replace('TartanAirV2','Tartan V2')} / {LABELS[g['model']]} & "
                    f"{m['baseline']['AbsRel']:.4f} & {m['exact']['AbsRel']:.4f} & {m['cached']['AbsRel']:.4f} & "
                    f"${d['mean']*1e3:+.2f}$ [{lo*1e3:+.2f}, {hi*1e3:+.2f}] & "
                    f"{g['max_cache_error_over_bound']:.3f}\\\\")
    table('fresh_summary_table','p{35mm}rrrLr',
          r'\textbf{Dataset / model} & \textbf{Base} & \textbf{Exact} & \textbf{Cache} & $10^3\Delta_e$ [95\%] & \textbf{Max. ratio}',rows)
    rows=[]
    for g in s['groups']:
        m=g['metrics'];d=g['changes']['exact_minus_baseline']['AbsRel'];c=g['changes']['cache_minus_exact']['AbsRel']
        rows.append(f"{g['dataset'].replace('TartanAirV2','Tartan V2')} & {LABELS[g['model']]} & {g['sigma_r']:g} & "
                    f"{g['S']} & {g['GPA_N']} & {m['baseline']['AbsRel']:.4f} & {m['exact']['AbsRel']:.4f} & "
                    f"{m['cached']['AbsRel']:.4f} & {m['GPA']['AbsRel']:.4f} & {d['mean']*1e3:+.2f} & "
                    f"{c['mean']*1e6:+.3f} & {g['max_cache_error_over_bound']:.3f}\\\\")
    table('fresh_full_table','l l r r r r r r r r r r',
        r'\textbf{Dataset} & \textbf{Model} & $\sigma_r$ & $S$ & $N$ & \textbf{Base} & \textbf{Exact} & \textbf{Cache} & \textbf{GPA} & $10^3\Delta_e$ & $10^6\Delta_c$ & \textbf{Ratio}',rows)
    rows=[]
    for dataset in ('Bonn','TartanAirV2'):
        rr=[x for x in f['frames'] if x['dataset']==dataset]
        for group in sorted({x['group'] for x in rr}):
            xx=[x for x in rr if x['group']==group]
            name=group.removeprefix('rgbd_bonn_').replace('_',r'\_')
            frames=[x['frame'].removesuffix('_lcam_front').replace('_',r'\_') for x in xx]
            rows.append(f"{dataset.replace('TartanAirV2','Tartan V2')} & {name} & {frames[0]} & {frames[1]}\\\\")
    table('fresh_manifest_table','l L r r',r'\textbf{Dataset} & \textbf{Group} & \textbf{First frame} & \textbf{Second frame}',rows)
    timings=json.loads((WORK/'timings.json').read_text())['records']
    rows=[]
    for x in timings:
        times={k:float(np.median(a)) for k,a in x['durations_ms'].items()}
        rows.append(f"{x['dataset'].replace('TartanAirV2','Tartan V2')} & {LABELS[x['model']]} & {x['sigma_r']:g} & "
            f"{x['S']} & {x['GPA_N']} & {times['direct']:.1f} & {times['uniform']:.1f} & {times['GPA']:.1f} & "
            f"{x['logical_snapshot_bytes']/2**20:.2f}\\\\")
    table('fresh_timing_full_table','l l r r r r r r r',
          r'\textbf{Dataset} & \textbf{Model} & $\sigma_r$ & $S$ & $N$ & \textbf{Direct} & \textbf{Uniform} & \textbf{GPA} & \textbf{Cache MiB}',rows)
    rows=[]
    for x in s['costs']:
        a=x['median_ms'];lo,hi=x['snapshot_MiB_range']
        rows.append(f"{x['sigma_r']:g} & {x['S']} & {x['GPA_N']} & {a['direct']:.0f} & {a['uniform']:.0f} & "
                    f"{a['GPA']:.0f} & {lo:.1f}--{hi:.1f}\\\\")
    table('fresh_cost_table','r r r r r r L',
          r'$\sigma_r$ & $S$ & $N$ & \textbf{Direct ms} & \textbf{Uniform ms} & \textbf{GPA ms} & \textbf{Snapshot MiB}',rows)
    # Fixed images, fixed setting: do not sort by error or task score.
    panels=[]
    for dataset in ('Bonn','TartanAirV2'):
        frame=next(x for x in f['frames'] if x['dataset']==dataset)
        for model in ('da2_small','zoedepth'):
            data=np.load(WORK/'arrays'/f'{frame["id"]}_{model}_qualitative.npz')
            record=next(x for x in r if x['frame']==frame['id'] and x['model']==model)
            R=float(np.ptp(data['baseline'])/2)
            panels.append((frame,model,data,record,abs(data['cached']-data['exact'])/R))
    vmax=max(float(x[-1].max()) for x in panels)
    fig,axes=plt.subplots(4,6,figsize=(11.9,8.1),layout='constrained')
    titles=['RGB','Reference depth','Original inference','Exact filter','Two-read, S = 32','Cache error / R']
    for i,(frame,model,data,record,error) in enumerate(panels):
        for j,key in enumerate(('rgb','gt','baseline','exact','cached','error')):
            ax=axes[i,j];ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if key=='rgb': ax.imshow(data[key])
            elif key=='error': e=ax.imshow(error,cmap='inferno',vmin=0,vmax=vmax)
            else:
                dep=data['gt'] if key=='gt' else metric_depth(data[key],record['calibration'])[0]
                dep=np.ma.masked_where(~data['valid'],dep) if key=='gt' else dep
                depth_view=ax.imshow(dep,cmap='viridis_r',vmin=0,vmax=frame['max_depth'])
            if i==0:ax.set_title(titles[j],fontsize=10)
            if j==0:
                label=frame['dataset'].replace('TartanAirV2','Tartan V2')+'\n'+LABELS[model]
                ax.set_ylabel(label,fontsize=10)
            if j==1:
                scale=ax.inset_axes([0,-.05,1,.035])
                colorbar=fig.colorbar(depth_view,cax=scale,orientation='horizontal',
                                    ticks=[0,frame['max_depth']/2,frame['max_depth']])
                colorbar.ax.tick_params(labelsize=7,pad=1,length=2)
                colorbar.set_label('Shared row depth scale (m)',fontsize=7,labelpad=1)
    cb=fig.colorbar(e,ax=axes[:,-1],shrink=.82,pad=.015)
    cb.set_label('Absolute native-value error / R',fontsize=9)
    save(fig,'fresh_depth_inference')
    # Distinguish task change from cache fidelity, and one-thread costs.
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.7),layout='constrained')
    a=axes[0];ys=np.arange(4)
    for i,g in enumerate(groups):
        d=g['changes']['exact_minus_baseline']['AbsRel'];lo,hi=d['CI95'];m=d['mean']
        a.errorbar(1e3*m,i,xerr=[[1e3*(m-lo)],[1e3*(hi-m)]],color=BLUE,marker='o',capsize=3)
        c=g['changes']['cache_minus_exact']['AbsRel']['mean']
        a.plot(1e3*c,i,'s',color=ORANGE,ms=5)
    a.axvline(0,color=MUTED,lw=.8,ls='--')
    a.set(yticks=ys,yticklabels=[g['dataset'].replace('TartanAirV2','Tartan V2')+' / '+LABELS[g['model']] for g in groups],
          xlabel=r'AbsRel change ($10^{-3}$), negative is better',title=r'(a) Fixed $\sigma_r=0.2$, $S=32$')
    a.plot([],[],'o',color=BLUE,label='Exact − original, paired 95%');a.plot([],[],'s',color=ORANGE,label='Cache − exact, mean')
    a.legend(fontsize=8,loc='best');a.grid(axis='x',alpha=.18)
    a=axes[1]
    for sr,ls in ((.2,'-'),(.4,'--')):
        costs=[c for c in s['costs'] if c['sigma_r']==sr]
        for method,color,marker in [('uniform',BLUE,'o'),('GPA',ORANGE,'s')]:
            a.plot([c['S'] for c in costs],[c['median_ms'][method] for c in costs],color=color,ls=ls,
                   marker=marker,label=f'{"Uniform" if method=="uniform" else "GPA"}, sigma = {sr}')
    a.set(xlabel='Matched uniform cache size S',ylabel='Complete image time (ms)',
          title='(b) Four fixed value images, one CPU thread',xticks=[8,32,64])
    a.grid(alpha=.18);a.legend(fontsize=8)
    save(fig,'fresh_accuracy_cost')
    print('Rendered two fixed-result figures and five reproducible tables.')

if __name__=='__main__':main()

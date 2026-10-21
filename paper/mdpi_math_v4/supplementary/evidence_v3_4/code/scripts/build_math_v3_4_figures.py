#!/usr/bin/env python3
"""Render frozen V3.4 results and tables; never rescore or select data."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import scripts.build_math_v3_2_explanatory_visuals as style

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work_dirs/mdpi_math_v3_4'
PAPER=ROOT/'paper/mdpi_math_v3_4'
BLUE,ORANGE,MUTED='#176783','#BA552F','#60727D'

def save(fig,name):
    for ext in ['pdf','png']: fig.savefig(PAPER/'figures'/f'{name}.{ext}',dpi=180)
    plt.close(fig)

def main():
    style.OUT=WORK/'visuals';style.figure_style()
    plt.rcParams.update({'font.size':10})
    s=json.loads((WORK/'summary.json').read_text())
    groups=[r for r in s['groups'] if r['tolerance']==.001]
    x=np.arange(4); labels=[str(r['sigma_r']) for r in groups]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.25),layout='constrained')
    a=axes[0];y=np.array([r['image_mean_ratio_median'] for r in groups]);ci=np.array([r['image_bootstrap95'] for r in groups])
    a.errorbar(x,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),marker='o',capsize=4,color=BLUE)
    a.axhline(1,color=MUTED,ls='--',lw=1)
    a.set(xticks=x,xticklabels=labels,xlabel=r'Range bandwidth $\sigma_r$',ylabel='Local uniform / adaptive cells',title=r'(a) Image-level ratios, tolerance $10^{-3}$',ylim=(.8,5))
    a.grid(alpha=.2)
    a=axes[1]; counts=[r['adaptive_better_same_S'] for r in groups]
    a.bar(x,np.array(counts)/384,color=BLUE,width=.55)
    for i,c in enumerate(counts):a.text(i,c/384+.025,f'{c}/384',ha='center',fontsize=9)
    a.set(xticks=x,xticklabels=labels,xlabel=r'Range bandwidth $\sigma_r$',ylabel='Fraction with smaller sampled error',title='(b) Adaptive wins at equal cache size',ylim=(0,1.03));a.grid(axis='y',alpha=.2)
    save(fig,'kodak_cache_validation')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.3),layout='constrained')
    a=axes[0];costs=s['costs']; node=[costs[k]['selection_ms_median'] for k in ['adaptive','uniform']];snap=[costs[k]['snapshot_ms_median'] for k in ['adaptive','uniform']]
    a.bar([0,1],node,color=MUTED,label='Node / certificate selection');a.bar([0,1],snap,bottom=node,color=BLUE,label='Snapshot evaluation')
    a.set(xticks=[0,1],xticklabels=['Adaptive','Local uniform'],ylabel='Construction time (ms)',title='(a) First probe in each of 24 images');a.legend(fontsize=8);a.grid(axis='y',alpha=.2)
    a=axes[1];q=np.array([r['Q'] for r in s['lifetime']])
    for arm,color,marker in [('adaptive',BLUE,'o'),('uniform',ORANGE,'^')]:
        y=np.array([r[arm+'_speedup'] for r in s['lifetime']]); raw=np.array([[v[arm+'_speedup'] for v in r['per_image']] for r in s['lifetime']]);lo,hi=np.quantile(raw,[.25,.75],axis=1)
        a.plot(q,y,color=color,marker=marker,label=arm.capitalize());a.fill_between(q,lo,hi,color=color,alpha=.15)
    a.axhline(1,color=MUTED,ls='--',lw=1,label='Break-even')
    a.set(xscale='log',yscale='log',xlabel='Queries sharing one cache (Q)',ylabel='Direct time / complete cache time',title='(b) Construction + metadata + queries');a.set_xticks(q,[str(z) for z in q]);a.xaxis.set_minor_formatter(NullFormatter());a.grid(alpha=.2);a.legend(fontsize=8,loc='upper left')
    save(fig,'cache_lifetime_cost')
    rows=[]
    for r in groups:
        lo,hi=r['image_bootstrap95']
        rows.append(f"{r['sigma_r']:g} & {r['adaptive_S_median']:g} & {r['local_uniform_S_median']:g} & {r['image_mean_ratio_median']:.2f} [{lo:.2f}, {hi:.2f}] & {r['adaptive_better_same_S']}/384 & {r['max_error_over_tolerance']:.3f}\\\\")
    (PAPER/'kodak_table_rows.tex').write_text('\n'.join(rows)+'\n')
    rows=[]
    for r in s['groups']:
        lo,hi=r['image_bootstrap95']
        rows.append(f"{r['sigma_r']:g} & ${r['tolerance']:g}$ & {r['adaptive_S_median']:g} & {r['local_uniform_S_median']:g} & {r['image_mean_ratio_median']:.2f} [{lo:.2f}, {hi:.2f}] & {r['adaptive_loses_storage']} & {r['adaptive_better_same_S']} & {r['max_error_over_tolerance']:.3f}\\\\")
    (PAPER/'kodak_full_table_rows.tex').write_text('\n'.join(rows)+'\n')
    rows=[f"{r['Q']} & {r['direct_ms']:.4f} & {r['adaptive_total_ms']:.4f} & {r['uniform_total_ms']:.4f} & {r['adaptive_faster_images']}/24 & {r['uniform_faster_images']}/24\\\\" for r in s['lifetime']]
    (PAPER/'lifetime_table_rows.tex').write_text('\n'.join(rows)+'\n')
    for name,columns,header in [
        ('kodak','r r r L r r',r'$\sigma_r$ & $S_a$ & $S_l$ & \textbf{Ratio [95\% interval]} & \textbf{Wins} & \textbf{Max. $e/\epsilon$}'),
        ('kodak_full','rrrrLrrr',r'$\sigma_r$ & $\epsilon$ & $S_a$ & $S_l$ & \textbf{Ratio [95\% interval]} & \textbf{More} & \textbf{Wins} & \textbf{Max.}'),
        ('lifetime','rLLLrr',r'$Q$ & \textbf{Direct} & \textbf{Adaptive total} & \textbf{Uniform total} & \textbf{Adaptive faster} & \textbf{Uniform faster}')]:
        body=(PAPER/f'{name}_table_rows.tex').read_text()
        (PAPER/f'{name}_table.tex').write_text(r'\begin{tabularx}{\textwidth}{'+columns+'}\n'+r'\toprule'+'\n'+header+r'\\'+'\n'+r'\midrule'+'\n'+body+r'\bottomrule'+'\n'+r'\end{tabularx}'+'\n')
    print('Two vector figures and three result tables generated from frozen JSON.')

if __name__=='__main__':main()

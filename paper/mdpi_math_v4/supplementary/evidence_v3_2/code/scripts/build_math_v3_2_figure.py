#!/usr/bin/env python3
"""Plot frozen v3.2 budgets and the first predeclared neighborhood."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import torch

from sokkanaem.certified_scalar_cache import interpolate, rows
from scripts.build_math_v3_2_explanatory_visuals import figure_style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dirs/mdpi_math_v3_2'
PAPER = ROOT / 'paper/mdpi_math_v3_2'


def main():
    torch.set_num_threads(4)
    summary = json.loads((OUT/'summary.json').read_text())
    data = [r for r in summary['groups'] if r['tolerance']==0.001]
    figure_style()
    plt.rcParams.update({'font.size':10})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.35), layout='constrained')
    ax = axes[0]
    x = [r['sigma_r'] for r in data]
    for field, label, colour, marker in (
            ('global_uniform_S','Global-span uniform','#60727D','s'),
            ('local_uniform_S_median','Local-span uniform, median','#BA552F','^'),
            ('adaptive_S_median','Variance adaptive, median','#176783','o')):
        ax.plot(x, [r[field] for r in data], label=label, color=colour, marker=marker)
    ax.set(xscale='log', yscale='log', xlabel=r'Range bandwidth $\sigma_r$',
           ylabel='Sufficient stored cells', title=r'(a) Normalized tolerance $10^{-3}$')
    ax.set_xticks(x, [str(z) for z in x]); ax.grid(alpha=.2); ax.legend(fontsize=8)
    ax.xaxis.set_minor_formatter(NullFormatter())
    raw = np.load(OUT/'sampled_families.npz')
    sr, tolerance = 0.05, 0.001
    key = torch.from_numpy(raw['keys_0_0'])
    nodes = torch.from_numpy(raw[f'nodes_{sr}_{tolerance}'])
    bound = torch.from_numpy(raw[f'bounds_{sr}_{tolerance}'])
    offset = torch.arange(-6,7,dtype=torch.float64)
    bias = -(offset[:,None].square()+offset[None,:].square()).flatten()/8-key.square()/(2*sr**2)
    keys = key/sr**2
    q = torch.linspace(0,1,257,dtype=torch.float64)
    exact = rows(q, keys, bias)
    uniform = torch.linspace(0,1,len(nodes),dtype=torch.float64)
    eadaptive = (exact-interpolate(q,nodes,rows(nodes,keys,bias))).abs().sum(-1)
    euniform = (exact-interpolate(q,uniform,rows(uniform,keys,bias))).abs().sum(-1)
    cell = (torch.searchsorted(nodes,q,right=True)-1).clamp(0,len(nodes)-2)
    ax = axes[1]
    ax.semilogy(q, np.maximum(euniform.numpy(),1e-12), color='#BA552F', label='Uniform, same S')
    ax.semilogy(q, np.maximum(eadaptive.numpy(),1e-12), color='#176783', label='Variance adaptive')
    ax.semilogy(q, np.maximum(bound[cell].numpy(),1e-12), color='#60727D', linestyle=':', label='Cell envelope')
    ax.axhline(tolerance,color='#203341',linestyle='--',linewidth=1,label='Declared tolerance')
    ax.set(xlabel='Independent guide query g',ylabel='Normalized worst-value error',
           ylim=(1e-8,1),title=f'(b) First frozen neighborhood, S={len(nodes)}')
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
    for suffix in ('pdf','png'):
        fig.savefig(PAPER/'figures'/f'variance_adaptive.{suffix}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()

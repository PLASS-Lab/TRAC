#!/usr/bin/env python3
"""Build explanatory vector figures from v3.2 mathematics and frozen records.

No inference, new scoring, data selection or changes to the frozen experiment.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from sokkanaem.certified_scalar_cache import interval_envelope, rows


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/mdpi_math_v3_2'
WORK = ROOT / 'work_dirs/mdpi_math_v3_2'
OUT = WORK / 'visual_revision'
BLUE, ORANGE, INK, MUTED = '#176783', '#BA552F', '#203341', '#60727D'
RULE, PALE_BLUE, PALE_ORANGE = '#CBD7DD', '#EBF4F7', '#FBF0EA'
COLOURS = [BLUE, ORANGE, '#417C55', '#735B91']


def figure_style():
    """Use the existing Pagella policy with valid quadratic TrueType outlines.

    Matplotlib's direct Type42 export of the local CFF OTF produces a font
    mismatch warning. Convert outlines locally, preserving cmap and metrics;
    use a derivative family name. No standalone font is packaged or installed.
    """
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.cu2quPen import Cu2QuPen
    fonts = Path('/usr/share/texmf/fonts/opentype/public/tex-gyre')
    destination = OUT/'local_fonts'
    destination.mkdir(parents=True,exist_ok=True)
    for style in ('regular','italic','bold','bolditalic'):
        target = destination/f'pagella-figure-{style}.ttf'
        original = TTFont(fonts/f'texgyrepagella-{style}.otf')
        if not target.exists():
            font = TTFont(fonts/f'texgyrepagella-{style}.otf')
            glyphset = font.getGlyphSet(); glyphs = {}
            for name in font.getGlyphOrder():
                pen = TTGlyphPen(None)
                glyphset[name].draw(Cu2QuPen(pen,max_err=1.0,reverse_direction=True))
                glyphs[name] = pen.glyph()
            del font['CFF ']
            font.sfntVersion='\x00\x01\x00\x00'
            font['glyf']=newTable('glyf');font['glyf'].glyphs=glyphs
            font['loca']=newTable('loca')
            font['maxp']=newTable('maxp');font['maxp'].tableVersion=0x00010000
            font['maxp'].numGlyphs=len(glyphs)
            for name in ('maxPoints','maxContours','maxCompositePoints','maxCompositeContours',
                         'maxZones','maxTwilightPoints','maxStorage','maxFunctionDefs',
                         'maxInstructionDefs','maxStackElements','maxSizeOfInstructions',
                         'maxComponentElements','maxComponentDepth'):
                setattr(font['maxp'],name,1 if name=='maxZones' else 0)
            font['post'].formatType=2.0;font['post'].extraNames=[];font['post'].mapping={}
            for record in font['name'].names:
                if record.nameID in (1,16): value='Pagella Figure'
                elif record.nameID in (3,4): value='Pagella Figure '+style
                elif record.nameID==6: value='PagellaFigure-'+style
                else: continue
                record.string=value.encode(record.getEncoding())
            font.save(target)
        converted=TTFont(target)
        assert original.getBestCmap()==converted.getBestCmap()
        assert original['hmtx'].metrics==converted['hmtx'].metrics
        font_manager.fontManager.addfont(str(target))
    plt.rcParams.update({'font.family':'Pagella Figure','font.serif':['Pagella Figure'],
        'font.size':9,'mathtext.fontset':'custom','mathtext.rm':'Pagella Figure',
        'mathtext.it':'Pagella Figure:italic','mathtext.bf':'Pagella Figure:bold',
        'mathtext.bfit':'Pagella Figure:bold:italic','mathtext.sf':'Pagella Figure',
        'mathtext.tt':'Pagella Figure','mathtext.cal':'Pagella Figure:italic',
        'mathtext.fallback':'stix','axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
        'pdf.fonttype':42,'ps.fonttype':42})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card(ax, x, y, width, height, text, colour=PALE_BLUE, fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle='round,pad=0.008,rounding_size=0.012',
                 edgecolor=RULE, facecolor=colour, linewidth=.8,
                 transform=ax.transAxes))
    ax.text(x+width/2, y+height/2, text, ha='center', va='center',
            fontsize=fontsize, color=INK, transform=ax.transAxes)


def save(fig, name):
    paths = []
    for suffix in ('pdf', 'png'):
        path = PAPER / 'figures' / f'{name}.{suffix}'
        fig.savefig(path, dpi=200, facecolor='white')
        paths.append(path)
    plt.close(fig)
    return {str(p.relative_to(PAPER)): sha(p) for p in paths}


def shared_atoms():
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.45))
    fig.subplots_adjust(left=.02, right=.98, top=.89, bottom=.24, wspace=.15)
    for ax in axes:
        ax.set(xlim=(0, 3.35), ylim=(-.12, 2.4))
        ax.axis('off')
    left, right = axes
    left.set_title('(a) Shared dictionary', fontweight='bold', fontsize=10.5)
    right.set_title('(b) Independent plane relaxation', fontweight='bold', fontsize=10.5)
    vertices = np.array([[.7, 1.7], [2.65, 1.7], [.7, .25], [2.65, .25]])
    for i in range(4):
        for j in range(i+1, 4):
            colour = ORANGE if (i,j)==(0,1) else BLUE if (i,j)==(0,2) else RULE
            left.plot(*vertices[[i,j]].T, color=colour, lw=2 if colour!=RULE else 1.2, zorder=1)
    for i, (x, y) in enumerate(vertices):
        left.add_patch(Circle((x,y), .17, facecolor=PALE_BLUE,
                             edgecolor=INK if i==0 else BLUE, lw=1.2, zorder=2))
        left.text(x,y,f'$A_{i+1}$',ha='center',va='center',fontsize=10,zorder=3)
    left.text(1.675,2.15,r'$S=4$ atoms; $\binom{4}{2}=6$ pair spans',ha='center',fontsize=9.4)
    left.text(1.675,.98,r'$A_1$ belongs to several pairs',ha='center',va='center',
              fontsize=8.8,bbox={'facecolor':'white','edgecolor':'none','pad':3})
    for n in range(6):
        col, row = n % 2, n // 2
        x, y = .43+1.72*col, 1.75-.7*row
        right.plot([x,x+.72],[y,y],color=BLUE,lw=1.7,zorder=1)
        for xx, name in ((x,'u'),(x+.72,'v')):
            right.add_patch(Circle((xx,y),.105,facecolor=PALE_BLUE,edgecolor=BLUE,lw=.9,zorder=2))
            right.text(xx,y-.2,f'${name}_{n+1}$',ha='center',va='top',fontsize=8.2)
        right.text(x+.36,y+.14,f'$P_{n+1}$',ha='center',va='bottom',fontsize=9)
    right.text(1.675,2.15,r'$M=6$ independently chosen spans',ha='center',fontsize=9.4)
    fig.text(.255,.16,'Every pair uses the same S atoms',ha='center',fontsize=9.2,color=INK)
    fig.text(.255,.075,r'$S^{-2}$ decay, up to a logarithm',ha='center',fontsize=10,color=BLUE)
    fig.text(.755,.16,'Each plane chooses its own generating pair',ha='center',fontsize=9.2,color=INK)
    fig.text(.755,.075,r'$\Theta(S^{-4})$ at $M=\binom{S}{2}$',
             ha='center',fontsize=10,color=ORANGE)
    return save(fig,'shared_atom_structure')


def adaptive_construction(raw):
    sr, tolerance = .05, .001
    guides = torch.from_numpy(raw['keys_0_0'])
    nodes = torch.from_numpy(raw[f'nodes_{sr}_{tolerance}'])
    key = guides/sr**2
    offset = torch.arange(-6,7,dtype=torch.float64)
    bias = -(offset[:,None].square()+offset[None,:].square()).flatten()/8-guides.square()/(2*sr**2)
    query = torch.linspace(0,1,1025,dtype=torch.float64)
    probability = rows(query,key,bias)
    mean = (probability*key).sum(-1,keepdim=True)
    variance = (probability*(key-mean).square()).sum(-1)
    span = float(key.max()-key.min())
    bounds,_ = interval_envelope(key,bias,nodes[:-1],nodes[1:])
    archived = torch.from_numpy(raw[f'bounds_{sr}_{tolerance}'])
    assert torch.allclose(bounds,archived,atol=1e-15,rtol=1e-12)
    assert float(bounds.max())<=tolerance and len(nodes)==212
    fig = plt.figure(figsize=(7,4.3))
    flow = fig.add_axes([.012,.78,.976,.21]);flow.axis('off')
    texts = ['Fix metadata\n'+r'$\kappa,b,[l,r],\varepsilon$',
             'Choose nodes\n'+r'bisect if $B(I)>\varepsilon$',
             'Store readouts\n'+r'$z_s=a(t_s)^\top v$',
             'Answer a query\n'+r'read $z_i,z_{i+1}$']
    for i,text in enumerate(texts):
        card(flow,.008+.255*i,.24,.218,.64,text,
             colour=PALE_ORANGE if i==2 else PALE_BLUE,fontsize=9.2)
        if i<3:
            flow.add_patch(FancyArrowPatch((.234+.255*i,.56),(.254+.255*i,.56),
                           arrowstyle='-|>',mutation_scale=11,color=MUTED,transform=flow.transAxes))
    flow.text(.24,.045,'Node selection uses keys and biases',ha='center',fontsize=8.4,color=BLUE)
    flow.text(.627,.045,'Values enter here',ha='center',fontsize=8.4,color=ORANGE)
    ax = fig.add_axes([.09,.36,.365,.34])
    ax.semilogy(query.numpy(),np.maximum(variance.numpy()/(span**2/4),1e-12),color=BLUE,lw=1.7)
    ax.axhline(1,color=MUTED,ls=':',lw=1)
    ax.set(xlim=(0,1),ylim=(1e-7,2),xlabel='Guide query g',ylabel=r'$\mathrm{Var}_{a(g)}(\kappa)/(K^2/4)$')
    ax.set_title('(a) Conditional mass changes',fontsize=10,fontweight='bold')
    ax.grid(alpha=.18)
    ax = fig.add_axes([.595,.36,.365,.34])
    gaps = torch.diff(nodes).numpy()
    ax.stairs(gaps,nodes.numpy(),color=BLUE,lw=1.5,label='Adaptive cells')
    equal_width = 1/(len(nodes)-1)
    sufficient_width = 4*np.sqrt(tolerance)/span
    ax.axhline(equal_width,color=ORANGE,ls='--',lw=1.2,label='Uniform, same S')
    ax.axhline(sufficient_width,color=MUTED,ls=':',lw=1.2,label='Range-only sufficient')
    ax.set(yscale='log',xlim=(0,1),ylim=(sufficient_width*.6,.2),xlabel='Guide query g',ylabel=r'Cell width $h_g$')
    ax.set_title('(b) Envelope chooses cell widths',fontsize=10,fontweight='bold')
    ax.grid(alpha=.18);ax.legend(fontsize=7.2,loc='upper left',framealpha=.95)
    raster = fig.add_axes([.235,.095,.725,.11])
    uniform = np.linspace(0,1,len(nodes))
    raster.plot(uniform,np.zeros_like(uniform),'|',color=ORANGE,ms=11,mew=.65)
    raster.plot(nodes.numpy(),np.ones(len(nodes)),'|',color=BLUE,ms=11,mew=.65)
    raster.set(xlim=(0,1),ylim=(-.55,1.55),xlabel='Stored guide levels',yticks=[0,1],
               yticklabels=[f'Uniform ({len(nodes)})',f'Adaptive ({len(nodes)})'])
    raster.tick_params(axis='y',length=0,pad=7,labelsize=8.5)
    for side in ('left','right','top'):
        raster.spines[side].set_visible(False)
    return save(fig,'adaptive_construction'),{
        'image_no':0,'probe':0,'sigma_r':sr,'tolerance':tolerance,'S':len(nodes),
        'min_cell_width':float(gaps.min()),'max_cell_width':float(gaps.max()),
        'range_sufficient_width':sufficient_width,'archived_bounds_reproduced':True}


def range_embedding(record):
    sr = record['sigma_r']
    key = torch.tensor(record['keys'],dtype=torch.float64)
    weights = torch.tensor(record['positive_weights'],dtype=torch.float64)
    g = torch.linspace(0,1,1025,dtype=torch.float64)
    x = 2**(2*g-1)
    gaussian = (weights.log()-(g[:,None]-key).square()/(2*sr**2)).softmax(-1)
    polynomial = torch.stack([x**j for j in range(4)],-1)
    polynomial /= polynomial.sum(-1,keepdim=True)
    gap = float((gaussian-polynomial).abs().max())
    assert gap<1e-14 and abs(gap-record['grid_max_abs'])<1e-15
    fig = plt.figure(figsize=(7,3.6))
    fig.text(.5,.955,r'$\sigma_r=0.4,\quad \kappa_j=2\sigma_r^2j\log 2,\quad w_j=\exp[(\kappa_j^2-\kappa_j)/(2\sigma_r^2)]$',
             ha='center',va='top',fontsize=10.2,color=INK)
    ax = fig.add_axes([.085,.335,.355,.48])
    ax.plot(g.numpy(),x.numpy(),color=BLUE,lw=1.8)
    sample=np.array([0,.5,1]);target=2**(2*sample-1)
    ax.scatter(sample,target,color=BLUE,s=24,zorder=3)
    ax.set(xlim=(-.02,1.04),ylim=(.42,2.12),xlabel='Independent guide query g',
           ylabel='Four-key query x',xticks=[0,.5,1],yticks=[.5,1,2])
    ax.text(.08,1.8,r'$x=2^{2g-1}$',fontsize=12,color=BLUE)
    ax.set_title('(a) A bijective query map',fontsize=10,fontweight='bold')
    ax.grid(alpha=.18)
    ax = fig.add_axes([.575,.335,.36,.48])
    selected=np.arange(0,len(g),128)
    for j,colour in enumerate(COLOURS):
        ax.plot(g.numpy(),gaussian[:,j].numpy(),color=colour,lw=1.6)
        ax.plot(g[selected].numpy(),polynomial[selected,j].numpy(),'o',mfc='white',mec=colour,ms=3.5,mew=.8)
        ax.text(1.035,float(gaussian[-1,j]),f'$a_{j}$',color=colour,va='center',fontsize=9)
    ax.set(xlim=(0,1.16),ylim=(0,.59),xlabel='Independent guide query g',ylabel='Normalized row weights',xticks=[0,.5,1])
    ax.set_title('(b) The rows coincide',fontsize=10,fontweight='bold')
    handles=[Line2D([0],[0],color=MUTED,lw=1.6,label='Weighted Gaussian'),
             Line2D([0],[0],color=MUTED,marker='o',mfc='white',ls='none',ms=3.5,label='Four-key row')]
    ax.legend(handles=handles,fontsize=7.7,loc='upper center',bbox_to_anchor=(.5,1.01),framealpha=.95)
    ax.grid(alpha=.18)
    footer=fig.add_axes([.02,.015,.96,.20]);footer.axis('off')
    card(footer,.012,.08,.476,.82,
         'Positive sample weights; independent values / queries\nExact hard-family embedding',fontsize=8.8)
    card(footer,.514,.08,.474,.82,
         'Fixed Gaussian kernel on a pixel lattice\nUpper guarantee; matching lower bound open',colour=PALE_ORANGE,fontsize=8.8)
    return save(fig,'range_embedding_identity'),{'sigma_r':sr,'query_count':len(g),'max_row_difference':gap}


def normalization_stability(records):
    order=sorted(records,key=lambda r:(r['layout']!='constant',r['dtype']))
    labels=[('Constant guide' if r['layout']=='constant' else 'Two guide regions')+
            (' / FP32' if r['dtype'].endswith('32') else ' / FP64') for r in order]
    old=np.array([r['legacy_constant_error'] for r in order])
    stable=np.array([r['stable_constant_error'] for r in order])
    fig=plt.figure(figsize=(7,2.85))
    ax=fig.add_axes([.025,.18,.43,.73]);ax.axis('off')
    ax.set_title('(a) Row mass controls constants',fontsize=10,fontweight='bold',pad=6)
    card(ax,.025,.55,.95,.31,'Archived denominator floor\n'+r'$\sum_j\widetilde a_j=D/\max(D,\tau)$',
         colour=PALE_ORANGE,fontsize=10)
    card(ax,.025,.09,.95,.31,'Normalized ratio\n'+r'$\sum_j a_j=D/D=1$',fontsize=10)
    ax.text(.5,.455,r'$D<\tau$ reduces the row mass',ha='center',color=ORANGE,fontsize=9,transform=ax.transAxes)
    ax.text(.5,-.02,'Common weight rescaling cancels in the ratio',ha='center',fontsize=8.3,color=MUTED,transform=ax.transAxes)
    ax=fig.add_axes([.72,.23,.25,.64])
    y=np.arange(len(order))
    ax.barh(y-.15,old,height=.27,color=ORANGE,label='Archived floor')
    ax.barh(y+.15,stable,height=.27,color=BLUE,label='Stable ratio')
    for shift,values,colour in ((-.15,old,ORANGE),(.15,stable,BLUE)):
        for yy,value in zip(y+shift,values):
            if value==0:
                ax.plot(0,yy,'o',mfc='white',mec=colour,ms=3.8,clip_on=False)
            ax.text(value+.025,yy,'0' if value==0 else f'{value:.0f}',va='center',fontsize=8,color=colour)
    ax.set(xlim=(0,1.15),xticks=[0,.5,1],yticks=y,yticklabels=labels,
           xlabel='Maximum constant error')
    ax.invert_yaxis();ax.tick_params(axis='y',length=0,labelsize=8)
    ax.set_title('(b) Archived regressions',fontsize=10,fontweight='bold',pad=6)
    ax.legend(loc='lower right',bbox_to_anchor=(1.05,-.36),fontsize=8,ncol=2,frameon=False)
    ax.grid(axis='x',alpha=.18)
    return save(fig,'normalization_stability'),{'plotted_fields':['legacy_constant_error','stable_constant_error'],
             'records':len(order),'zero_errors_drawn_as_zero_markers':True}


def main():
    torch.set_num_threads(4)
    figure_style()
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(WORK/'results.json')==protocol['results_sha256']
    assert sha(WORK/'sampled_families.npz')==protocol['families_sha256']
    data=json.loads((WORK/'results.json').read_text())
    raw=np.load(WORK/'sampled_families.npz')
    files=shared_atoms()
    artifacts,adaptive=adaptive_construction(raw);files.update(artifacts)
    artifacts,embedding=range_embedding(data['embedding']);files.update(artifacts)
    artifacts,stability=normalization_stability(data['stability']);files.update(artifacts)
    manifest={'scope':'explanatory_diagrams_and_replots_of_frozen_records; no_new_experiment',
              'font_embedding':'local Pagella quadratic TrueType; cmap and metrics preserved; STIX serif fallback',
              'font_sources':{p.name:sha(p) for p in Path('/usr/share/texmf/fonts/opentype/public/tex-gyre').glob('texgyrepagella-*.otf')},
              'source_script_sha256':sha(Path(__file__)),
              'results_sha256':sha(WORK/'results.json'),'families_sha256':sha(WORK/'sampled_families.npz'),
              'figures':files,'adaptive':adaptive,'embedding':embedding,'stability':stability}
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()

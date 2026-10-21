#!/usr/bin/env python3
"""Verify V3.4 source preservation, every stored cache and published artifacts."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from scripts.evaluate_math_v3_4 import nrows, ninterp, nenvelope, metadata, nnodes

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper/mdpi_math_v3_4'; BASE=ROOT/'paper/mdpi_math_v3_3'; WORK=ROOT/'work_dirs/mdpi_math_v3_4'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(p):
    return re.sub(r'\\input\{([^}]+)\}',lambda m:resolve(p.parent/(m[1] if m[1].endswith('.tex') else m[1]+'.tex')),p.read_text())

def main():
    checks={}
    def check(name,ok): checks[name]=bool(ok)
    baseline=json.loads((WORK/'v3_3_baseline_sha256.json').read_text())
    check('V3_3_all_1230_files_unchanged',len(baseline)==1230 and
          set(baseline)=={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()} and
          all(sha(BASE/p)==h for p,h in baseline.items()))
    protected={p:h for p,h in baseline.items() if p.startswith(('figures/','supplementary/'))}
    protected.pop('supplementary/SHA256SUMS',None)
    check('all_inherited_figure_and_evidence_assets_unchanged',all(sha(PAPER/p)==h for p,h in protected.items()))
    a,b=resolve(BASE/'manuscript.tex'),resolve(PAPER/'manuscript.tex')
    mathpattern=r'\\begin\{(Theorem|Proposition|Lemma|Corollary|proof)\}.*?\\end\{\1\}'
    ma=[m.group(0) for m in re.finditer(mathpattern,a,re.S)];mb=[m.group(0) for m in re.finditer(mathpattern,b,re.S)]
    changed=[i for i,(x,y) in enumerate(zip(ma,mb)) if x!=y]
    check('43_math_blocks_only_two_endpoint_proofs_clarified',len(ma)==len(mb)==43 and len(changed)==2 and
          all(ma[i].startswith('\\begin{proof}') for i in changed) and
          'finite improper' in mb[changed[0]] and '\\overline I' in mb[changed[1]])
    equations=r'\\begin\{(equation|align)\}.*?\\end\{\1\}'
    check('all_original_numbered_equations_byte_identical',
          [m.group(0) for m in re.finditer(equations,a,re.S)]==[m.group(0) for m in re.finditer(equations,b,re.S)])
    eq=set(re.findall(r'\\label\{(eq:[^}]+)\}',b))
    check('all_30_equations_referenced',len(eq)==30 and all(re.search(r'\\(?:eqref|ref)\{'+re.escape(e)+r'\}',b) for e in eq))
    check('original_multiscale_constant_proof_byte_identical',sha(PAPER/'multiscale_circle_proof.tex')==sha(BASE/'multiscale_circle_proof.tex'))
    supp=resolve(PAPER/'supplementary.tex')
    check('main_12_figures_supplement_9_figures',b.count('\\begin{figure}')==12 and supp.count('\\begin{figure}')==9)
    check('actual_network_inference_figure_retained_once',b.count('figures/qualitative_depth_replay.pdf')==1 and 'figures/qualitative_depth_replay.pdf' not in supp)
    check('new_audit_and_kodak_in_main_and_supplement',all(s in b for s in ['sec:proof-audit','sec:kodak','tab:related-resources','fig:lifetime']) and all(s in supp for s in ['S:audit-v34','S:kodak-v34']))
    frozen=json.loads((WORK/'freeze.json').read_text()); result=json.loads((WORK/'results.json').read_text()); summary=json.loads((WORK/'summary.json').read_text())
    check('experiment_sources_match_prescore_freeze',all(sha(ROOT/p)==h for p,h in frozen['sources'].items()))
    check('24_complete_images_pinned_mirror',len(frozen['images'])==24 and [i['index'] for i in frozen['images']]==list(range(1,25)) and len(frozen['commit'])==40)
    check('record_counts_and_frozen_inputs',len(result['records'])==3072 and len(result['timings'])==24 and sha(WORK/'freeze.json')==result['freeze_sha256'] and sha(WORK/'families.npz')==result['families_sha256'])
    arrays=np.load(WORK/'families.npz');q=np.linspace(0,1,1025);off=np.arange(-6,7);logsp=-(off[:,None]**2+off[None,:]**2).ravel()/8
    replay_failures=[];max_difference=0.
    for r in result['records']:
        prefix=f'i{r["image"]:02d}_p{r["probe"]:02d}';suffix=prefix+f'_s{r["sigma_r"]:g}_e{r["tolerance"]:g}'
        g=arrays[prefix+'_guides'];v=arrays[prefix+'_values'];k=g/r['sigma_r']**2;b0=logsp-g*g/(2*r['sigma_r']**2)
        nodes=arrays[suffix+'_nodes'];env=arrays[suffix+'_bounds'];p=nrows(q,k,b0)
        err=np.sum(abs(p-ninterp(q,nodes,nrows(nodes,k,b0))),axis=-1);lo,_=metadata(q,nodes)
        uniform=np.linspace(0,1,len(nodes));ue=np.sum(abs(p-ninterp(q,uniform,nrows(uniform,k,b0),True)),axis=-1)
        actual=np.max(abs((p-ninterp(q,nodes,nrows(nodes,k,b0)))@(v-(v.max()+v.min())/2)))
        ok=(len(nodes)==r['adaptive_S'] and nodes[0]==0 and nodes[-1]==1 and np.all(np.diff(nodes)>0) and
            np.all(err<=env[lo]+2e-12) and np.max(env)<=r['tolerance'] and
            np.max(abs(nenvelope(k,b0,nodes[:-1],nodes[1:])-env))<2e-12 and
            abs(actual-r['actual_red_error'])<2e-12 and abs(ue.max()-r['same_S_uniform_error'])<2e-12)
        max_difference=max(max_difference,abs(err.max()-r['row_l1_error']))
        if not ok: replay_failures.append(suffix)
    check('all_3072_caches_and_3148800_queries_replayed',not replay_failures and max_difference<2e-12)
    # Reconstruct predeclared pseudorandom query batches and adaptive reference
    # grids; check raw repetitions and total-cost calculations independently.
    rng=np.random.default_rng(20261009); timing_ok=True
    for t in result['timings']:
        prefix=f'i{t["image"]:02d}_p00';g=arrays[prefix+'_guides'];k=g/.1**2;b0=logsp-g*g/(2*.1**2)
        nodes,_=nnodes(k,b0,.001)
        timing_ok &= np.array_equal(nodes,arrays[prefix+'_s0.1_e0.001_nodes'])
        for batch in t['batches']:
            query=rng.uniform(0,1,batch['Q'])
            timing_ok &= hashlib.sha256(query.tobytes()).hexdigest()==batch['queries_sha256']
            timing_ok &= all(len(times)==15 and min(times)>0 for times in [*batch['query_ms'].values(),*batch['metadata_ms'].values()])
    for group in summary['lifetime']:
        for row in group['per_image']:
            t=next(t for t in result['timings'] if t['image']==row['image']); batch=next(x for x in t['batches'] if x['Q']==group['Q'])
            for arm in ['adaptive','uniform']:
                total=sum(np.median(t['arms'][arm][field]) for field in ['node_selection_ms','snapshot_ms'])+np.median(batch['query_ms'][arm])
                timing_ok &= abs(total-row[arm+'_total_ms'])<1e-12
                timing_ok &= abs(np.median(batch['query_ms']['direct'])/total-row[arm+'_speedup'])<1e-12
    check('timing_grids_query_hashes_15_repeats_and_complete_totals',timing_ok)
    check('zero_violations_and_retained_unfavorable_cases',summary['violations']==0 and sum(g['adaptive_loses_storage'] for g in summary['groups'])==8)
    check('result_tables_match_frozen_summary',all((PAPER/(name+'_table_rows.tex')).read_text().strip() in (PAPER/(name+'_table.tex')).read_text() for name in ['kodak','kodak_full','lifetime']))
    audit=json.loads((WORK/'proof_audit.json').read_text());geometry=json.loads((WORK/'targeted_geometry.json').read_text());constants=json.loads((WORK/'constant_verification.json').read_text())
    check('new_exact_audit_four_checks_current_source',len(audit['checks'])==4 and all(c['passed'] for c in audit['checks']) and all(sha(PAPER/p)==h for p,h in audit['reviewed_sources'].items()))
    check('retained_44_constants_and_7_geometry_current_source',len(constants['checks'])==44 and not constants['failures'] and len(geometry['checks'])==7 and geometry['all_checks_passed'] and geometry['manuscript_sha256']==sha(PAPER/'manuscript.tex'))
    pages={}
    for name in ['manuscript','supplementary']:
        pdf=PAPER/(name+'.pdf'); info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages[name]=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
        log=(PAPER/(name+'.log')).read_text(); extracted=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
        check(name+'_layout_references_and_pdf_text',not any(x in log for x in ['undefined','Overfull \\hbox','Missing character:','multiply defined','Rerun to get cross-references right']) and '??' not in extracted)
        f=subprocess.run(['pdffonts',str(pdf)],capture_output=True,text=True,check=True)
        check(name+'_valid_embedded_fonts',not f.stderr.strip() and all(line.split()[-5]=='yes' for line in f.stdout.splitlines()[2:]))
    aux=(PAPER/'supplementary.aux').read_text()
    check('supplement_S12_S13_navigation', '\\newlabel{S:audit-v34}{{S12}' in aux and '\\newlabel{S:kodak-v34}{{S13}' in aux)
    output={'passed':all(checks.values()),'checks':checks,'pages':pages,'replayed_caches':3072,
         'replayed_queries':3072*1025,'max_replay_difference':max_difference,'replay_failures':replay_failures,
         'V3_3_files_preserved':len(baseline),'inherited_assets_preserved':len(protected)}
    (WORK/'version_validation.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
    raise SystemExit(0 if output['passed'] else 1)

if __name__=='__main__':main()

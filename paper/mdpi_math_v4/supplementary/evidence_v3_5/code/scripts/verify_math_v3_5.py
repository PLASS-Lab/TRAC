#!/usr/bin/env python3
"""Verify preservation, fresh evidence, document consistency and PDF health."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work_dirs/mdpi_math_v3_5'
PAPER=ROOT/'paper/mdpi_math_v3_5'
BASE=ROOT/'paper/mdpi_math_v3_4'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve(p):
    return re.sub(r'\\input\{([^}]+)\}',lambda m:resolve(p.parent/(m[1] if m[1].endswith('.tex') else m[1]+'.tex')),p.read_text())

def main():
    checks={}
    def check(name,condition):checks[name]=bool(condition)
    baseline=json.loads((WORK/'v3_4_baseline_sha256.json').read_text())
    check('V3_4_all_1282_files_unchanged',len(baseline)==1282 and
        set(baseline)=={str(p.relative_to(BASE)) for p in BASE.rglob('*') if p.is_file()} and
        all(sha(BASE/p)==h for p,h in baseline.items()))
    protected={p:h for p,h in baseline.items() if p.startswith(('figures/','supplementary/'))}
    protected.pop('supplementary/SHA256SUMS',None)
    check('all_inherited_figures_and_evidence_byte_identical',all(sha(PAPER/p)==h for p,h in protected.items()))
    a,b=resolve(BASE/'manuscript.tex'),resolve(PAPER/'manuscript.tex')
    pattern=r'\\begin\{(Theorem|Proposition|Lemma|Corollary|proof)\}.*?\\end\{\1\}'
    ma=[m[0] for m in re.finditer(pattern,a,re.S)];mb=[m[0] for m in re.finditer(pattern,b,re.S)]
    check('all_43_math_blocks_byte_identical',len(ma)==len(mb)==43 and ma==mb)
    pattern=r'\\begin\{(equation|align)\}.*?\\end\{\1\}'
    check('all_30_numbered_equations_byte_identical',
        [m[0] for m in re.finditer(pattern,a,re.S)]==[m[0] for m in re.finditer(pattern,b,re.S)])
    eq=set(re.findall(r'\\label\{(eq:[^}]+)\}',b))
    check('all_30_equations_still_referenced',len(eq)==30 and
        all(re.search(r'\\(?:eqref|ref)\{'+re.escape(e)+r'\}',b) for e in eq))
    check('multiscale_proof_file_byte_identical',sha(PAPER/'multiscale_circle_proof.tex')==sha(BASE/'multiscale_circle_proof.tex'))
    supplement=resolve(PAPER/'supplementary.tex')
    check('14_main_figures_9_supplement_figures',b.count('\\begin{figure}')==14 and supplement.count('\\begin{figure}')==9)
    check('old_and_new_actual_inference_figures_in_main',b.count('figures/qualitative_depth_replay.pdf')==1 and
        b.count('figures/fresh_depth_inference.pdf')==1 and 'figures/fresh_depth_inference.pdf' not in supplement)
    check('new_comparison_protocols_and_results_linked',all(x in b for x in
        ['sec:fresh-depth','sec:whole-image-cost','tab:fresh-summary','tab:fresh-cost']) and
        all(x in supplement for x in ['S:prior-v35','S:fresh-v35','tab:fresh-full','tab:fresh-timing-full']))
    check('no_obsolete_no_new_measurements_statement','This revision adds no' not in b)
    frozen=json.loads((WORK/'freeze.json').read_text())
    result=json.loads((WORK/'results.json').read_text());summary=json.loads((WORK/'summary.json').read_text())
    check('sources_match_before_prediction_freeze',all(sha(ROOT/p)==h for p,h in frozen['sources'].items()))
    check('36_frames_two_fixed_models_72_predictions',len(frozen['frames'])==36 and len(frozen['models'])==2 and len(result['records'])==72)
    check('correct_official_tartan_byte_order_checked',frozen['official_depth_decoder_matches_OpenCV'])
    check('432_checks_and_157286400_pixels_per_method',summary['uniform_configurations']==summary['GPA_configurations']==432
          and summary['pixels_per_method']==157286400 and summary['violations']==0)
    check('all_stored_errors_below_proved_budgets',all(z['cache_max_error']<=z['bound']+z['tolerance'] and
        z['GPA_max_error']<=z['GPA_output_bound']+z['tolerance'] for r in result['records'] for z in r['records']))
    check('one_baseline_calibration_per_prediction',all('calibration' in r and all('calibration' not in z for z in r['records']) for r in result['records']))
    groups=[g for g in summary['groups'] if g['S']==32 and g['sigma_r']==.2]
    check('unfavorable_task_and_cost_outcomes_retained',sum(g['changes']['exact_minus_baseline']['AbsRel']['mean']>0 for g in groups)==2 and
          all(c['median_direct_over_uniform']<1 for c in summary['costs'] if c['S']==64) and 'exceeding' in b)
    replay=json.loads((WORK/'portable_replay.json').read_text())
    check('independent_native_filter_and_all_baseline_replay_pass',replay['passed'] and replay['baseline_scores_replayed']==72 and
        replay['native_filter_configurations_replayed']==6 and replay['summary_groups_replayed']==24 and replay['paired_intervals_replayed']==288)
    tests=json.loads((WORK/'implementation_tests.json').read_text())
    check('18_prior_reference_oracle_cases_pass',tests['passed'] and len(tests['cases'])==18)
    pages={}
    for name in ('manuscript','supplementary'):
        info=subprocess.check_output(['pdfinfo',str(PAPER/(name+'.pdf'))],text=True)
        pages[name]=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
        log=(PAPER/(name+'.log')).read_text()
        txt=subprocess.check_output(['pdftotext',str(PAPER/(name+'.pdf')),'-'],text=True)
        check(name+'_references_and_layout_clean',not any(x in log for x in
            ['undefined','Overfull \\hbox','Missing character:','multiply defined','Rerun to get cross-references right']) and '??' not in txt)
        fonts=subprocess.run(['pdffonts',str(PAPER/(name+'.pdf'))],text=True,capture_output=True,check=True)
        check(name+'_fonts_embedded',not fonts.stderr.strip() and
            all(line.split()[-5]=='yes' for line in fonts.stdout.splitlines()[2:]))
    aux=(PAPER/'supplementary.aux').read_text()
    check('supplement_navigation_S14_S15','\\newlabel{S:prior-v35}{{S14}' in aux and '\\newlabel{S:fresh-v35}{{S15}' in aux)
    output={'passed':all(checks.values()),'checks':checks,'pages':pages,'V3_4_files_preserved':len(baseline),
            'inherited_assets_preserved':len(protected),'fresh_predictions':72,'checks_per_method':432}
    (WORK/'version_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
    raise SystemExit(0 if output['passed'] else 1)

if __name__=='__main__':main()

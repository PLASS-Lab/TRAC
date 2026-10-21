#!/usr/bin/env python3
"""Check final v3.2 PDF/source consistency and reporting provenance."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    paper = root / 'paper/mdpi_math_v3_2'
    work = root / 'work_dirs/mdpi_math_v3_2'
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    old = root / 'paper/mdpi_math_v3_1'
    inventory = json.loads((paper / 'supplementary/evidence_v3_2/v3_1_baseline_sha256.json').read_text())
    check('all_1060_baseline_files_unchanged', len(inventory) == 1060 and
          all(digest(old / name) == value for name, value in inventory.items()))
    main_text = (paper / 'manuscript.tex').read_text()
    original = (old / 'manuscript.tex').read_text()
    pattern = r'\\begin\{Theorem\}.*?\\end\{Theorem\}'
    check('original_two_theorem_statements_unchanged',
          len(re.findall(pattern, main_text, re.S)) == 2 and
          re.findall(pattern, main_text, re.S) == re.findall(pattern, original, re.S))
    tex = '\n'.join(p.read_text() for p in paper.glob('*.tex') if p.name != 'supplementary.tex')
    labels = set(re.findall(r'\\label\{(eq:[^}]+)\}', tex))
    check('all_30_main_equations_referenced', len(labels) == 30 and
          all(re.search(r'\\(?:eqref|ref)\{' + re.escape(label) + r'\}', tex) for label in labels))
    info = {}
    for name, expected in [('manuscript', 28), ('supplementary', 23)]:
        result = subprocess.check_output(['pdfinfo', str(paper / (name + '.pdf'))], text=True)
        pages = int(re.search(r'^Pages:\s*(\d+)', result, re.M)[1])
        info[name + '_pages'] = pages
        check(name + '_page_count', pages == expected)
        log = (paper / (name + '.log')).read_text()
        check(name + '_references_and_layout', not any(x in log for x in (
            'There were undefined references', 'Citation `', 'Reference `',
            'Overfull \\hbox', 'Missing character:', 'multiply defined',
            'Rerun to get cross-references right')))
        extracted = subprocess.check_output(['pdftotext', str(paper / (name + '.pdf')), '-'], text=True)
        check(name + '_text_no_unresolved_markers', '??' not in extracted and 'error,+same' not in extracted)
    check('new_table_exactly_frozen_rows',
          (paper / 'adaptive_table_rows.tex').read_bytes() == (work / 'adaptive_table_rows.tex').read_bytes()
          and (work / 'adaptive_table_rows.tex').read_text().strip() in (paper / 'adaptive_table.tex').read_text())
    frozen = json.loads((work / 'freeze.json').read_text())
    check('source_manifest_matches_freeze',
          digest(root / 'work_dirs/practical_upgrade_20261001/nyu_test_manifest.json') == frozen['source_manifest_sha256'])
    archive = work / 'reporting_v1_archive'
    oldfreeze = json.loads((archive / 'freeze.json').read_text())
    for path, value in oldfreeze['sources'].items():
        if path.endswith('/PLAN.md'):
            actual = archive / 'PLAN.md'
        elif path == 'scripts/evaluate_math_v3_2.py':
            actual = archive / 'evaluate_math_v3_2.py'
        else:
            actual = root / path
        check('initial_freeze_source_' + path, digest(actual) == value)
    initial = json.loads((archive / 'results.json').read_text())['records']
    current = json.loads((work / 'results.json').read_text())['records']
    keys = ('image', 'probe', 'sigma_r', 'tolerance', 'adaptive_S', 'sampled_row_l1_error', 'witness_error')
    check('reporting_amendment_preserves_adaptive_results',
          len(initial) == len(current) == 3072 and
          all(all(a[k] == b[k] for k in keys) for a, b in zip(initial, current)))
    check('reporting_amendment_preserves_sampled_families',
          digest(archive / 'sampled_families.npz') == digest(work / 'sampled_families.npz'))
    check('repository_replay_passed', json.loads((work / 'replay_verification.json').read_text())['passed'])
    constants = json.loads((work / 'improved_constant.json').read_text())
    check('constant_verification_passed', len(constants['checks']) == 44 and
          not constants['failures'] and all(row['pass'] for row in constants['checks']))
    geometry = json.loads((work / 'targeted_geometry.json').read_text())
    check('geometry_verification_passed_on_final_source', geometry['all_checks_passed'] and
          geometry['manuscript_sha256'] == digest(paper / 'manuscript.tex'))
    visual = work / 'visual_revision'
    protocol = json.loads((visual / 'protocol.json').read_text())
    manifest = json.loads((visual / 'figure_manifest.json').read_text())
    check('visual_revision_frozen_evidence_unchanged',
          protocol['results_sha256'] == manifest['results_sha256'] == digest(work / 'results.json') and
          protocol['families_sha256'] == manifest['families_sha256'] == digest(work / 'sampled_families.npz'))
    check('all_four_new_figures_match_manifest', len(manifest['figures']) == 8 and
          all(digest(paper / path) == value for path, value in manifest['figures'].items()))
    check('visual_generator_matches_manifest', manifest['source_script_sha256'] ==
          digest(root / 'scripts/build_math_v3_2_explanatory_visuals.py'))
    check('visual_numeric_checks', manifest['adaptive']['archived_bounds_reproduced'] and
          manifest['embedding']['max_row_difference'] < 1e-14 and manifest['stability']['records'] == 4)
    check('visual_array_and_embedding_validation_passed',
          json.loads((visual / 'visual_validation.json').read_text())['passed'])
    snapshot = visual / 'pre_visual'
    check('pre_visual_snapshot_unchanged', all(digest(snapshot / path) == value
          for path, value in json.loads((snapshot / 'SHA256.json').read_text()).items()))
    labels = ['fig:shared-atoms', 'fig:adaptive-construction', 'fig:range-embedding',
              'fig:normalization-stability']
    all_tex = tex + (paper / 'supplementary.tex').read_text()
    check('new_figures_have_text_references', all('\\ref{' + label + '}' in all_tex for label in labels))
    output = {'passed': all(checks.values()), 'checks': checks,
              'v3_1_files_preserved': len(inventory), **info,
              'scope': 'internal_static_and_finite_checks; not_external_review_or_formal_proof'}
    (work / 'final_artifact_validation.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    raise SystemExit(0 if output['passed'] else 1)


if __name__ == '__main__':
    main()

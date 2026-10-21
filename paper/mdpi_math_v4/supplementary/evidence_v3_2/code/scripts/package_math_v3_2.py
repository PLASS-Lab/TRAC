#!/usr/bin/env python3
"""Assemble v3.2 evidence and refresh manifests after PDF validation.

This does not regenerate experimental results or alter earlier versions.
Run again after any manuscript, PDF, documentation or evidence change.
"""
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/mdpi_math_v3_2'
WORK = ROOT / 'work_dirs/mdpi_math_v3_2'
EVIDENCE = PAPER / 'supplementary/evidence_v3_2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    for name in ('PLAN.md', 'freeze.json', 'results.json', 'summary.json',
                 'sampled_families.npz', 'run.log', 'adaptive_table_rows.tex'):
        copy(WORK / name, EVIDENCE / name)
    shutil.copytree(WORK / 'reporting_v1_archive', EVIDENCE / 'reporting_v1_archive',
                    dirs_exist_ok=True)
    sources = [
        'sokkanaem/range_cache_filter.py', 'sokkanaem/certified_scalar_cache.py',
        'scripts/evaluate_math_v3_2.py', 'scripts/build_math_v3_2_figure.py',
        'scripts/build_math_v3_2_explanatory_visuals.py',
        'scripts/verify_math_v3_2_visuals.py',
        'scripts/verify_math_v3_2.py', 'scripts/package_math_v3_2.py',
        'scripts/validate_math_v3_2_artifacts.py',
    ]
    for name in sources:
        copy(ROOT / name, EVIDENCE / 'code' / name)
    (EVIDENCE / 'code/sokkanaem/__init__.py').write_text('')
    copy(ROOT / 'tests/test_math_v3_2.py', EVIDENCE / 'tests/test_math_v3_2.py')
    copy(ROOT / 'work_dirs/practical_upgrade_20261001/nyu_test_manifest.json',
         EVIDENCE / 'source_manifest.json')
    visual = WORK / 'visual_revision'
    if visual.exists():
        shutil.copytree(visual, EVIDENCE / 'visual_revision', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*preview*.png', 'local_fonts'))
    for pattern in ('improved_constant.*', 'targeted_geometry.*',
                    'replay_verification.*', 'tests.log', 'manuscript_validation.log',
                    'bilateral_table_validation.log', 'final_artifact_validation.json',
                    'build_manuscript_final_*.log', 'build_supplementary_final_*.log'):
        for source in WORK.glob(pattern):
            copy(source, EVIDENCE / 'verification' / source.name)
    copy(PAPER / 'supplementary.pdf',
         PAPER / 'supplementary/S1-S11_depth_variance_diagnostics.pdf')
    # Preserve the earlier manifests as historical records, rather than
    # silently labelling their old contents as current v3.2 checksums.
    copy(ROOT / 'paper/mdpi_math_v3_1/ARTIFACT_SHA256.json',
         EVIDENCE / 'v3_1_ARTIFACT_SHA256.json')
    copy(ROOT / 'paper/mdpi_math_v3_1/supplementary/SHA256SUMS',
         EVIDENCE / 'v3_1_supplementary_SHA256SUMS')
    ignored = {'.aux', '.log', '.out', '.fls', '.fdb_latexmk', '.synctex.gz'}

    def include(path):
        rel = path.relative_to(PAPER)
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            return False
        # Archived experimental/verification logs are evidence. TeX build
        # scratch at the package root is reproducible and deliberately omitted.
        if len(rel.parts) == 1 and (path.suffix in ignored or path.name.startswith('build_')):
            return False
        return True

    supplement = PAPER / 'supplementary'
    rows = []
    for path in sorted(supplement.rglob('*')):
        if include(path) and path != supplement / 'SHA256SUMS':
            rows.append(f'{sha(path)}  {path.relative_to(supplement)}')
    (supplement / 'SHA256SUMS').write_text('\n'.join(rows) + '\n')
    manifest = PAPER / 'ARTIFACT_SHA256.json'
    hashes = {str(p.relative_to(PAPER)): sha(p) for p in sorted(PAPER.rglob('*'))
              if include(p) and p != manifest}
    manifest.write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    print(f'Packaged {len(rows)} supplementary checksums and {len(hashes)} artifacts.')


if __name__ == '__main__':
    main()

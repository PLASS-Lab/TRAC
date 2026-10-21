#!/usr/bin/env python3
"""Package V3.5 evidence and refresh checksums, without altering older assets."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper/mdpi_math_v3_5';WORK=ROOT/'work_dirs/mdpi_math_v3_5'
DEST=PAPER/'supplementary/evidence_v3_5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)

def main():
    for name in ('PLAN.md','freeze.json','v3_4_baseline_sha256.json','implementation_tests.json',
                 'results.json','summary.json','predictions.json','timings.json',
                 'openalex.json','semantic.json','curtin_article.json'):
        copy(WORK/name,DEST/name)
    for folder in ('arrays','frame_results'):
        for path in sorted((WORK/folder).glob('*')):
            if path.is_file():copy(path,DEST/folder/path.name)
    for path in sorted(WORK.iterdir()):
        if path.is_file() and (path.suffix=='.log' or path.name in
            ('portable_replay.json','packaged_replay.json','version_validation.json','visual_review.json')):
            copy(path,DEST/'verification'/path.name)
    for name in ('evaluate_math_v3_5.py','build_math_v3_5_figures.py','replay_math_v3_5.py',
                 'verify_math_v3_5.py','package_math_v3_5.py','build_math_v3_2_explanatory_visuals.py'):
        copy(ROOT/'scripts'/name,DEST/'code/scripts'/name)
    copy(ROOT/'sokkanaem/range_cache_filter.py',DEST/'code/sokkanaem/range_cache_filter.py')
    for name in ('scripts','sokkanaem'):(DEST/'code'/name/'__init__.py').write_text('')
    for name in ('PRIOR_ART_V3_5.md','REVISION_V3_5.md','VERSION.md'):
        copy(PAPER/name,DEST/name)
    (DEST/'README.md').write_text('''# V3.5 portable review evidence

PLAN.md and freeze.json predate new model predictions and task scores.
arrays/ contains 36 selected decoded RGB/GT pairs, 72 raw FP32 predictions,
and four fixed qualitative outputs. Original datasets/model weights are not
included. Provenance and checkpoint hashes are retained. results.json stores
all 432 configurations per method; summary.json gives equal-group summaries
and paired descriptive intervals; timings.json includes 24 x 7 raw paired
repeat sets with complete construction and readout costs.

Offline replay (Python with NumPy, SciPy, PyTorch, Pillow; no CUDA or network):

    python code/scripts/replay_math_v3_5.py --output replay_result.json

This reproduces all 72 baseline metrics, six native filter configurations
against archived PyTorch and explicit GPA oracles, 24 summary groups,
288 paired intervals and timing medians. New inference requires the frozen
public checkpoints and source datasets; evaluate_math_v3_5.py documents the
stages. Recorded source paths are the original execution provenance.

The mathematical statements and proofs are unchanged from V3.4. These
computational checks do not constitute external expert proof review.
PRIOR_ART_V3_5.md describes primary-formula sources and the unavailable
original An derivation. Historical evidence directories retain their own
versions, protocols and source hashes.
''')
    copy(PAPER/'supplementary.pdf',PAPER/'supplementary/S1-S15_depth_cache_diagnostics.pdf')
    def include(path):
        rel=path.relative_to(PAPER)
        return path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc' and not (
            len(rel.parts)==1 and (path.suffix in ('.aux','.log','.out','.fls','.fdb_latexmk') or path.name.startswith('build_')))
    sup=PAPER/'supplementary'
    lines=[f'{sha(p)}  {p.relative_to(sup)}' for p in sorted(sup.rglob('*')) if include(p) and p.name!='SHA256SUMS']
    (sup/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    manifest=PAPER/'ARTIFACT_SHA256.json'
    hashes={str(p.relative_to(PAPER)):sha(p) for p in sorted(PAPER.rglob('*')) if include(p) and p!=manifest}
    manifest.write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    print(f'Packaged {len(hashes)} artifacts and {len(lines)} supplementary checksums.')

if __name__=='__main__':main()

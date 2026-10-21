#!/usr/bin/env python3
"""Package new V3.4 evidence; retain all inherited evidence unchanged."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper/mdpi_math_v3_4';WORK=ROOT/'work_dirs/mdpi_math_v3_4'
DEST=PAPER/'supplementary/evidence_v3_4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)

def main():
    for name in ['PLAN.md','freeze.json','download_manifest.json','v3_3_baseline_sha256.json',
                 'results.json','summary.json','families.npz']:
        copy(WORK/name,DEST/name)
    for p in WORK.iterdir():
        if p.is_file() and (p.suffix=='.log' or p.name in ['proof_audit.json','constant_verification.json',
                     'targeted_geometry.json','version_validation.json','visual_review.json','portable_replay.json']):
            copy(p,DEST/'verification'/p.name)
    for name in ['evaluate_math_v3_4.py','audit_math_v3_4.py','build_math_v3_4_figures.py',
                 'verify_math_v3_4.py','replay_math_v3_4.py','package_math_v3_4.py']:
        copy(ROOT/'scripts'/name,DEST/'code/scripts'/name)
    for name in ['certified_scalar_cache.py']:
        copy(ROOT/'sokkanaem'/name,DEST/'code/sokkanaem'/name)
    copy(ROOT/'scripts/build_math_v3_2_explanatory_visuals.py',DEST/'code/scripts/build_math_v3_2_explanatory_visuals.py')
    for name in ['PROOF_AUDIT_V3_4.md','PRIOR_ART_V3_4.md','REVISION_V3_4.md']:
        copy(PAPER/name,DEST/name)
    copy(PAPER/'supplementary.pdf',PAPER/'supplementary/S1-S13_depth_cache_diagnostics.pdf')
    def include(p):
        rel=p.relative_to(PAPER)
        return (p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and
            not (len(rel.parts)==1 and (p.suffix in ['.aux','.log','.out','.fls','.fdb_latexmk'] or p.name.startswith('build_'))))
    supplement=PAPER/'supplementary'
    lines=[f'{sha(p)}  {p.relative_to(supplement)}' for p in sorted(supplement.rglob('*')) if include(p) and p.name!='SHA256SUMS']
    (supplement/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    manifest=PAPER/'ARTIFACT_SHA256.json'
    hashes={str(p.relative_to(PAPER)):sha(p) for p in sorted(PAPER.rglob('*')) if include(p) and p!=manifest}
    manifest.write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    print(f'Packaged {len(hashes)} artifacts, {len(lines)} supplementary checksums.')

if __name__=='__main__':main()

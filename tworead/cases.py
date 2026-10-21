"""Rebuild the dataset-derived evidence arrays from the public datasets and verify them.

The repository does not redistribute dataset pixels or depths. For every
evidence file that contained them, cases/manifest.json records the member
order and the archived SHA-256, and cases/derived/ keeps the remaining
(non-dataset) arrays. `build` extracts exactly the images, frames and patches
used in the manuscript, merges them with the derived arrays and accepts the
rebuilt file only if it matches the archived SHA-256 byte for byte. `verify`
checks every rebuilt file and runs the replays on the completed evidence.

    python -m tworead.cases build --download-kodak --nyu-mat MAT --bonn ROOT --tartan ROOT
    python -m tworead.cases verify
"""
import argparse
import hashlib
import io
import json
import urllib.request
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .common import PACKAGE, dump, sha

REPO = PACKAGE.parent
CASES = REPO/'cases'
EVIDENCE = REPO/'paper/mdpi_math_v4/supplementary'
NYU_LABELED_MAT_SHA256 = '2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa'


def kind(path):
    if path.name == 'families.npz': return 'kodak'
    if path.name == 'sampled_families.npz': return 'nyu'
    return 'depth'


def from_dataset(kind, name):
    if kind == 'kodak': return name.endswith(('_guides', '_values'))
    if kind == 'nyu': return name.startswith(('keys_', 'values_'))
    return name in ('rgb', 'gt')


def probes(shape):
    """The 16 fixed radius-six probe centres of the storage diagnostics."""
    ys = np.linspace(6, shape[0]-7, 4).round().astype(int)
    xs = np.linspace(6, shape[1]-7, 4).round().astype(int)
    return [(y, x) for y in ys for x in xs]


def patch(a, y, x):
    return a[y-6:y+7, x-6:x+7].reshape(-1, *a.shape[2:])


def luma(rgb):
    """0.299r+0.587g+0.114b rounded exactly as the archived runs (NumPy 2 matmul).

    That kernel rounds the green product, then fuses the red and blue
    multiply-adds. Exact rational arithmetic reproduces it bitwise on any
    platform; plain float code differs in the last bit for some pixels.
    """
    def one(r, g, b):
        x = float(Fraction(g)*Fraction(.587))
        x = float(Fraction(r)*Fraction(.299)+Fraction(x))
        return float(Fraction(b)*Fraction(.114)+Fraction(x))
    return np.array([one(*map(float, p)) for p in rgb])


# ----- dataset extraction -----------------------------------------------------
def kodak(evidence, folder, download):
    arrays = {}
    for image in json.loads((evidence/'evidence_v3_4/freeze.json').read_text())['images']:
        i = image['index']
        candidates = [Path(folder)/f'kodim{i:02d}.png', Path(folder)/f'{i:02d}.png'] if folder else []
        path = next((p for p in candidates if p.is_file()), None)
        if path is not None:
            body = path.read_bytes()
        elif download:
            with urllib.request.urlopen(image['source_url'], timeout=60) as f:
                body = f.read()
        else:
            raise FileNotFoundError(f'Kodak image {i:02d} not found in {folder}')
        assert hashlib.sha256(body).hexdigest() == image['sha256'], f'Kodak {i:02d} differs'
        rgb = np.asarray(Image.open(io.BytesIO(body)).convert('RGB'), dtype=np.float64)/255
        for probe, (y, x) in enumerate(probes(rgb.shape)):
            arrays[f'i{i:02d}_p{probe:02d}_guides'] = luma(patch(rgb, y, x))
            arrays[f'i{i:02d}_p{probe:02d}_values'] = patch(rgb[:, :, 0], y, x)
    return arrays


def nyu(evidence, mat):
    import h5py
    assert sha(mat) == NYU_LABELED_MAT_SHA256, 'not the official nyu_depth_v2_labeled.mat'
    arrays = {}
    with h5py.File(mat, 'r') as f:
        for image_no, row in enumerate(json.loads((evidence/'evidence_v3_2/freeze.json').read_text())['images']):
            rgb = np.asarray(f['images'][row['index']]).transpose(2, 1, 0)
            depth = np.asarray(f['depths'][row['index']], dtype=np.float32).T
            saved = io.BytesIO(); np.save(saved, depth)     # the archived per-image .npy
            assert hashlib.sha256(saved.getvalue()).hexdigest() == row['depth_sha256']
            rgb = rgb.astype(np.float64)/255
            for probe, (y, x) in enumerate(probes(rgb.shape)):
                arrays[f'keys_{image_no}_{probe}'] = luma(patch(rgb, y, x))
                arrays[f'values_{image_no}_{probe}'] = patch(depth, y, x).astype(np.float64)
    return arrays


def relocate(path, group, root):
    parts = Path(path).parts
    return str(Path(root, *parts[parts.index(group):]))


def depth_frames(evidence, bonn, tartan):
    from .experiments.depth import decoded
    arrays = {}
    for r in json.loads((evidence/'evidence_v3_5/freeze.json').read_text())['frames']:
        root = bonn if r['dataset'] == 'Bonn' else tartan
        if root is None: continue
        r = dict(r, rgb=relocate(r['rgb'], r['group'], root), depth=relocate(r['depth'], r['group'], root))
        for key in ('rgb', 'depth'):
            assert sha(r[key]) == r[key+'_sha256'], f'{r[key]} differs from the frozen frame'
        arrays[r['id']] = dict(zip(('rgb', 'gt'), decoded(r)))
    return arrays


# ----- assembly ----------------------------------------------------------------
def assemble(evidence, rel, entry, source):
    derived = np.load(CASES/'derived'/rel)
    arrays = {m: source[m] if from_dataset(entry['kind'], m) else derived[m] for m in entry['members']}
    path = evidence/rel
    np.savez_compressed(path, **arrays)
    if sha(path) != entry['sha256']:
        path.unlink()
        raise AssertionError(f'{rel}: rebuilt file differs from the archived SHA-256')
    print('rebuilt', rel, flush=True)


def build(args):
    manifest = json.loads((CASES/'manifest.json').read_text())['files']
    sources = {}
    if args.kodak or args.download_kodak: sources['kodak'] = kodak(args.evidence, args.kodak, args.download_kodak)
    if args.nyu_mat: sources['nyu'] = nyu(args.evidence, args.nyu_mat)
    if args.bonn or args.tartan: sources['depth'] = depth_frames(args.evidence, args.bonn, args.tartan)
    if not sources: raise SystemExit('nothing to build: pass a dataset option (see --help)')
    for rel, entry in manifest.items():
        source = sources.get(entry['kind'])
        if entry['kind'] == 'depth' and source is not None:
            source = source.get(Path(rel).name[:3])
            if source is not None and rel.endswith('_qualitative.npz'):   # audit stored GT in FP64
                source = dict(source, gt=source['gt'].astype(np.float64))
        if source is not None:
            assemble(args.evidence, rel, entry, source)


def verify(args):
    from .experiments import depth, kodak as kodak_experiment
    manifest = json.loads((CASES/'manifest.json').read_text())['files']
    status = {}
    for rel, entry in manifest.items():
        p = args.evidence/rel
        status[rel] = 'missing' if not p.exists() else 'ok' if sha(p) == entry['sha256'] else 'differs'
    assert 'differs' not in status.values(), [r for r, s in status.items() if s == 'differs']
    ready = lambda kind: all(status[r] == 'ok' for r, e in manifest.items() if e['kind'] == kind)
    result = {'files': {k: sum(v == k for v in status.values()) for k in ('ok', 'missing')}}
    result['nyu'] = 'byte-identical to the archived sampled_families.npz' if ready('nyu') else 'not built'
    result['kodak_replay'] = kodak_experiment.replay(args.evidence/'evidence_v3_4') if ready('kodak') else 'not built'
    result['depth_replay'] = depth.replay(args.evidence/'evidence_v3_5') if ready('depth') else 'not built'
    if args.output: dump(args.output, result)
    print(json.dumps(result, indent=2))


def strip(args):
    """Maintainer step: split archived files into manifest + derived arrays (run once)."""
    files = {}
    for path in sorted(args.evidence.rglob('*.npz')):
        rel = path.relative_to(args.evidence)
        data = np.load(path); k = kind(path)
        if k == 'depth' and not path.name.endswith(('_inputs.npz', '_qualitative.npz')): continue
        files[str(rel)] = {'kind': k, 'sha256': sha(path), 'members': list(data.files)}
        out = CASES/'derived'/rel; out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **{m: data[m] for m in data.files if not from_dataset(k, m)})
    dump(CASES/'manifest.json', {'evidence': str(args.evidence.relative_to(REPO)), 'files': files})
    print(len(files), 'files described')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=('build', 'verify', 'strip'))
    parser.add_argument('--evidence', type=Path, default=EVIDENCE)
    parser.add_argument('--kodak', type=Path, help='folder with kodim01.png ... kodim24.png')
    parser.add_argument('--download-kodak', action='store_true', help='fetch the 24 images from the pinned public mirror')
    parser.add_argument('--nyu-mat', type=Path, help='official nyu_depth_v2_labeled.mat (needs h5py)')
    parser.add_argument('--bonn', type=Path, help='folder containing the rgbd_bonn_* sequences')
    parser.add_argument('--tartan', type=Path, help='TartanAir V2 root containing the environment folders')
    parser.add_argument('--output', type=Path, help='verification JSON record (verify)')
    args = parser.parse_args()
    {'build': build, 'verify': verify, 'strip': strip}[args.stage](args)


if __name__ == '__main__':
    main()

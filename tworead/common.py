"""Hashing and JSON output shared by the theory and experiment modules."""
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def single_thread():
    """Pin BLAS threads before NumPy is imported, as in the timed experiments."""
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    """Atomic JSON write; NaN is rejected so records stay strict JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_hashes(*modules):
    """Hashes of package source files, keyed by their path inside the package."""
    return {str(Path(m.__file__).resolve().relative_to(PACKAGE.parent)): sha(m.__file__)
            for m in modules}

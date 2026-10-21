# V3.4 evidence packet

This packet adds new Kodak cache validation, complete CPU lifetime costs,
internal proof audit and primary-source comparison to the inherited evidence.
No model training or new depth inference is performed.

- PLAN.md and freeze.json: pre-score protocol and source/input hashes.
- families.npz: all 384 guide/value patches and all 3072 node/envelope arrays.
- results.json: every scored cache and raw timings (24 images, 15 repetitions).
- summary.json: image-level ratios, bootstrap intervals and complete-cost summaries.
- verification/: exact audit, 44 constant / 7 geometry outputs, builds and replay.
- code/: exact source copies. Run from a checkout with scripts/ and sokkanaem/.
- v3_3_baseline_sha256.json: complete 1230-file untouched baseline inventory.

The code snapshot uses Python, NumPy, PyTorch, Pillow and SymPy; figures additionally
use Matplotlib/fontTools. Recorded scoring versions: NumPy 2.4.6, PyTorch 2.8.0.
One-thread BLAS/OMP settings and the actual CPU are in results.json. Canonical
photo URLs, mirror commit and hashes permit redownload; full photographs are
not required for replay of the portable neighborhood arrays.

Workspace verification:

    PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_math_v3_4.py

Portable replay (NumPy only):

    python code/scripts/replay_math_v3_4.py .

The standalone NumPy replayer accepts this evidence directory as an argument;
it needs neither source photographs nor the original absolute data paths.
Source experiments are immutable: run download -> freeze -> run -> summarize
in a separate output location to conduct another experiment. Never overwrite
this frozen report. Byte counts refer to packed required persistent payload,
not Python objects or measured resident memory. Scope/limits are in S12–S13.

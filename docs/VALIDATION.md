# Validation record

Recomputed with this package on 2026-10-20 (Python 3.11, NumPy 1.26.4,
PyTorch 2.8.0) and re-checked from this repository layout on 2026-10-21.
References for comparison: the archived records in the supplementary evidence
package (`paper/mdpi_math_v4/supplementary/`) and the outputs of the original
research scripts. Result JSON files are in `validation/`. The raw NYUv2 and
Kodak result arrays (npz) are not stored because of their size.

| Target | Command | Result |
|---|---|---|
| Core self-check | `tests/check_core.py` | passed |
| Thm. 1, Prop. 1 constants | `theory.constants` | 44/44 passed. Exact interval-arithmetic parts identical to the original record; floating evidence within relative error 1e-11 |
| Lemmas 3–5 boundary cases | `theory.geometry_audit` | all 4 checks identical to the original record (only the manuscript-hash field removed) |
| Thm. 1 upper-construction grid | `theory.rates` | slope −2.0533, within 1e-13 of the original record |
| Prop. 2 relaxation grid | `theory.relaxation` | grid values within 1e-13 of the original record. The separation-ratio column now uses the manuscript constants and exceeds 1 from S = 461 (as in Sec. 2.4) |
| Kodak replay (archived evidence) | `experiments.kodak replay` | 3072 caches, 3,148,800 queries, max difference 1.2e-16 |
| Depth replay (archived evidence) | `experiments.depth replay` | 72 predictions, 288 intervals, 24 timing configurations, metric gap 0 |
| Filter oracle | `experiments.depth test` | identical to the original `implementation_tests.json` |
| NYUv2 rerun from original data | `experiments.variance freeze/run/summarize` | 3072 records; all integer fields equal, real fields within relative 2.7e-14. Stable-normalization and embedding records identical |
| Kodak rerun from original data | `experiments.kodak freeze/run/summarize` | 3072 records; all integer fields equal, real fields within relative 1.6e-14. Storage group statistics identical |

On 2026-10-21 the commands (`check_core`, `constants`,
`geometry_audit`, `rates`, `relaxation`, `kodak replay`, `depth replay`,
`depth test`) were run again from the repository root; every output was
identical to the files in `validation/`. The replays were also run against
the in-repository evidence (`paper/mdpi_math_v4/supplementary/`) with
identical results, and `evidence_proof_v3_1/constant_verification.json` has
the same 44 check names as `validation/constants.json`, all passing.

## Dataset cases

The 43 dataset-derived evidence files were removed from the repository and
rebuilt with `python -m tworead.cases build` from the original downloads
(Kodak from the pinned public mirror, the official `nyu_depth_v2_labeled.mat`,
the Bonn RGB-D and TartanAir V2 frames). All 43 rebuilt files were
byte-identical to the archived files, and `sha256sum -c SHA256SUMS` passed for
all 419 evidence files. `python -m tworead.cases verify` then reproduced the
Kodak and depth replay records exactly (`validation/cases.json`).

The archived NYUv2 and Kodak guides were computed with the NumPy 2 matrix
product, which rounds the green product and then fuses the red and blue
multiply-adds. NumPy 1.26 differs from it in the last bit for some pixels, so
`tworead.cases` reproduces that rounding in exact rational arithmetic and is
bitwise independent of the installed NumPy/BLAS.

## Comparison with the manuscript (rerun values)

- NYUv2 (Sec. 5.1): median ratio 8.49 (σr = 0.05) and 0.93 (σr = 0.4); same-size adaptive wins 2/384; max error/tolerance 0.7872.
- Kodak (Sec. 5.1, Table 1): 4.29 [4.10, 4.62] and 2.09 [1.95, 2.20]; wins 25/384; more nodes in 2 cases; packed state 2356/1640 bytes; max error/tolerance 0.77105 in both the archived record and the rerun (reported as 0.771).
- Kodak lifetime cost, median speedup at Q = 4096: rerun 5.67/18.93, manuscript 5.79/19.42. These are wall-clock timings and depend on machine state; the conclusions (e.g. the number of images where the adaptive cache is faster at Q = 256) are unchanged.
- Depth (Sec. 5.2–5.3, Tables 2–3): reproduced from the archived arrays by `depth replay` with zero metric gap.

## Not rerun

The full `depth freeze/predict/audit/time` pipeline (GPU inference) was not
rerun. The numbers from those stages were checked with `depth replay` and
`depth test` on the archived arrays.

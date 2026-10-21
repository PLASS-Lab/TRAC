# v3.2 evidence

This directory contains new evidence only. Other `evidence*` directories and
older S1–S9 PDF copies are historical records; the current supplement is
`../../supplementary.pdf` or `../S1-S11_depth_variance_diagnostics.pdf`.
The original manuscript remains in `../../../mdpi_math_v3_1/`.

The analytic variance envelope holds in real arithmetic. FP64 calculations
and finite query grids below are diagnostics, not interval-arithmetic
certificates or a proof of a continuous incidence statement.

## Replay without NYUv2 or model weights

Dependencies: Python 3.10+, NumPy and PyTorch. From this directory:

```bash
PYTHONPATH="$PWD/code" python code/scripts/verify_math_v3_2.py --evidence "$PWD"
```

This verifies frozen code/plan hashes, all 3072 adaptive schemes and 789504
sampled query rows, sufficient uniform budgets, grouped medians, constant
preservation records and the four-key embedding records. It regenerates
adaptive nodes from the stored 384 neighborhoods rather than trusting the
recorded node counts. It needs neither NYUv2 files nor downloaded weights.
The original v3.1 tree is checked only when present beside the repository;
the standalone replay therefore reports zero baseline files checked.
The packaged standalone replay passed all 12309 checks; the repository
replay passed 13369 checks including the 1060 original baseline files.

Regression tests additionally require pytest:

```bash
PYTHONPATH="$PWD/code" python -m pytest -c /dev/null -p no:cacheprovider --noconftest tests/test_math_v3_2.py
```

The eight cases check normalization, local underflow, a direct snapshot
oracle, the variance envelope, constant keys and invalid node ordering.
The repository regression run also checks 13 existing depth/visual cases.

## Files and provenance

- `PLAN.md` and `freeze.json`: workload, recorded reporting amendment,
  all input SHA-256 values and code hashes fixed before corrected scoring.
- `results.json`: every scheme, stability and embedding records. There are
  1536 families, each used at two declared tolerances; these are correlated
  diagnostics on 24 reused images, not 3072 independent observations.
- `summary.json` and `adaptive_table_rows.tex`: all eight bandwidth/tolerance
  groups, including unfavorable adaptive comparisons.
- `sampled_families.npz`: actual keys and values for all 384 neighborhoods;
  all eight first-neighborhood grids/bounds; other grids regenerate exactly.
- `source_manifest.json`: the earlier image manifest, whose hash is frozen.
  Paths record the original local data layout; full NYUv2 images are not
  redistributed. The replay uses only the extracted numerical neighborhoods.
- `code/` and `tests/`: current implementations and exact source copies.
  `evaluate_math_v3_2.py` and `build_math_v3_2_figure.py` retain the repository
  layout; they are archival generation sources, not standalone replay CLIs.
- `range_cache_filter_legacy.py` and `evaluate_bilateral_cache_legacy.py`:
  unmodified pre-v3.2 programs for interpreting historical numerical results.
- `reporting_v1_archive/`: complete initial scoring outputs, original runner,
  original pre-amendment plan and freeze. The initial sufficient uniform
  comparator reported one node for constant keys; the current comparator
  enforces two. No adaptive grid, data, tolerance or query set changed.
- `verification/`: final PDF/static checks, existing finite proof checks,
  repository regression tests and complete numerical replay. These are
  internal checks, not external peer review.
- `visual_revision/`: source protocol, manifests, validation and pre-visual
  source/PDF snapshot for the three added main figures and one supplementary
  figure. These explain existing mathematics and records; there is no new
  experiment. Locally derived font files are not redistributed.
- `v3_1_baseline_sha256.json`: all 1060 original files, including historical
  build files. The other v3.1 manifests are retained under explicit v3.1 names.
- `an2015_crossref.json`, `chaudhury2016_crossref.json`: primary DOI metadata;
  the related-work scope and access limits are in
  `../source_notes/PRIOR_ART_V3_2.md`.

Full generation uses `scripts/evaluate_math_v3_2.py --freeze --run --summarize`
from the repository with the original data paths, source manifest and archived
bilateral results. A completed run refuses to overwrite its freeze/results;
use an isolated output directory for further research. Figure generation is
`scripts/build_math_v3_2_figure.py`. Package assembly is
`scripts/package_math_v3_2.py`; it does not rerun or select experiments.
The added explanatory figures use
`scripts/build_math_v3_2_explanatory_visuals.py` and the same frozen records;
figure generation additionally needs Matplotlib, fontTools and system Pagella.

Current package hashes are `../../ARTIFACT_SHA256.json` (relative to the
manuscript directory) and `../SHA256SUMS` (relative to supplementary/).
Regenerate them after changing any artifact; their historical v3.1 copies
are not current integrity manifests.

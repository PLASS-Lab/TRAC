# v3.2 variance-envelope diagnostics — 2026-10-08

Declared before any v3.2 diagnostic scoring. The preceding code regression
reproduced the constant-image denominator-floor failure; no adaptive result
has been inspected when this plan is written.

Use the same 24 NYUv2 images as the original bilateral panel, every 27th
manifest row. No model inference, training, GT accuracy ranking, independent
holdout claim, or full-image speed experiment. Values are the archived depth
image; node selection never reads values. Dataset files are not redistributed.

At each of a fixed 4 by 4 grid of interior coordinates take the 13 by 13
neighborhood with sigma_s=2, radius 6. For each fixed neighborhood keep its
169 guides and spatial weights fixed, and allow an independent query g in
[0,1]. This is the continuous weighted range-query class. It is distinct
from changing a guide pixel in a complete image and rebuilding the family.

Range sigma: 0.05, 0.1, 0.2, 0.4. Declared normalized tolerances: 0.01,
0.001. Per family, bisect every cell whose proved variance-tilting envelope
h^2 min(K^2/4, exp(Kh/2) Var_mid)/4 exceeds the tolerance. Values and
evaluation queries are not used in node selection. Stop only when all cells
pass. Report the numerical FP64 evaluation as such, not as an outward-rounded
machine certificate. The guarantee is the analytic theorem in real arithmetic.

For 257 uniform queries, report worst row l1 error (the exact value-box
worst case at each query), a corner-value witness, and error of the archived
depth values. Compare storage with globally certified uniform spacing,
local-key-range certified uniform spacing, and sampled error of uniform
spacing at the same S. Report all outcomes, including cases where adaptive
spacing is worse. No universal instancewise dominance claim.

Expected workload: 24 images * 16 neighborhoods * 4 bandwidths = 1536
families, each at 2 tolerances = 3072 schemes. These share images and are
not independent statistical observations. No confidence intervals or
significance tests. Archive all rows, actual keys/values, selected first
family grids, code and input hashes. Do not rerun or overwrite a completed
freeze/result. Retain all v3.1 evidence byte for byte.

Separately check the positive-weight four-key embedding symbolically and
on a dense scalar grid; record constant-image and disjoint-region stability
before/after in FP32 and FP64; recover the 48-group versus 1152-individual
quantile comparison directly from the old raw records.

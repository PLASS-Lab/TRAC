# V3.4 upgrade — 2026-10-09

Three recommended items were addressed. The 1230 files of V3.3 are preserved
in their original directory by SHA-256, and the new manuscript is built in
`mdpi_math_v3_4/`.

1. **Review of the core proof.** The two normalization branches of fixed
   contact centers and clipping, simultaneous localization of two atoms,
   summation over half-open bins, and the heavy-arc cover across scales
   were re-examined from their definitions. The limit of the singular
   integral and the closure of bins were made explicit. New exact
   rational/integer checks cover 38,226 center cases, 1950 bin
   configurations, 12,950 arc families and the same number of
   repeated-center measures. The existing 44 constant checks and 7
   geometric checks are also rerun. The supporting record is in main
   Section 3.2, Supplementary S12 and `PROOF_AUDIT_V3_4.md`.
   This is not presented as an external expert review or formal
   verification.
2. **Comparison with prior work.** Main Section 1.1 adds a comparison table
   (object, error norm, storage, lookup, guarantee) and a comparison of
   filter formulas in the same intensity units. The kernel/output error
   conditions of the original papers are stated, and An's estimate of the
   number of components is identified as the formula reproduced in
   Eq. (34) of Chaudhury–Dabhade. Because An's original derivation and
   assumptions could not be accessed, no superiority of constants is
   claimed. Supplementary S12 and `PRIOR_ART_V3_4.md` explain the sources.
3. **New data and complete cost.** All 24 Kodak photographs were frozen
   before any result was inspected. 3,148,800 queries were checked on 384
   new neighborhoods, 1536 families and 3072 caches disjoint from NYUv2,
   with zero violations. A separate NumPy implementation reproduces the
   PyTorch results, and image-level statistics are used. The CPU
   experiment on the first neighborhood includes construction, metadata
   and lookup. Reported in main Section 4.5, a table, two new vector
   figures and Supplementary S13.

Savings in stored cells, error improvement at equal size, byte counts
including metadata, and amortization of construction cost are different
things. At narrow bandwidths the cell count was reduced, but in the selected
cost experiment the adaptive cache had a higher complete cost than the
uniform cache. At Q=4096 both methods were faster than direct evaluation,
with median paired speed ratios of 5.79 and 19.42. This is not generalized
to whole-image, GPU or neural-network speed results.

The existing four-key theorem, lower-bound constants, multiscale-circle
proof file and the 30 numbered equations are kept. Of the 43 formal
environments, only the endpoint explanations of two proofs were changed;
the other 41 are preserved. Figure 10 of the earlier depth inference and all
previous figures and experimental evidence are also kept. No new depth-model
inference or training was performed.

`supplementary/evidence_v3_4/` provides the frozen plan, input/source hashes,
raw results, all patch/node/envelope arrays, repeated measurements and
verification logs. The current PDF page counts and verification status are
in `VERSION.md` and `UPGRADE_RESULTS.md`.

---
*English translation of the original Korean revision note (2026-10-21).
Section, table and figure numbers refer to the draft of that time, not to the manuscript.*

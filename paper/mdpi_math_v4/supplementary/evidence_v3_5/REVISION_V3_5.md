# V3.5 changes — 2026-10-09

This revision focuses on the requested prior-work comparison and the
structure of the applications. The 1282 files of the V3.4 directory were
frozen as a baseline before a separate V3.5 was created. The 43 existing
proof/theorem blocks, the 30 numbered equations and the multiscale proof
file are all identical. No claim of an external proof review is added.

## Comparison with prior work

- Main Section 1.1 briefly connects the key differences among dictionary
  width, independent spaces, attention resources and error-controlled
  filtering.
- The DeVore–Temlyakov positive-volume assumption versus the measure-zero
  curve here, and the S^-4 of independent planes versus the S^-2 of shared
  atoms, are kept.
- Public access to the An paper was rechecked. Curtin has metadata only;
  OpenAlex/Semantic Scholar list no public full text. S14 states that the
  review of the original derivation is incomplete and that it was checked
  only indirectly through Chaudhury–Dabhade Eq. (34).
- An independent depth-value extension of GPA, based on the published
  Eqs. (7) and (15), was implemented, and S14 derives the denominator lower
  bound needed at the centre query and the output bound. All methods use
  the same spatial kernel and a sufficient error budget.
- Joint GPA uses 2N scalar filterings; uniform uses 2S and then two snapshot
  reads. The N+1 optimization of the self-bilateral filter is not applied
  where it does not hold.

## Real inference and application description

- 72 new GPU FP32 inferences with fixed public DA-V2 Small / ZoeDepth
  models: Bonn, 8 sequences × 2 frames; TartanAir V2, 10 environments ×
  2 frames.
- Input, code and model SHA hashes frozen beforehand; no training,
  parameter selection or attention change.
- FP64 output filters evaluated at the native 480×640 / 640×640 size.
  Tartan was checked beforehand against the official BGRA float32 decoder;
  Bonn uses uint16/5000.
- The DA-V2 GT-assisted inverse-depth fit is computed once on the original
  output and reused for all filtered outputs. ZoeDepth follows the raw
  metric protocol.
- The main text separates fidelity, task accuracy and complete cost. A
  representative table and real original/exact-filter/cache-filter/error
  maps are in the main text; all settings and repeated records are in S15.
  The earlier real-inference figure and the 11-model results are preserved.

## Results and limitations

- Zero bound violations in 432 image/setting checks each for uniform and
  GPA, each over 157,286,400 pixel evaluations. Uniform maximum
  error/bound 0.129024.
- Mean AbsRel change of the exact filter at fixed sigma_r=0.2, S=32:
  Bonn DA-V2 -0.000977, ZoeDepth +0.000195;
  Tartan DA-V2 -0.001897, ZoeDepth +0.000110.
  All cache-minus-exact mean changes are below 3e-7 in magnitude.
- At S=32, whole-image uniform about 257 ms; GPA about 123/70 ms at
  sigma_r=.2/.4. Paired median uniform/GPA ratios 2.10/3.67.
- At S=64, uniform about 511 ms is slower than direct about 332 ms. Logical
  FP64 snapshots 150–200 MiB. At S=8, sigma_r=.2, uniform is faster than
  GPA. No overall cost advantage is claimed; the meaning of the resource
  model is made concrete.
- Measurements: 4 fixed value images, 24 timing configurations, one CPU
  thread, two warmups/seven repeats. This compares the reference
  implementations of this study and does not establish superiority over
  optimized author code or GPU inference.
- The data were reused from the existing project, and overlap with
  pretraining data is unverified. Group bootstrap intervals are descriptive,
  not claims of generalization or significance. In particular, the Bonn
  sequences share a recording space. Raw and GT-assisted scores are not
  merged into one model ranking.

## Reproduction checks

18 explicit GPA / existing PyTorch oracle cases pass. All 72 original
metrics, 24 group summaries, 288 paired intervals and 24 timing medians were
reconstructed. Six settings on one native image were compared with a
separate PyTorch/explicit polynomial oracle, with a maximum difference
within 3.56e-15. Copied replay code that runs without data, weights or a GPU
is provided in supplementary evidence_v3_5.

The preservation checks for existing formal blocks and equations and the PDF
reference/font/layout checks are recorded in version_validation.json. The
three author-input placeholders are kept.

---
*English translation of the original Korean revision note (2026-10-21).
Section, table and figure numbers refer to the draft of that time, not to the manuscript.*

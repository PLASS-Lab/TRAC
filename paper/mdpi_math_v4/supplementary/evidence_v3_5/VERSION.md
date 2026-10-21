# Mathematics V3.5 — 2026-10-09

Current PDFs: manuscript.pdf (37 pages), supplementary.pdf (26 pages, S1–S15).
Baseline: ../mdpi_math_v3_4/, all 1282 files preserved byte for byte.

Focus: concise source-level prior comparison, fresh fixed-checkpoint inference,
and whole-image filtering costs with a matched sufficient output-error bound.
All 43 formal mathematical blocks, 30 numbered equations, theorem constants,
multiscale proof, inherited figures and evidence remain unchanged.

New panel: 36 native-resolution Bonn/TartanAir V2 frames, two fixed public
checkpoints, 72 new FP32 predictions. Uniform and joint GPA each have 432
FP64 maximum-error checks, each covering 157286400 pixel evaluations, with
zero violations. Fixed baseline relative-depth calibration is reused after
filtering. Negative accuracy and whole-image cost outcomes are retained.

Main figures: 14, including two new actual-inference/accuracy-cost figures;
supplement: 9 inherited figures, with new protocols/tables in S14/S15.
Whole-image timing: four fixed value images, 24 configurations, one CPU
thread, two warmups and seven measured paired repeats.

Read REVISION_V3_5.md and PRIOR_ART_V3_5.md. New portable evidence lives in
supplementary/evidence_v3_5/. Build main first, then supplement, twice each.
No external expert proof review, submission, upload or publication occurred.
An's original derivation remains unavailable; the indirect formula source is
explicit. Three visible author inputs remain in the submission checklist.

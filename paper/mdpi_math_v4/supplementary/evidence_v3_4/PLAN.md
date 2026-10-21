# V3.4 protocol, frozen before scoring (2026-10-09)

1. Audit the fixed-center, per-bin and cross-scale arguments from their
   definitions. Record every branch, endpoint convention and quantifier;
   new exact finite adversarial checks complement the retained verifiers.
   This is an internal audit, not external specialist or formal verification.
2. Compare target set, norm, storage, query accesses and guarantee. Use primary
   full texts for formula comparisons; label abstract-only sources explicitly.
3. Evaluate all 24 Kodak PCD0992 photographs, kodim01--24, without outcome
   selection or parameter tuning. They are disjoint from all earlier NYUv2
   diagnostic images. Canonical source: https://r0k.us/graphics/kodak/;
   public mirror and immutable commit are recorded in freeze.json.
   No training, task-accuracy scoring or depth-model inference is performed.

Use native-resolution RGB /255, luminance guide (0.299,0.587,0.114), red-channel
values, a 4x4 evenly spaced interior grid, radius 6, spatial sigma 2. For each
of 384 patches, evaluate sigma_r = .05,.1,.2,.4 and normalized tolerance
.01,.001, giving 3072 new caches. Query grid: 1025 points on [0,1]. Nodes must
depend only on keys, weights and tolerance. Store raw patches, all nodes,
envelopes and results. Check row L1 errors, actual bounded-value errors,
same-size uniform errors, and local/global certified uniform budgets.

Report per-image means before image-level medians and paired bootstrap
95% intervals (2000 resamples, seed 20261009). Pixels and queries are not
independent statistical units; the interval describes this fixed image suite.

Timing: NumPy FP64 reference backend, one CPU thread, all 24 images' first
predeclared probe, sigma_r=.1, tolerance=.001. For each cache arm (adaptive,
local-span sufficient uniform), measure node/certificate construction,
snapshot evaluation and persistent bytes separately. Certificates are retained
for the reported storage total. Compare with stable direct softmax queries.
Measure metadata lookup, complete cached queries and direct queries at
Q=1,16,256,4096 (one query batch per newly constructed cache), 3 warmups and
15 paired repeats with cyclic arm order. No disk I/O or image decoding enters
the timer. Report measured total construction + query cost and observed
break-even on this four-point grid; no whole-image/GPU/network speed claim.

Independent replay uses NumPy, not the PyTorch cache implementation. Freeze
all input and implementation hashes before scoring; reject result overwrite.
The published FP64 envelope is an analytic guarantee evaluated numerically,
not an outward-rounded interval certificate.

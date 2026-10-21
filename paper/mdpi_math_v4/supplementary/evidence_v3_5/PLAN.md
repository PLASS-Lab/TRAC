# V3.5 frozen application extension

Freeze this plan, evaluator and inputs before any new model output or task score.
Preserve V3.4 and its analytic proof statements; this is not an external proof review.

- Two public pretrained checkpoints: Depth Anything V2 Small and ZoeDepth
  NYU/KITTI, exact revisions from the archived V2 weight manifest. No training,
  tuning, checkpoint selection or attention modification. GPU FP32 inference,
  TF32 disabled, no mixed precision; public training-data overlap is unknown.
- Bonn: first eight installed RGB-D sequence directories in lexicographic order;
  nearest RGB/depth timestamp pairs within 0.02 s, frames at rounded fractions
  1/3 and 2/3 of each paired sequence. TartanAir V2: all ten installed environments,
  Data_easy/P002, left-front camera, same fractions of paired frame IDs. This is
  a new filter-panel replication on reused project data, not a held-out benchmark.
- Native RGB resolution, no crop. Bonn uint16 depth / 5000, valid 0.1--10 m;
  Tartan depth: official OpenCV BGRA byte ordering, little-endian float32,
  valid 0.1--80 m. PIL RGBA bytes are reordered before reinterpretation and
  checked against OpenCV on the first selected synthetic frame. Nonfinite/zero
  reference depths are excluded. No prediction-dependent mask.
- Relative DA-V2: least-squares affine inverse-depth fit to the *unfiltered*
  prediction on the fixed valid mask, reused unchanged for all filtered outputs.
  Inverse depths are bounded below by 1 / the dataset's maximum valid depth.
  This GT-assisted diagnostic is not metric-depth deployment. ZoeDepth uses raw
  metre outputs, with only a positivity floor of 0.001 m for metric evaluation.
  Record calibration coefficients and clipping fractions for every output.
- FP64 CPU output filters, identical luminance guide [0,1], spatial sigma 2,
  finite radius 6 and replicated boundaries. Range sigma 0.2, 0.4; S=8,32,64.
  Uniform snapshots and direct exact joint bilateral oracle. Centre each value
  image at the midpoint of its extrema; radius R is half its native range.
  Check every pixel against R/[16 sigma_r^4(S-1)^2], plus 1e-10 max(1,R).
- Joint Gaussian-polynomial (GPA) reference adapts Chaudhury--Dabhade (2016),
  Eq. (7), to independent depth values. Choose the smallest integer N>lambda,
  lambda=(0.5/sigma_r)^2, whose Chernoff kernel bound is at most
  w0*epsilon/(2+epsilon); epsilon=1/[16 sigma_r^4(S-1)^2], w0 is the normalized
  spatial centre weight. The resulting joint output bound is R*epsilon.
  Two separate convolutions per polynomial term (2N scalar filterings); the
  self-bilateral N+1 optimization is inapplicable. Stable factorial recurrence,
  streamed accumulators. This is our reference implementation, not optimized
  author software. Oracle-test it before freezing.
- AbsRel, RMSE, delta1 and edge AbsRel use the same fixed masks. Edge mask:
  adjacent valid GT log-depth jumps >0.15, dilated to a 7x7 neighbourhood and
  intersected with validity. Equal frame weights within sequence/environment,
  then equal group weights. Paired group bootstrap, 3000 resamples, seed
  20261009. Report both positive and negative changes; no significance claims.
- Figures use the first manifest frame of each dataset for both models,
  sigma_r=0.2 and S=32, regardless of scores. Shared depth/error scales within
  rows. Show RGB, measured/reference depth, baseline, exact filter, cached
  filter and absolute cache-minus-exact error. Invalid GT is visibly masked.
- Whole-image CPU costs: first frame of each dataset, both models, both range
  sigmas and all three S values (24 configurations), one CPU thread. Include
  construction, allocations, metadata and complete output for uniform/GPA;
  direct oracle includes all its work. Two warmups plus seven measured repeats,
  cyclic paired arm order. Record raw durations, logical snapshot bytes and
  spatial-filter counts; do not claim GPU inference or optimized-filter speedup.
- Save original input hashes, portable decoded inputs, raw FP32 predictions,
  all per-frame scores and bounds, fixed qualitative arrays, environment and
  source hashes. Independently replay representative results with the archived
  PyTorch filter and explicit GPA kernel oracle. Retain old negative outcomes.

Primary sources: https://arxiv.org/html/1603.08109 ;
https://tartanair.org/modalities.html ;
https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/ ;
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats .

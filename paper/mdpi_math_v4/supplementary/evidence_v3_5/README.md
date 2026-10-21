# V3.5 portable review evidence

PLAN.md and freeze.json predate new model predictions and task scores.
arrays/ contains 36 selected decoded RGB/GT pairs, 72 raw FP32 predictions,
and four fixed qualitative outputs. Original datasets/model weights are not
included. Provenance and checkpoint hashes are retained. results.json stores
all 432 configurations per method; summary.json gives equal-group summaries
and paired descriptive intervals; timings.json includes 24 x 7 raw paired
repeat sets with complete construction and readout costs.

Offline replay (Python with NumPy, SciPy, PyTorch, Pillow; no CUDA or network):

    python code/scripts/replay_math_v3_5.py --output replay_result.json

This reproduces all 72 baseline metrics, six native filter configurations
against archived PyTorch and explicit GPA oracles, 24 summary groups,
288 paired intervals and timing medians. New inference requires the frozen
public checkpoints and source datasets; evaluate_math_v3_5.py documents the
stages. Recorded source paths are the original execution provenance.

The mathematical statements and proofs are unchanged from V3.4. These
computational checks do not constitute external expert proof review.
PRIOR_ART_V3_5.md describes primary-formula sources and the unavailable
original An derivation. Historical evidence directories retain their own
versions, protocols and source hashes.

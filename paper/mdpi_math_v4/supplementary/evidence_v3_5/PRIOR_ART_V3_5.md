# V3.5 prior-art sources and comparison record

The V3.4 evidence records are preserved unchanged. This version adds a joint GPA
comparison based on the published filter formulas and does not claim priority
for the existing interpolation upper bound.

## Scope checked against the published text

- Chaudhury & Dabhade, IEEE TIP 25(6), 2519–2528 (2016),
  DOI 10.1109/TIP.2016.2548363.
  https://arxiv.org/html/1603.08109
- Eq. (7): Gaussian × truncated exponential polynomial kernel.
- Eq. (13), (15): Poisson tail and Chernoff kernel error bound, N > lambda.
- Proposition III.2 / Eq. (31), Corollary III.3 / Eq. (32): centre spatial
  weight provides a denominator bound for self bilateral output error.
- Eq. (34): An et al. (2015) component estimate is reproduced for the
  8-bit setting. It is an indirect formula source for An, not a substitute
  for verification of An's original derivation.
- Algorithm 2: self-bilateral uses N+1 scalar spatial filterings and six
  working images excluding input/output. Our independent depth-value
  extension instead needs numerator and denominator filterings separately,
  or 2N. S14 derives the required joint denominator and output bound.

The new reference shares the exact finite Gaussian kernel, replicated
boundaries, FP64 precision and sufficient output-error budget with the
two-read snapshot implementation. It is our implementation, not the
authors' optimized MATLAB software. Runtime results support only this
controlled reference comparison. The lower-bound novelty remains the
shared-atom obstruction for the explicit four-key family.

## An (2015): full-text access rechecked — 2026-10-09

DOI: https://doi.org/10.1109/LSP.2014.2353694

- OpenAlex DOI work API reports `is_oa=false`, `oa_url=null`, no repository
  full text. `openalex.json` retains the response.
- Semantic Scholar DOI API reports `openAccessPdf.status=CLOSED` and an
  empty URL. `semantic.json` retains the response.
- Curtin's migrated record:
  https://curate.curtin.edu.au/articles/journal_contribution/Quantitative_error_analysis_of_bilateral_filtering/31679041
  Public API https://api.figshare.com/v2/articles/31679041 reports
  `is_metadata_record=true`, `files=[]`, `has_linked_file=false`.
  `curtin_article.json` retains the response.
- UWA UUID 53442627-8a9e-495f-975d-4645cefe731d and Murdoch's author-institution
  publication record confirm bibliographic metadata but yielded no legal
  public original PDF. The author publication listing also exposed no PDF
  for this item. No author contact or paid-access bypass was attempted.

Consequently the manuscript states the original derivation is unverified;
it gives the available common-unit comparison in S14, avoiding repeated
access-status prose in the main comparison table.

## New dataset-format sources

- Bonn official dataset and citation:
  https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/
  Registered RGB/depth, TUM-compatible storage.
- TUM official file format:
  https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
  PNG uint16 / 5000 metres, zero invalid.
- TartanAir V2 official modality description:
  https://tartanair.org/modalities.html
  640×640 native images; aligned depth stored as four-byte PNG; decode
  cv2.IMREAD_UNCHANGED then view('<f4'). New evaluator reorders PIL RGBA
  to OpenCV BGRA and validates bitwise equivalence before prediction.

The original data files and model weights are not bundled. The local review
packet contains selected decoded arrays and raw predictions, with provenance;
external distribution must follow the original dataset/model terms.

# V3.4 source comparison — 2026-10-09

Main Table `tab:related-resources` compares target set, norm, stored state,
query accesses and guarantee. The existing full-original DT1997/Tem1998
source notes remain unchanged in `supplementary/source_notes/`.

| Source | Primary location | Checked content / limit |
|---|---|---|
| DeVore–Temlyakov 1997 | [DOI](https://doi.org/10.1006/jcom.1997.0458); retained publisher-PDF audit | p.492 dictionary entropy, p.493 positive-volume hypothesis; curve has zero 4-volume |
| Temlyakov 1998 | [MathNet original](https://www.mathnet.ru/php/archive.phtml?jrnid=mzm&option_lang=eng&paperid=1360&wshow=paper) | Independent subspaces, no common dictionary requirement; original audit retained |
| Temlyakov survey | [Author-hosted PDF](https://www.math.ucdavis.edu/~saito/data/hadc/temlyakov-nonlin.pdf) | Section 9 defines nonlinear widths and their relation to dictionary errors |
| Attention rank | [arXiv v2](https://arxiv.org/abs/2608.28150v2) | Maximum row-L1 approximation rank; coefficients may all be read |
| Coresets | [Author-deposited paper](https://arxiv.org/abs/2605.05602) | Norm-bounded query coreset size; different query/resource model |
| Streaming | [Author-deposited paper](https://arxiv.org/abs/2606.07205) | Stream-space bounds, rather than fixed two-cell reads |
| An et al. 2015 | [Murdoch author-institution record](https://researchportal.murdoch.edu.au/esploro/outputs/journalArticle/Quantitative-error-analysis-of-bilateral-filtering/991005545322407891) | Scope verified; original full formulas unavailable from institutional/publisher access |
| Chaudhury–Dabhade 2016 | [Full author-deposited text](https://arxiv.org/html/1603.08109) | Proposition III.2 / eq.(31), Corollary III.3 / eq.(32), eqs.(33)–(34), reference [16] |

The last source supplies the full kernel-to-output comparison and reproduces
an estimate attributed to An et al. in equation (34). This is a formula
comparison through a published reproduction, not a claim that An's own
derivation or its hypotheses have been inspected. Original-source access
attempts did not produce a PDF; no author contact or access-control bypass
was used. No numerical constant or priority dominance is asserted.

For the shared self-bilateral intensity convention f in [-T,T], the
manuscript derives its OWN sufficient budget directly from the many-key
bound: query width 2T, key span 2T/sigma_r^2, value radius T. The output
bound is T^5/[sigma_r^4(S-1)^2], so absolute error eta is guaranteed by
S >= 1+ceil(T^(5/2)/(sigma_r^2 sqrt(eta))). This conversion preserves
units and separates a sufficient bound from an approximate component
estimate. It also shows why the earlier interpolation work cannot be
replaced by a claim of improved universal constants.

An original-source equation audit and independent expert priority assessment
remain open. Copyrighted article files are not added to the evidence packet.

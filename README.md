# TRAC — Two-Read Attention Caches

Code and verification records for the manuscript

> H. Kim and Y. Son, *Two-Read Linear Caches for Softmax Attention:
> A Shared-Atom Lower Bound and Applications* (main text and Supplementary
> Materials, Sections S1–S6).

The main result is a shared-atom lower bound for two-read linear caches on a
specific four-key softmax family, and its separation from an independent-plane
relaxation. The depth-estimation and image-filtering experiments are
applications of the constructive upper bound.

The `tworead` package contains only the code behind the claims of the manuscript:
the theorem and proposition checks (Sections 2–3, Supplementary S2) and the
representative diagnostics (Section 5, Supplementary S3–S5).

- **[docs/CLAIMS.md](docs/CLAIMS.md)** — claim-by-claim map from the
  manuscript (theorems, propositions, lemmas, figures, tables, supplementary
  sections) to the module, command, record file and expected value.
- **[docs/VALIDATION.md](docs/VALIDATION.md)** — what was recomputed with this
  package and how it compares with the archived records and the manuscript.

## Layout

| Path | Contents | Manuscript |
|---|---|---|
| `tworead/cache/scalar.py` | Scalar-query two-read cache, variance envelope, certified adaptive nodes | Prop. 3–4, Sec. 4.1 |
| `tworead/cache/bilateral.py` | Two-read joint bilateral filter with stable normalization | Cor. 1, Sec. 4.2 |
| `tworead/cache/legacy_bilateral.py` | Earlier implementation with a denominator floor, kept for the regression comparison | Sec. 5.1 |
| `tworead/cache/reference.py` | NumPy references: direct sum, uniform cache, joint GPA | Sec. 5.1, 5.3; Supp. S4 |
| `tworead/theory/constants.py` | Exact interval-arithmetic certificate of the constants (36 exact/symbolic + 8 numerical) | Thm. 1, Prop. 1; Sec. 3; Supp. S2 |
| `tworead/theory/geometry_audit.py` | Exact checks of the geometric boundary cases | Lemmas 3–5, Sec. 2.2; Supp. S2 |
| `tworead/theory/rates.py` | Upper construction of Theorem 1 on a dense query grid (slope −2.05) | Thm. 1; Sec. 3; Fig. 4 |
| `tworead/theory/relaxation.py` | Independent-plane relaxation library (slope −4.07) and separation at S = 461 | Prop. 2, Sec. 2.4; Fig. 4 |
| `tworead/experiments/variance.py` | NYUv2 adaptive storage, stable normalization, range embedding | Sec. 5.1; Prop. 5; Supp. S3 |
| `tworead/experiments/kodak.py` | Kodak storage and lifetime costs | Sec. 5.1; Table 1; Fig. 8; Supp. S3 |
| `tworead/experiments/depth.py` | Fresh depth inference, filter fidelity, matched complete costs | Sec. 5.2–5.3; Tables 2–3; Figs. 9–10; Supp. S4–S5 |
| `tworead/cases.py` | Rebuilds the dataset cases used in the manuscript and verifies them | Sec. 5; Supp. S3–S6 |
| `cases/` | Case manifest (member order, archived SHA-256) and the non-dataset arrays of those files | — |
| `scripts/` | The eight programs named in Supplementary Table S1; each runs the corresponding `tworead` module with the same arguments | Supp. S6 |
| `tests/check_core.py` | One-second self-check of the core claims | — |
| `validation/` | JSON records recomputed with this package | see `docs/VALIDATION.md` |
| `paper/mdpi_math_v4/supplementary/` | Frozen evidence named in Supplementary Table S1 | Supp. S6 |
| `paper/validation.json` | Editorial validation record of the manuscript package (formal statements, proofs, tables and figures preserved; references and layout checked). The submitted text also corrects the Kodak error/tolerance in Sec. 5.1 from 0.772 to 0.771, matching Table 1 and the records | Supp. S6 |

## Environment

Tested with Python 3.11, NumPy 1.26.4, SciPy, PyTorch 2.8.0, SymPy, mpmath and
Pillow. Everything except `depth predict` runs on CPU. `depth predict` also
needs `transformers` and a GPU.

```bash
pip install -e .            # or: pip install numpy scipy torch sympy mpmath pillow
pip install -e .[cases]     # adds h5py, only for rebuilding the NYUv2 cases
pip install -e .[inference] # only for depth predict
```

All commands are run from the repository root.

## Theory checks (no data)

```bash
python tests/check_core.py                                         # ~1 s
python -m tworead.theory.constants      --output out/constants.json  # ~40 s
python -m tworead.theory.geometry_audit --output out/geometry.json
python -m tworead.theory.rates          --output out/rates.json
python -m tworead.theory.relaxation     --output out/relaxation.json
```

## Datasets

The diagnostics of Section 5 use four public datasets. Their pixels and depths
are **not redistributed** in this repository. Only the cases used in the
manuscript are needed, and `tworead.cases` extracts exactly those cases from
your own copy of each dataset.

| Dataset | Used in | Cases used | What to download |
|---|---|---|---|
| Kodak PCD0992 ([r0k.us/graphics/kodak](https://r0k.us/graphics/kodak/)) | Sec. 5.1, Table 1, Fig. 8 | all 24 images, 16 fixed 13×13 neighbourhoods each | nothing: `--download-kodak` fetches the images from the public mirror `MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite` at the frozen commit `dd2e1105fe8d9afe1120cd7e4c0e5e41f509a81b`; or pass a folder with `kodim01.png` … `kodim24.png` |
| NYU Depth V2 ([dataset page](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)) | Sec. 5.1 | 24 labeled test images (every 27th row of the 654-image test split), 16 neighbourhoods each | `nyu_depth_v2_labeled.mat` (about 2.8 GiB, SHA-256 `2d724b0c…cb86aa`) |
| Bonn RGB-D Dynamic ([dataset page](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)) | Sec. 5.2–5.3, Tables 2–3, Figs. 9–10 | 16 frames: two per sequence from `balloon`, `balloon2`, `balloon_tracking`, `balloon_tracking2`, `crowd`, `crowd2`, `crowd3`, `kidnapping_box` | those eight `rgbd_bonn_*` sequences, unpacked into one folder |
| TartanAir V2 ([tartanair.org](https://tartanair.org/)) | Sec. 5.2–5.3, Tables 2–3, Figs. 9–10 | 20 frames: two per environment from `AbandonedFactory`, `AmusementPark`, `Downtown`, `Hospital`, `JapaneseAlley`, `ModularNeighborhood`, `Office`, `OldTownFall`, `SeasonalForestSpring`, `Supermarket` | for each environment, `Data_easy/P002` with `image_lcam_front` and `depth_lcam_front` |

The exact frame IDs, image indices and source-file SHA-256 values are in the
frozen records (`evidence_v3_5/freeze.json`, `evidence_v3_2/freeze.json`,
`evidence_v3_4/freeze.json` under `paper/mdpi_math_v4/supplementary/`). Use of
each dataset is governed by its own terms.

### Rebuild and verify the cases

```bash
python -m tworead.cases build --download-kodak \
    --nyu-mat path/to/nyu_depth_v2_labeled.mat \
    --bonn path/to/rgbd_bonn_dataset \
    --tartan path/to/tartanair_v2
python -m tworead.cases verify --output out/cases.json
```

Any subset of the dataset options can be given; `verify` reports the cases
not yet built. `build` checks every source file against its frozen SHA-256,
extracts only the frames and neighbourhoods listed above, merges them with
the non-dataset arrays in `cases/derived/`, and keeps a rebuilt file only if
it is **byte-identical** to the archived file (SHA-256 in `cases/manifest.json`
and `paper/mdpi_math_v4/supplementary/SHA256SUMS`). The 43 rebuilt files are
written back into the evidence tree and are ignored by git. `verify` then runs
the Kodak and depth replays on the completed evidence; the expected output is
[`validation/cases.json`](validation/cases.json). Rebuilding takes about a minute
after the downloads; the replays take about half a minute.

## Replay from archived evidence (no weights or GPU)

The frozen evidence cited in Supplementary Section S6 is in
[`paper/mdpi_math_v4/supplementary/`](paper/mdpi_math_v4/supplementary/README.md)
(contents, provenance and checksums in its README). After the cases are
rebuilt:

```bash
E=paper/mdpi_math_v4/supplementary
python -m tworead.experiments.kodak replay --evidence $E/evidence_v3_4 --output out/kodak_replay.json
python -m tworead.experiments.depth replay --evidence $E/evidence_v3_5 --output out/depth_replay.json
python -m tworead.experiments.depth test   --out out/depth             # no data needed
```

## Full rerun from the original data

Each pipeline first runs `freeze`, which records input and source hashes before
any scoring. Existing results are never overwritten.

```bash
python -m tworead.experiments.variance freeze --out out/nyu --manifest NYU_TEST_MANIFEST.json
python -m tworead.experiments.variance run --out out/nyu && python -m tworead.experiments.variance summarize --out out/nyu

python -m tworead.experiments.kodak download --out out/kodak     # or copy an existing download_manifest.json
python -m tworead.experiments.kodak freeze --out out/kodak
python -m tworead.experiments.kodak run --out out/kodak && python -m tworead.experiments.kodak summarize --out out/kodak

python -m tworead.experiments.depth test    --out out/depth
python -m tworead.experiments.depth freeze  --out out/depth --bonn BONN_ROOT --tartan TARTANAIR_V2_ROOT --models-freeze MODELS.json
python -m tworead.experiments.depth predict --out out/depth          # GPU, needs transformers
python -m tworead.experiments.depth audit   --out out/depth
python -m tworead.experiments.depth time    --out out/depth
python -m tworead.experiments.depth summary --out out/depth
```

`MODELS.json` holds `models.da2_small` and `models.zoedepth`, each with a
`local_path` and the checkpoint file hashes. Pretrained weights (Depth
Anything V2 Small, ZoeDepth) are public and are not redistributed here.

## Differences from the original research scripts

The computation is the code frozen for the manuscript experiments. Only
input/output handling, function structure and comments were changed:

- Output locations are taken from `--output`/`--out` instead of fixed `work_dirs/...` paths.
- Dataset and checkpoint paths are taken from `--manifest`, `--bonn`, `--tartan` and `--models-freeze` instead of being hard-coded.
- Source hashes recorded by `freeze` refer to the files of this package. The original also hashed its `PLAN.md`; that is omitted.
- The top-level code of `constants` and `geometry_audit` is wrapped in `main(output)`, so importing does not run it.
- `geometry_audit` no longer records the hash of an earlier manuscript draft. The checks are unchanged.
- The separation-ratio table in `relaxation` uses the Theorem 1 lower bound of the manuscript,
  `max{1/(12000 S²(1+log S)), 1/(960 S²(1+log S)²)}`. The original used the earlier
  illustrative constant `1/(6·10⁸ S²(1+log S)²)`. The grid computation is unchanged.
- `variance` drops an earlier quantile-experiment comparison (`legacy_quantiles`) that the manuscript does not use.
- Figure generation, manuscript packaging, training and other-model experiments are not included.

Supplementary Table S1 (Section S6) names the research programs by their
original names. Those names exist in `scripts/` and run the corresponding
modules of this package (e.g. `python scripts/verify_improved_constant.py
--output out/constants.json`); the mapping is in
[docs/CLAIMS.md](docs/CLAIMS.md#supplementary-section-s6-program-map).

## License

Code is released under the [MIT License](LICENSE). Documentation, result
records and prediction arrays are released under
[CC BY 4.0](LICENSE-DATA). Dataset pixels and depths are not distributed;
cases rebuilt with `tworead.cases` remain under the original dataset terms.

# Frozen evidence for the manuscript

Supplementary Section S6 of the manuscript states that the evidence
directories are in `paper/mdpi_math_v4/supplementary/` relative to the
repository root. This directory holds the subset named in Supplementary
Table S1 and the rate records named in the S6 text.

| Directory | Supports | Read by |
|---|---|---|
| `evidence_proof_v3_1/` | Thm. 1, Prop. 1 constants (44 checks) and the seven targeted geometric checks; Sec. 3, Supp. S2 | compare with `tworead.theory.constants` |
| `evidence_v3_2/` | NYUv2 variance-adaptive storage, stable normalization, Prop. 5 embedding; Sec. 5.1, Supp. S3 | compare with `tworead.experiments.variance` |
| `evidence_v3_4/` | Geometric boundary audit (Lemmas 3–5); Kodak storage and lifetime costs, Table 1, Fig. 8; Sec. 5.1, Supp. S2–S3 | `python -m tworead.experiments.kodak replay --evidence paper/mdpi_math_v4/supplementary/evidence_v3_4` (after rebuilding the cases) |
| `evidence_v3_5/` | Fresh inference, filter fidelity and matched costs, Tables 2–3, Figs. 9–10; Sec. 5.2–5.3, Supp. S4–S5 | `python -m tworead.experiments.depth replay --evidence paper/mdpi_math_v4/supplementary/evidence_v3_5` (after rebuilding the cases) |
| `verification/` | Constructive rates and separation, Fig. 4 (`two_read_bridge.json`, `relaxation_separation.json`); Sec. 3 | compare with `tworead.theory.rates`, `tworead.theory.relaxation` |

Each evidence directory has its own `README.md`, frozen plan (`PLAN.md`),
input/source hashes (`freeze.json`), raw results and the original code that
produced them (`code/`). Section, table and figure numbers inside those
files refer to the earlier drafts that created them, not to the manuscript;
[docs/CLAIMS.md](../../../docs/CLAIMS.md) gives the manuscript mapping.

## Provenance

Copied on 2026-10-21 from the supplementary evidence package of the research tree
(`SOKKANAEM/paper/mdpi_math_v4/supplementary/`). Every copied file was checked
against that package's `V4_SHA256SUMS`; all 419 matched. `SHA256SUMS` here
lists all 419 archived files with the hashes they have in this repository.

## Dataset-derived arrays (not distributed)

43 files contain pixels or depths of the public datasets and are not part of
this repository: `evidence_v3_2/sampled_families.npz` and
`evidence_v3_2/reporting_v1_archive/sampled_families.npz` (NYUv2 patches),
`evidence_v3_4/families.npz` (Kodak patches), and
`evidence_v3_5/arrays/*_inputs.npz` / `*_qualitative.npz` (Bonn and TartanAir V2
frames). Their non-dataset arrays are kept in `../../../cases/derived/`.
`python -m tworead.cases build` rebuilds them byte for byte from the datasets
(see the repository README, section *Datasets*). Before rebuilding, check the
distributed files with `sha256sum -c --ignore-missing SHA256SUMS`; afterwards
`sha256sum -c SHA256SUMS` covers all 419 files.

Changes from the original package:

- Not distributed: the 43 dataset-derived arrays above (rebuilt by `tworead.cases`).
- Not included: `__pycache__/` directories, and `evidence_v3_2/visual_revision/`
  (figure-restyling build logs and an earlier manuscript-draft
  snapshot; no manuscript claim depends on it, although `evidence_v3_2/README.md`
  still mentions it). Other evidence directories of the original package are
  development history for claims not in the manuscript.
- Three working notes originally written in Korean were translated into
  English. Their content is otherwise unchanged; no record, array or code
  file was modified. Original SHA-256:

  | File | Original SHA-256 |
  |---|---|
  | `evidence_v3_4/REVISION_V3_4.md` | `71855258d567fae77c2843c314b6517b1bd106b5d1e42d809706b41ff1a2762c` |
  | `evidence_v3_5/REVISION_V3_5.md` | `9bb4538f0af297693eb0ff28814890fbff679849b38f38e838dc7fd618062120` |
  | `evidence_v3_5/PRIOR_ART_V3_5.md` | `f8399638b2c0b5401d750ae0fbf552d097c703f418069a6ec4e26aabb36edc32` |

Frozen records keep the absolute paths of the machine they were produced on
(e.g. in `freeze.json`); they are recorded provenance, not paths to configure.

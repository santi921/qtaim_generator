---
title: Stream F - Cross-method noise floors (B1-B5)
type: plan
date: 2026-05-03
stream: F
parent: docs/plans/2026-05-03-analysis-implementation-plan.md
---

# Stream F: Cross-method noise floors

The five noise-floor sub-analyses described in sections 6.2-6.6 of the paper outline. Reads `charge.lmdb`, `bond.lmdb`, and `qtaim.lmdb` per vertical to produce per-element residual distributions, bond-order disagreement statistics, QTAIM internal redundancy correlations, per-vertical noise floors, and high-disagreement chemistry exemplars. Largest analytical surface in the paper.

## Dependencies

- **Stream B (streaming_aggregator)** - required.

## Locked inputs

- LMDB layout, charge schemes (6 independent schemes: Hirshfeld, CM5, ADCH, Becke, Mulliken-ORCA, Loewdin-ORCA). mayer_orca is excluded - ORCA's Mayer population QA column is labelled "Mulliken gross atomic charge" and duplicates mulliken_orca (observed MAR ~2.5e-5 e). Bond-order schemes (universal): mayer_orca (ORCA Mayer), loewdin_orca (ORCA Loewdin), fuzzy_bond (Multiwfn). IBSI (ibsi_bond, Tier 1) and Laplacian (laplacian_bond, Tier 2) included where present. Note: mayer_orca bond orders ARE a distinct scheme (valid); only the mayer_orca charge column is a Mulliken duplicate. "Wiberg" does not exist as a key in bond.lmdb.
- "Noise floor" definition: median absolute residual per descriptor across schemes, on the same structure / atom / pair.
- **ECP outlier note (2026-05-05)**: ADCH and Becke real-space integration breaks on ECP-treated heavy metals (Ca2+, Cu2+, Ba2+, Mo, etc.) because ECPs remove core electrons from the wavefunction. Observed outlier range: -28 e to +18 e on neighbor atoms (N, S) in 5A_elytes and mo_hydrides. Hirshfeld is ECP-resistant (density-ratio based). CM5 charges are valid (correction is additive to Hirshfeld). Recommended handling for the noise-floor table: winsorize ADCH/Becke at +/-5 e and report the outlier fraction per element/vertical as a separate column. Also emit an ECP-affected flag per atom row (element in ECP-treated set) so downstream users can filter.

## What this stream locks

- `qtaim_gen/source/analysis/noise_floors.py` with B1-B5 implementations.
- CLI `analysis-noise-floors --root <vertical_root> --output <noise.parquet> --topk 50`.
- Per-vertical noise-floor schema (parquet): one row per (descriptor, element-or-pair) with median absolute residual, IQR, n_observations, max-disagreement key list.
- B5 exemplar emission schema: (key, atom_or_pair, schemes_compared, residual, descriptor).
- Notebook 04 (`qtaim_gen/notebooks/analysis/04_noise_floors.ipynb`) producing the F5 noise-floor matrix and F6 high-disagreement exemplar gallery.

## What this stream unlocks

- F5 noise-floor matrix and F6 exemplar figure in section 6 of the paper. Does not block other analysis streams.

## Tasks

### B1 - charge-method comparison

| # | Task |
|---|---|
| F1 | Per atom: extract charges from each of the five schemes. Compute pairwise residuals. Aggregate per element. |
| F2 | Per (vertical, element, scheme-pair) median absolute residual + IQR. |

### B2 - bond-order comparison

| # | Task |
|---|---|
| F3 | Per atom pair: extract bond orders from mayer_orca / loewdin_orca / fuzzy_bond (universal set). Compute pairwise residuals. |
| F4 | Per (vertical, element-pair, scheme-pair) disagreement statistics. |

### B3 - QTAIM internal redundancy

| # | Task |
|---|---|
| F5 | For each BCP: extract `rho`, `Laplacian`, `ellipticity`, `delocalization_index` (when present). |
| F6 | Per (vertical, element-pair) compute pairwise Pearson r between BCP descriptors. |

### B4 - per-vertical noise-floor table

| # | Task |
|---|---|
| F7 | Roll up B1-B3 into a single per-(vertical, descriptor) median absolute residual. This is the published "floor". |
| F8 | Emit T4 table source rows. |

### B5 - high-disagreement chemistry

| # | Task |
|---|---|
| F9 | Rank atoms / pairs by total cross-method spread within each descriptor. Emit top `--topk` (default 50) exemplars per (descriptor, element-or-pair) with key + atom_or_pair indices for case studies. |

### Plumbing

| # | Task |
|---|---|
| F10 | CLI `analysis-noise-floors` registered in `pyproject.toml`. |
| F11 | Tests in `tests/test_analysis_noise_floors.py` against four-folder fixture: B1 produces non-zero residuals on multi-atom records; B5 emits at least one exemplar. |
| F12 | Notebook 04: per-element residual heatmap (B1), bond-order disagreement violin (B2), QTAIM redundancy correlation matrix (B3), per-vertical floor bar chart (B4), exemplar gallery (B5). |

## Verification

- `pytest tests/test_analysis_noise_floors.py -v` passes.
- `analysis-noise-floors --root data/OMol4M_lmdbs/rmechdb --output /tmp/nf.parquet` produces per-element and per-pair rows; medians are finite; n_observations > 0 for major elements (C, H, N, O).
- `analysis-noise-floors --root data/OMol4M_lmdbs/mo_hydrides --output /tmp/nf_tm.parquet` produces non-zero residuals for TM elements (Mo, Fe, Pt, ...).
- Notebook 04 renders all five panels without errors.

## Open items not blocking

- Whether to include the dipole alignment row from Stream E in the F5 noise-floor matrix directly (vs as a separate panel). Locked default: include as a row in F5 (per master plan section 6 figure budget). Coordinate with Stream E to share the schema.
- Whether B5 should rank by total spread (current plan) or by per-scheme outlier z-score (alternative). Default to total spread; revisit if exemplar quality is poor.
- Whether to compute correlations in B3 within a vertical only, or pooled across verticals. Default to within-vertical to expose vertical-specific redundancy.

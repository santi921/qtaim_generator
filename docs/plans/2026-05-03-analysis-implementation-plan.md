---
title: Analysis implementation plan for QTAIM-OMol-4M paper (master index)
type: plan
date: 2026-05-03
references:
  - docs/plans/2026-04-28-paper-outline.md
  - /home/santiagovargas/.claude/plans/i-d-like-to-add-logical-pretzel.md
---

# Context

The 2026-04-28 paper outline (with the 2026-05-03 edits) lists six analytical contributions plus one new bond-classification cross-vertical analysis added in this round. None of the analysis code exists yet under `qtaim_gen/source/analysis/`. This plan defines the modules, CLIs, notebooks, comparator data pulls, calibration tests, and workstream split needed to land all of them.

Local dev runs on full-coverage test verticals under `data/OMol4M_lmdbs/`. HPC runs on the full 32-vertical corpus (some analyses additionally run on a train-set uniform sample). The plan is split into eight independent stream files so multiple agents can work in parallel; this file is the master index. Each stream file is self-contained.

# Decisions locked from 2026-05-03 review

- Geometric bond reference: `geom_bonded = (d <= 1.3 * (r_cov_i + r_cov_j))`. Same covalent-radii basis as the neighbor pool, just a tighter multiplier.
- Candidate pair pool for bond classification: `d <= 1.4 * (r_cov_i + r_cov_j)`.
- Bond-order threshold default: 0.5 across mayer_orca / loewdin_orca / fuzzy_bond (universal set in bond.lmdb). "Wiberg" does not exist in the data; "LBO" is Tier-2 only (laplacian_bond). To be calibrated on `geom_orca6` first; escalate to `tm_react` if signal is weak (see Stream D). TODO!
- Code packaging: module + thin CLI per analysis + driver notebook. CLIs registered in `pyproject.toml [project.scripts]`.
- Intermediate aggregation output format: parquet.
- Comparator data home: `data/comparators/<dataset>/raw/`. Plan attempts scriptable downloads; flags any that need manual pulls.
- Cross-dataset descriptor comparison is NOT attempted. Different DFT levels, different partial-charge schemes. Comparator analysis is structural-only (UMAP overlay) plus % canonical-SMILES overlap.
- Streaming aggregator included as `analysis/streaming_aggregator.py`. Thin LMDB wrapper, but every analysis benefits from the same query API and join behavior (B6 needs `charge.lmdb` joined with `orca.lmdb`).

# Stream index

| Stream | Title | File | Depends on | Unlocks |
|---|---|---|---|---|
| A | Comparator dataset pulls | [_A_comparator_pulls.md](2026-05-03-analysis-implementation-plan_A_comparator_pulls.md) | none | G |
| B | Streaming aggregator foundation | [_B_streaming_aggregator.md](2026-05-03-analysis-implementation-plan_B_streaming_aggregator.md) | none | C, D, E, F, G |
| C | Per-vertical census (T1) | [_C_census.md](2026-05-03-analysis-implementation-plan_C_census.md) | B | section 3 T1 deliverable |
| D | Bond agreement (incl. calibration) | [_D_bond_agreement.md](2026-05-03-analysis-implementation-plan_D_bond_agreement.md) | B | T3 sharpening in section 8.1 |
| E2 | Pairwise dipole agreement (B6) | [2026-05-05-analysis-implementation-plan_E2_charge_dipole_comprehensive.md](2026-05-05-analysis-implementation-plan_E2_charge_dipole_comprehensive.md) | B | section 6.7 figure / paragraph (supersedes Stream E) |
| F | Cross-method noise floors (B1-B5) | [_F_noise_floors.md](2026-05-03-analysis-implementation-plan_F_noise_floors.md) | B | F5 / F6 / T4 in section 6 |
| G | Comparator structural embedding | [_G_comparator_embedding.md](2026-05-03-analysis-implementation-plan_G_comparator_embedding.md) | A, B | F4 / T3 in section 5 |
| H | Post-split per-vertical distribution | [_H_post_split_distribution.md](2026-05-03-analysis-implementation-plan_H_post_split_distribution.md) | none | section 8.5 appendix table; train-key subset for G |

# Dependency graph

```
A ----------------+
                  v
B --+--> C        G
    +--> D
    +--> E
    +--> F
    +--> G

H (independent, also feeds train-key subsample to G)
```

Streams A, B, H can start immediately. C, D, E, F unblock as soon as B lands. G needs both A and B.

# Module structure (cross-stream summary)

```
qtaim_gen/source/analysis/
  __init__.py
  streaming_aggregator.py        # Stream B
  census.py                      # Stream C
  dipole_alignment.py            # Stream E
  bond_agreement.py              # Stream D
  noise_floors.py                # Stream F
  comparator_embedding.py        # Stream G
  post_split_distribution.py     # Stream H
  cli/
    __init__.py
    analysis_census.py
    analysis_dipole_alignment.py
    analysis_bond_agreement.py
    analysis_noise_floors.py
    analysis_comparator_embedding.py
    analysis_post_split_distribution.py

qtaim_gen/notebooks/analysis/
  01_census.ipynb                # Stream C
  02_dipole_alignment.ipynb      # Stream E
  03_bond_agreement.ipynb        # Stream D
  03b_bond_threshold_calibration.ipynb  # Stream D calibration
  04_noise_floors.ipynb          # Stream F
  05_comparator_embedding.ipynb  # Stream G
  06_post_split_distribution.ipynb  # Stream H

data/comparators/                # Stream A
  pcqm4mv2/
  qmugs/
  qm7x/
  schnet4aim/

tests/
  test_analysis_streaming_aggregator.py  # Stream B
  test_analysis_census.py                # Stream C
  test_analysis_dipole_alignment.py      # Stream E
  test_analysis_bond_agreement.py        # Stream D
  test_analysis_noise_floors.py          # Stream F
  test_analysis_comparator_embedding.py  # Stream G
  test_analysis_post_split_distribution.py  # Stream H
```

# Test verticals (local development)

Eight full-coverage verticals under `data/OMol4M_lmdbs/`:

`5A_elytes`, `droplet`, `noble_gas`, `noble_gas_compounds`, `rmechdb`, `geom_orca6`, `mo_hydrides`, `geom_orca6/merged`.

Stream-by-stream fixture mapping is documented in each stream file.

# HPC scope

Per-vertical (all 32 verticals): every analysis runs once per vertical. Output one parquet per (analysis, vertical).

Full-corpus or train-set uniform sample: census, bond_agreement, noise_floors, comparator_embedding additionally run on a corpus-wide aggregation. comparator_embedding specifically uses a uniform sample on the train split (default ~50k, configurable) to keep the UMAP cost tractable.

CLIs all accept `--root` so HPC migration is just pointing at the full corpus root and submitting a job per vertical (and one extra job for full-corpus / sampled passes).

# Suggested launch order for parallel agents

1. Day one: launch agents on Streams A (comparator pulls), B (streaming aggregator), H (post-split distribution). All independent.
2. As soon as B lands: launch agents on C (census), D (bond agreement, including calibration), E (dipole alignment), F (noise floors).
3. As soon as A and B both land: launch agent on G (comparator embedding).
4. Final pass once all streams produce parquet: integrate outputs into the paper figures and tables (separate writeup task, not part of this plan).

# Open items not blocking start

- Lock structural-fingerprint choice for comparator embedding (Morgan-r2-1024 default, revisit if structural overlap looks degenerate). Stream G owns.
- Decide whether the disagreement-subset emission from D feeds directly into the eval-protocol T3 sharpening (likely yes; defer wiring until D output is on disk). Stream D owns.
- Decide whether HPC sampled passes use raw uniform or stratified-by-vertical sampling on the train split (default raw uniform; revisit after first run). Stream G owns.

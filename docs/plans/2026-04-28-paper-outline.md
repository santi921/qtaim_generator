---
title: NeurIPS 2026 Evaluations and Datasets track paper outline
type: plan
date: 2026-04-28
---

# Paper outline: QTAIM-OMol-4M dataset and pipeline

## Target

NeurIPS 2026 Evaluations and Datasets track. Submission window ~8 days from 2026-04-28. Track explicitly emphasizes evaluative claims, scope, and assumptions; encourages dataset-only submissions, audits, and methodology papers without requiring novel models.

## Three lead claims (abstract + intro)

1. **Scale**: ~4M structures with multi-method post-DFT descriptors (5+ partial-charge methods, 4+ bond-order methods, full QTAIM topology, fuzzy descriptors, ORCA-derived properties), one to three orders of magnitude beyond prior published descriptor datasets.
2. **Reusable infrastructure**: open HPC pipeline (qtaim_generator) with measured throughput on ALCF and NERSC, sharded LMDB output, end-to-end reproducible from ORCA inputs to graph-ready datasets.
3. **Evaluation protocol with explicit claims**: canonical tasks, composition-ordered splits, 6+ independent held-out chemistry stress tests inheriting OMol25's hardest axes, plus quantified per-descriptor noise floors from cross-method comparison.

## Working title

"OMol-Descriptors-4M: A Multi-Method Post-DFT Descriptor Dataset and Open HPC Pipeline for Reproducible Chemistry ML Evaluation"

(Title can iterate. Lead with descriptors, not QTAIM, since charges, bond orders, and fuzzy descriptors are equal contributions.)

## Three writing conventions to adopt now

1. Never call SPICE, ANI, etc "comparators." They are sources. Comparators are prior descriptor datasets only.
2. Floor baseline is a floor, not a baseline. Reviewers will read "baseline" as a model contribution.
3. Use "quantified noise floors" consistently in abstract, intro, and section 6.

## Page budget (NeurIPS D&B = 9 pages main + unlimited refs + appendix)

| Section | Pages | Maps to deliverable |
|---|---|---|
| 1. Introduction | 1.0 | claims, gap, track-fit |
| 2. Related Work | 0.5 | prior descriptor datasets, sources |
| 3. Dataset | 1.5 | sources, descriptors, per-vertical table, C1' |
| 4. Pipeline as Contribution | 1.0 | F1-F6 |
| 5. Coverage vs Prior Descriptor Datasets | 1.0 | A1, A2 reframed |
| 6. Cross-Method Noise Analysis | 1.5 | B1-B5 |
| 7. Census Highlights | 0.5 | E5, E6 |
| 8. Evaluation Protocol | 1.5 | D1, D2, D3' folded, D5, optional D4 |
| 9. Limitations + Broader Impact | 0.5 | |
| 10. Conclusion | 0.25 | |
| Slack | 0.25 | figure overflow |
| Total | ~9.0 | |

## Section-by-section outline

### 1. Introduction

- Motivating gap: descriptor-based ML in chemistry has been bottlenecked by O(10^3) - O(10^5) labeled datasets, making it impossible to study generalization, transferability, or noise floors at modern ML scale.
- One paragraph each on the three claims.
- Track-fit paragraph: explicit evaluative claims, scope, what the dataset can and cannot evaluate.
- Final paragraph: contributions summary.

### 2. Related Work

- **Prior QM-descriptor datasets**: 5-10 references, smallest to largest, with sizes. Frame the gap honestly.
- **Geometry datasets included in our data**: OMol25, SPICE, ANI-1x, ANI-2x, GEOM, trans1x, etc. Name as sources, not comparators.
- **ML benchmarks for chemistry**: QM9, MD17, MoleculeNet, OE62, etc. Different tasks (energies, forces). Complementary, not overlapping.

### 3. Dataset

- 3.1 Provenance: 34 verticals totaling ~3.99M structures, derived from OMol25's public 4M release. License inheritance.
- 3.2 Descriptor families:
  - Partial charges: Lowdin, MBIS, Hirshfeld, CM5, ADCH, Mayer
  - Bond orders: Mayer, Wiberg, fuzzy, Laplacian-bond-order
  - QTAIM topology: BCPs (rho, Laplacian, ellipticity, energy density), ring/cage CPs, non-nuclear attractors, atomic basins
  - Fuzzy / integration-based: atomic volumes, ESP-derived
  - ORCA-derived globals: energies, orbital energies, dipole, gradient, SCF metadata
- 3.3 Per-vertical breakdown (Table T1).
- 3.4 Storage: LMDB layout per descriptor family, sharding (forward reference to F4).
- 3.5 One-line on corrupt-folder rate (C1').

Figures: F2 dataset composition (sunburst or stacked bar).
Tables: T1 per-vertical breakdown.

### 4. Pipeline as Contribution

- 4.1 Architecture diagram (F3).
- 4.2 Sharded execution and LMDB merge (F4). Brief.
- 4.3 ML-ready post-processing: merge, multi-vertical splitting, train-only scaler fitting. New subsection (added 2026-04-30).
- 4.4 HPC portability: ALCF (PBS), NERSC (Slurm), Tuolumne (Flux fallback to single-node), single-node (F5). One paragraph each. HPC config snippets deferred from this version.
- 4.5 Throughput and parallel scaling (F3). Local single-node sweep ONLY: concurrent workers in {1, 2, 4, 8} crossed with the three calculation tiers (full_set = 0, 1, 2), running the basic test fixture set with ORCA 6.0.0 and Multiwfn 3.8 pinned. Reportable: wallclock cost per folder vs concurrent workers, one curve per tier. Motivation: no published characterization of Multiwfn under intra-node concurrency, so even a clean single-workstation curve is a contribution. To do (paper task): script and run this sweep, capture timings via the existing timings.json provenance, plot F3.
- 4.6 Reproducibility: ORCA 6.0.0 + Multiwfn 3.8 pinning, deterministic conventions (LMDB key derivation, per-tier expected JSON keys, ORCA enum-state-machine parser), end-to-end smoke test = pytest -q on tests/test_files/lmdb_tests/ fixtures. W&B and tracking_db.py intentionally not mentioned in paper (too green).

TODO (paper task, blocks 4.6 claim): add an environment smoke test that exercises the reproducibility claim. Right now we tell readers `pytest -q` is the smoke test; we have not actually validated this from a clean conda env on a fresh machine. Action items: (a) write a `scripts/helpers/smoke_test.sh` (or a `qtaim-gen smoke-test` entrypoint) that creates the conda env from `environment.yml`, installs the package in editable mode, and runs the relevant `tests/test_files/lmdb_tests/` subset end-to-end (json-to-lmdb -> shard merge -> generator-to-embed -> scaler fit/apply); (b) document expected wallclock and disk footprint; (c) verify on a non-development machine (laptop or fresh VM) before submission. If we cannot validate before submission, soften the §4.6 claim to "the existing pytest suite serves as a smoke test" rather than asserting environment reproducibility.

TODO (post-submission, target +1 week after submission): Chemprop-native conversion. Add a `lmdb-to-chemprop` (or similar) script that exports per-vertical merged LMDBs to Chemprop-compatible CSV / pickled feature dicts. Concretely: SMILES (from geometry LMDB or a per-vertical SMILES LMDB) plus per-atom and per-bond descriptor columns drawn from the charge / bond / qtaim / fuzzy / orca families, with the same shared keys we already emit. Goal: a Chemprop user should be able to point at the exported files and train without writing any LMDB-aware code. This is the "portable" path we already gesture at in §4.1; making it real expands the audience beyond `qtaim_embed` users. Defer until after the NeurIPS submission to avoid scope creep on the first version.

Figures: F1 pipeline diagram, F3 throughput / scaling.
Tables: dropped T2 (HPC throughput) since HPC sweep is out of scope for this submission.

### 5. Coverage vs Prior Descriptor Datasets

- 5.1 Reframe: comparison is against prior descriptor releases, not geometry datasets that are inside our data.
- 5.2 Method: project prior descriptor datasets into our descriptor space (per-element charge histograms, BCP property histograms). Compute coverage(D, B) = fraction of B's descriptor distribution within D's support.
- 5.3 Results: coverage table, gap chemistry. Be explicit about which prior datasets we project.

Figures: F4 coverage map.
Tables: T3 prior dataset vs coverage.

### 6. Cross-Method Noise Analysis

The longest analytical section. Core of the model-free narrative.

- 6.1 Methodology: same structures, multiple methods, residual distributions and rank correlations. Define "noise floor" precisely.
- 6.2 B1 charge-method comparison: per-element residual distributions across Lowdin / MBIS / Hirshfeld / CM5 / ADCH.
- 6.3 B2 bond-order comparison: Mayer / Wiberg / fuzzy / LBO disagreement statistics.
- 6.4 B3 QTAIM internal redundancy: rho-BCP, delocalization index, ellipticity correlations.
- 6.5 B4 per-vertical noise floors: a published table that future ML work should not claim to beat.
- 6.6 B5 high-disagreement chemistry: where methods disagree most, with chemical interpretation.

Figures: F5 noise floor matrix (descriptor by element), F6 high-disagreement exemplars.
Tables: T4 noise floors.

### 7. Census Highlights

- 7.1 E5 bond-order histograms by element pair (top 30-50 pairs).
- 7.2 E6 cross-vertical fingerprint comparison: do the same chemistry classes look statistically the same across verticals?

Figures: F7 bond-order histograms or cross-vertical fingerprint, whichever is sharper.

### 8. Evaluation Protocol

Keystone for the track. Reviewers will read this carefully. All counts below are anchored to the 2026-04-28 manifest (3,986,738 ok rows, 100% read success).

#### 8.1 Tasks (D1)

T1 Per-atom partial charge regression. Multi-target: predict ADCH, Lowdin, Mulliken, Hirshfeld, CM5 charges jointly per atom. Report per-method, per-element MAE and Pearson r on the test split. Long-tail per-element table in appendix.

T2 BCP property regression. Targets: rho, Laplacian, ellipticity at every QTAIM bond critical point. Headline metric: per-element-pair MAE on rho; Pearson r per pair; stratified by bond type (single, double, aromatic, metal-ligand) inferred from QTAIM topology.

T3 Bond classification. Binary: does a QTAIM BCP exist between an atom pair within 2x sum-of-covalent-radii. Compared against Mayer / Wiberg / fuzzy agreement. Report macro-F1 plus per-element-pair confusion stats.

TODO (T3 sharpening): bond classification correlates strongly with naive distance cutoffs in the bulk. After bond.lmdb / qtaim.lmdb are populated, mine the disagreement set (QTAIM says bonded but distance does not, or vice versa). The disagreement subset is the discriminating evaluation slice for T3.

#### 8.2 Main split (D2)

Composition-ordered, deterministic. Each `formula_hill` is hashed via `blake2b` into one of N=10 buckets. Buckets 0-7 are train, 8 is val, 9 is test. Realized sizes from the manifest:

- train: 3,189,390 (8 buckets)
- val:     399,690 (bucket 8)
- test:    415,913 (bucket 9)

Bucket-size variance is under 5% of the mean; per-bucket unique-formula count is approximately 112k each, confirming the partition balances composition rather than row count alone.

#### 8.3 Held-out evaluation sets

Independent of the main split. Overlaps allowed. Membership is decided from the manifest (and, where required, from bond / QTAIM data) before any training. Held-out sets are not used for early stopping or hyperparameter selection.

| ID | Set                  | Construction                                                                                                  | Candidate pool                                  | Size           | Status                |
|----|----------------------|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------|----------------|-----------------------|
| H1 | Metal-ligand pairs   | Sample 50-100 (TM, ligand) bond pairs; hold out every structure containing at least one such bond              | 859 (TM, nonmetal) co-occurrence pairs          | ~5k-30k est    | needs bond data       |
| H2 | PDB-TM               | TM-containing structures inside PDB-family verticals (`protein_*`, `pdb_pockets_*`, `pdb_fragments_*`)         | n/a                                             | TBD            | manifest-filterable   |
| H3 | Reactivity           | TS- and reaction-path-adjacent geometries from `rmechdb`, `pmechdb`, `electrolytes_reactivity`, `trans1x`, `tm_react` | n/a                                       | ~330k          | manifest-filterable   |
| H6 | Lanthanide-ligand    | Sample 30-50 (Ln, ligand) bond pairs; same construction as H1                                                  | 419 (Ln, nonmetal) co-occurrence pairs, 15 Ln   | ~5k-15k est    | needs bond data       |
| H7 | Large systems        | `n_atoms > 200`                                                                                                | n/a                                             | 45,232         | manifest-locked       |
| H8 | Weird charges        | `net_charge_abs >= 4`                                                                                          | n/a                                             | 50,618         | manifest-locked       |

H4 (anions) and H5 (cations) reported as appendix-only distributional statistics. Charge sign is not orthogonal to the main split, so they are not standalone held-out tasks.

H1 and H6 follow the OMol25 convention: hold out a small set of specific M-L (or Ln-L) element-pair bonds, then test on every structure containing at least one such pair. This is a sharper generalization signal than removing all has_tm or has_lanthanide rows (which would lose 17.24% and 1.78% of the dataset respectively, and obliterate the vertical representation of `low_spin_23`, `tm_react`, `ml_mo`, etc.).

The candidate pool sizes (859 TM pairs, 419 Ln pairs) come from `metal_nonmetal_pairs.csv`, which is element co-occurrence inside `element_set`, not true bond counts. It is a coarse upper bound: a structure containing both Pt and N is counted as a Pt-N pair regardless of whether a Pt-N bond exists. Final hold-out pair selection (and exact size) is gated on `bond.lmdb` / `qtaim.lmdb` mining to confirm the bond actually appears.

For each held-out set, report the same per-task metrics as the main test split, plus the gap = `held_out_metric - main_test_metric`. The gap is the headline generalization signal.

#### 8.4 Metrics (D3 folded)

Per task: MAE, RMSE, Pearson r. Stratifications:
- T1: per-method, per-element, per-vertical, per-charge-bucket.
- T2: per-element-pair, per-bond-type (covalent / metal-ligand / H-bond), per-vertical.
- T3: macro-F1, per-element-pair confusion, per-distance-bucket recall.

Held-out sets report the same metrics plus the gap to the main test number.

#### 8.5 Methodology and reporting (D5)

Train on the 8 train buckets. Tune on val (bucket 8). Final numbers reported once on test (bucket 9) and on each held-out set, with no further tuning. Held-out sets must not be used for hyperparameter selection or early stopping.

We provide a reference scoring script that emits the full table from a predictions LMDB. TODO: implement `qtaim_gen/source/scripts/helpers/score_predictions.py` with CLI entry `score-predictions --pred <pred_lmdb> --truth <truth_lmdb> --manifest <manifest_dir> --output <table.md>`. Spec lives in the punchlist below; treat as gating before camera-ready, optional for first submission if time-pressed.

The floor numbers from Section 6 (cross-method noise floors) define a lower bound on achievable error. We argue that any model whose error is below the noise floor is overfitting noise rather than learning chemistry, and we ask future work using this benchmark to report the floor alongside model error.

#### 8.6 Optional floor (D4)

Per-element-pair lookup table emitting the median descriptor value for the matching pair from the train bucket. Submitted as a floor, never a baseline. Page-budget contingent.

#### 8.7 What this protocol can and cannot evaluate

Can: descriptor regression generalization across composition; transfer to rare element-pair bond chemistry (M-L, Ln-L); robustness to system size and charge extremes; cross-method bond-classification agreement.

Cannot: force or energy prediction (not labels in this dataset); MD trajectory quality; conformational diversity beyond the source datasets; level-of-theory transferability (functional fixed by OMol25).

#### 8.8 Section 8 punchlist

| Item | Type | Owner | Blocker | Status |
|---|---|---|---|---|
| Lock H8 size: count rows with `net_charge_abs >= 4` | analysis | manifest notebook | none | done: 50,618 |
| Element-pair co-occurrence analysis (TM + Ln) | analysis | manifest notebook | none | done: 859 TM pairs, 419 Ln pairs (`metal_nonmetal_pairs.csv`) |
| Sample 50-100 TM M-L bond pairs for H1 | analysis | bond / QTAIM mining | needs bond.lmdb across verticals | open |
| Sample 30-50 Ln-L bond pairs for H6 | analysis | bond / QTAIM mining | needs bond.lmdb across f-element verticals | open |
| Build H2 PDB-TM filter (vertical in PDB family AND has_tm) and report size | analysis | manifest notebook | none | open |
| Build H3 reactivity union (5 verticals) and report size | analysis | manifest notebook | none | open |
| Mine T3 disagreement subset (QTAIM vs distance) | analysis | bond / QTAIM mining | needs bond.lmdb + qtaim.lmdb | open |
| Implement `score-predictions` reference script | code | new helper | spec only; non-blocking for first submission | open |
| Lock T5 (tasks x splits x metrics) once H1/H6 sizes are known | writing | section 8 | depends on H1/H6 rows above | open |

Figures: optional F8 protocol illustration.
Tables: T5 tasks-by-splits-by-metrics.

### 9. Limitations and Scope

- Level-of-theory transferability: functional choice inherited from OMol25.
- Inherited OMol25 biases: chemistry coverage, geometry sampling.
- What is NOT supported: force prediction, MD, conformational sampling, time-dependent properties.
- Descriptor-method-specific limitations: ECP-related artifacts, basis-set sensitivity.
- Rare chemistry counts: lanthanide, actinide totals from manifest, with honest call-out if small.

### 10. Broader Impact

- Open dataset and pipeline lower the barrier to reproducible chemistry ML.
- Inherits OMol25 license terms, clearly stated.
- Low dual-use risk profile.

### 11. Conclusion

Three sentences. Recap claims. State the dataset DOI.

## Figure budget (8 main figures, hard cap)

| # | Figure | Section | Source data |
|---|---|---|---|
| F1 | Pipeline diagram | 4 | hand-drawn |
| F2 | Dataset composition by vertical | 3 | manifest |
| F3 | Throughput / parallel scaling: local single-node sweep, concurrent workers x tier, wallclock per folder | 4 | local run logs (ORCA 6.0.0 + Multiwfn 3.8 pinned) |
| F4 | Coverage vs prior descriptor datasets | 5 | A1/A2 outputs |
| F5 | Noise floor matrix (descriptor x element) | 6 | B1-B4 |
| F6 | High-disagreement chemistry exemplars | 6 | B5 |
| F7 | Bond-order histograms or cross-vertical fingerprint | 7 | E5/E6 |
| F8 | Optional protocol illustration | 8 | hand-drawn |

If 8 is too many, drop F8 first, then F7 second.

## Table budget (5 main tables)

| # | Table | Section |
|---|---|---|
| T1 | Per-vertical breakdown (counts, corrupt rate, descriptor coverage) | 3 |
| T2 | DROPPED. HPC throughput out of scope for this submission. |  |
| T3 | Coverage vs prior descriptor datasets | 5 |
| T4 | Noise floors per descriptor | 6 |
| T5 | Tasks x splits x metrics | 8 |

## Held-out evaluation sets

Independent sets, overlaps allowed. Built from manifest queries (and bond / QTAIM data where flagged). Sizes anchored to the 2026-04-28 manifest where ready.

| ID | Set | Construction | Size | Source verticals (top 5) |
|---|---|---|---|---|
| H1 | Metal-ligand pairs | Sample 50-100 specific TM-L element-pair bonds (OMol25 convention) from a candidate pool of 859 (TM, nonmetal) co-occurrence pairs; hold out every structure containing at least one | ~5k-30k est | tm_react, ml_mo, electrolytes_reactivity, low_spin_23, electrolytes_redox |
| H2 | PDB-TM | TM inside PDB-family verticals | TBD | pdb_pockets_*, pdb_fragments_*, protein_core, protein_interface, ml_protein_interface |
| H3 | Reactivity | TS / reaction-path geometries | ~330k | rmechdb, pmechdb, electrolytes_reactivity, trans1x, tm_react |
| H6 | Lanthanide-ligand pairs | Sample 30-50 (Ln, ligand) element-pair bonds from a candidate pool of 419 (Ln, nonmetal) co-occurrence pairs; same construction as H1 | ~5k-15k est | tm_react, omol, electrolytes_reactivity, ml_mo, mo_hydrides |
| H7 | Large systems | `n_atoms > 200` | 45,232 | omol, ml_protein_interface, ml_elytes, pdb_pockets_300K, scaled_separations_exp |
| H8 | Weird charges | `net_charge_abs >= 4` | 50,618 | cross-vertical |

H4 (anions) and H5 (cations) appear as appendix-only distributional statistics, not standalone held-out tasks. Whole-vertical hold-outs (`rmechdb`, `mo_hydrides`, `nakb`, `noble_gas_compounds`) dropped: removing whole verticals at OMol25 inheritance level erases too much representation for the size of the dataset; per-pair element hold-outs (H1, H6) carry the same generalization signal with less collateral.

## Appendix / Supplementary

- A. Datasheet for Datasets (Gebru et al template)
- B. License terms (inherited from OMol25)
- C. Reproducibility checklist (NeurIPS template)
- D. Detailed held-out construction logic for H1-H8
- E. Per-method descriptor definitions
- F. HPC config files (NERSC + ALCF Parsl configs)
- G. Per-element-pair extended bond-order histograms
- H. Additional cross-method residual plots
- I. Manifest schema documentation
- J. Code structure overview

## Streaming aggregator (referenced, not implemented yet)

Single harness signature for the analysis notebooks:

```
streaming_aggregator(
  verticals: list[str],
  lmdb_types: list[str],          # charge, qtaim, bond, fuzzy, orca
  per_record_fn: Callable,        # (key, value) -> dict of features
  groupby: list[str],             # e.g. ["element", "vertical"]
) -> DataFrame
```

A1, B1-B5, E5, E6 each become 30-line per_record_fn implementations on top of one streaming pass per LMDB. Implement as a separate plan when section 6 drafting begins.

## What to drop if pages run tight (in order)

1. F8 protocol illustration
2. F7 bond-order histograms (move to appendix)
3. Section 7 census highlights collapses to one paragraph in section 3
4. D4 floor baseline drops entirely
5. H4 / H5 held-out sets drop
6. B5 high-disagreement exemplars compresses to one paragraph

## What MUST stay

- Track-fit framing in introduction
- Per-vertical breakdown table with corrupt rate (T1)
- Coverage analysis vs prior descriptor datasets (section 5)
- Noise floor analysis (section 6)
- Evaluation protocol with held-out sets (section 8)
- Datasheet, license, reproducibility checklist in appendix

## 8-day rhythm (gating items in bold)

| Day | Primary | Secondary |
|---|---|---|
| 1 | **G1 finish orca pipeline, G2 freeze snapshot**, W1 outline (this doc), W3 figure list | Zenodo DOI reservation |
| 2 | **G3 manifest build + run** (~few hours), threshold decisions, H1/H6/H7/H8 construction | begin G4 cross-vertical merge |
| 3 | A1 + A2 reframed comparison, B1, B2 cross-method | E5 + E6 census |
| 4 | B3, B4, B5 noise analyses | C4 if orca.lmdb ready |
| 5 | D1, D2, D5 protocol section + dataset description table | F1, F2 throughput |
| 6 | Drafting: intro, dataset, infra (F3-F6) | revise figures |
| 7 | Drafting: analyses, protocols, limitations, broader impact | G6 datasheet + license |
| 8 | Final pass, reproducibility smoke test, submit | optional D4 floor baseline |

## Open decisions before drafting prose

1. Title: lead with descriptors (broader, more accurate) over QTAIM (narrower).
2. Should T1 include a "primary chemistry tag" column to help reviewers parse the verticals? If yes, add to manifest schema.
3. How many prior descriptor datasets to project in section 5? Suggest 3-5 well-known ones; more dilutes the message.
4. Reviewer-FAQ section in appendix? Untraditional but increasingly common in D&B; preempts predictable questions.
5. Hosting wording: "DOI reserved on Zenodo, full bytes hosted on ALCF (upload in progress as of submission)". Confirm this phrasing is acceptable to your institution.

## Iteration plan

This outline is a living doc. Iterate section by section in conversation: pick a section, draft prose, revise, lock figures and tables for that section, move to next. Recommended order:

1. Section 8 (evaluation protocol) - highest reviewer scrutiny, design first
2. Section 6 (noise analysis) - core analytical contribution
3. Section 3 (dataset) + Section 4 (pipeline) - factual, lowest risk
4. Section 5 (coverage) - depends on prior dataset selection
5. Section 7 (census) - shortest, do last
6. Sections 1, 2, 9, 10 - prose-heavy, draft after analytical sections lock

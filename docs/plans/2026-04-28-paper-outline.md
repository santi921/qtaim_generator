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
| 6. Cross-Method Noise Analysis | 1.5 | B1-B6 |
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
- Track-fit paragraph: explicit evaluative claims, scope, what is in scope for v1 and what is deferred (e.g., forces and energies to v2).
- Final paragraph: contributions summary.

### 2. Related Work

- **Prior QM-descriptor datasets**: 5-10 references, smallest to largest, with sizes. Frame the gap honestly.
- **Geometry datasets included in our data**: OMol25, SPICE, ANI-1x, ANI-2x, GEOM, trans1x, etc. Name as sources, not comparators.
- **Add mention of QM9 , PubChemQC, QM40, ANI1 but that they aren't being measured here because they are pretty much in OMol.**
- **ML benchmarks for chemistry**: QM9, MD17, MoleculeNet, OE62, etc. Different tasks (energies, forces). Complementary, not overlapping.

### 3. Dataset

- 3.1 Provenance: 34 verticals totaling ~3.99M structures, derived from OMol25's public 4M release. License inheritance.
- 3.2 Descriptor families:
  - Partial charges: Lowdin, MBIS, Hirshfeld, CM5, ADCH, Mayer
  - Bond orders: Mayer, Wiberg, fuzzy, Laplacian-bond-order
  - QTAIM topology: BCPs (rho, Laplacian, ellipticity, energy density), ring/cage CPs, non-nuclear attractors, atomic basins
  - Fuzzy / integration-based: atomic volumes, ESP-derived
  - ORCA-derived globals: energies, orbital energies, dipole, gradient, SCF metadata
- 3.3 Per-vertical breakdown (Table T1). Columns: `n_structures`, `n_unique_formulas`, `n_partial_charge_records` (sum across charge schemes; not every job carries every scheme), `n_bcps` (QTAIM bond critical points), `n_bonds_total` (across Mayer / Wiberg / fuzzy / LBO), `n_atom_records` (per-atom charge / fuzzy targets), corrupt-folder rate. Counts are read directly off the per-type LMDBs, not extrapolated from the manifest, so a reader can verify e.g. that "5+ partial charge methods on 4M structures" actually means ~20M charge records and not fewer.
- 3.4 Storage: per vertical, eight LMDBs (`structure`, `charge`, `qtaim`, `bond`, `fuzzy`, `other`, `orca`, `timings`) sharing a common key derived from job folder relpath. One sentence per LMDB on which descriptors live there and which converter consumes it (`timings.lmdb` is provenance-only, no converter consumes it). Sharding details forward-reference F4.
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

- 5.1 Reframe: comparison is against prior descriptor releases, not geometry datasets that are inside our data. Locked comparator list:
  - **PCQM4Mv2** - 4M drug-like organics, descriptor breadth = 1 (HOMO-LUMO gap). Wide chemistry, narrow descriptor axis.
  - **QM7-X** - 4.2M structures, ~7k unique CNOSCl compositions at <=7 heavy atoms. Narrow chemistry, moderate descriptor breadth (energy decomposition + Hirshfeld + dispersion + forces).
  - **SchNet4AIM source dataset** - the QTAIM-labeled training set used in the SchNet4AIM paper (confirm exact dataset name during drafting).
  - **QMugs** - candidate. Keep only if its structural overlap with OMol25 is small enough to give signal; drop if it is largely a subset.
  - **Santiago check** - check that the format of these datasets is correct, for example that claude doesn't use smiles when xyz is there
- 5.2 Method: descriptor-side projection across datasets is unfair (we publish many more descriptors per structure; no unified descriptor embedding works). Replace with **structural embedding** of the comparator structures that are NOT already inside OMol25:
  - Compute a single structural fingerprint (Morgan / SOAP / one chosen at drafting) per comparator structure and per a representative subsample of our 4M.
  - Visualize via UMAP. The figure shows where each comparator sits relative to the QTAIM-OMol-4M cloud; gaps indicate chemistries we cover that prior descriptor datasets do not, and vice versa.
  - If a comparator is a strict subset of OMol25 by construction, exclude it and state the exclusion explicitly (no signal in re-embedding the same structures).
  - Where a comparator publishes a directly comparable scalar descriptor (e.g., a single Hirshfeld charge), add one per-element residual histogram on the structural overlap. This is method-agreement on shared chemistry, not coverage.
- 5.3 Results: per-comparator row giving (n_structures, % overlap with OMol25 by canonical SMILES, embedding-distance summary, descriptor breadth). Drop the earlier "fraction of B's descriptor distribution within D's support" metric.

Figures: F4 structural UMAP overlay (comparators vs QTAIM-OMol-4M subsample).
Tables: T3 per-comparator structural-overlap and descriptor-breadth row.

### 6. Cross-Method Noise Analysis

The longest analytical section. Core of the model-free narrative.

- 6.1 Methodology: same structures, multiple methods, residual distributions and rank correlations. Define "noise floor" precisely.
- 6.2 B1 charge-method comparison: per-element residual distributions across Lowdin / MBIS / Hirshfeld / CM5 / ADCH.
- 6.3 B2 bond-order comparison: Mayer / Wiberg / fuzzy / LBO / QTAIM disagreement statistics.
- 6.4 B3 QTAIM internal redundancy: rho-BCP, delocalization index, ellipticity correlations.
- 6.5 B4 per-vertical noise floors: a published table that future ML work should not claim to beat.
- 6.6 B5 high-disagreement chemistry: where methods disagree most, with chemical interpretation.
- 6.7 B6 dipole alignment across schemes. Five Multiwfn partial-charge schemes (CM5, ADCH, Becke, Hirshfeld, VDD) emit a charge-derived dipole per job (`dipole_info["mag"]` / `["xyz"]` in `parse_multiwfn.py`), and ORCA emits the DFT dipole (`dipole_au`, `dipole_magnitude_au`, both in `DEFAULT_ORCA_FILTER`). All six are present per job for the full ~4M corpus. We compare magnitude and direction of the charge-derived dipoles against the ORCA reference and against each other, broken down by element composition and net charge. Reportable: (a) which charge scheme tracks the ORCA dipole most closely on real molecules, (b) chemistries where charge-only dipoles diverge from the DFT reference. The analysis is a single streaming pass joining `charge.lmdb` and `orca.lmdb` per vertical, so it is cheap and reuses the streaming aggregator. Page-budget impact: small; B6 piggybacks on F5 as an extra row, no new figure.

Figures: F5 noise floor matrix (descriptor by element, dipole alignment row appended for B6), F6 high-disagreement exemplars.
Tables: T4 noise floors (dipole alignment row included).

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

| ID | Set                  | Construction                                                                                                  | Size       | Status                |
|----|----------------------|----------------------------------------------------------------------------------------------------------------|------------|-----------------------|
| H1 | Metal-ligand pairs   | Stratified-sampled (TM, partner) bond pairs across log-spaced frequency bands (see 8.3.1); hold out every structure containing at least one bond | **15,030** | locked (bond data, 15/15 TM verticals) |
| H3 | Reactivity           | Composition-stratified subsample of TS- and reaction-path-adjacent geometries from `tm_react` (5k) + `electrolytes_reactivity` (5k) + `pmechdb` (2.5k) | **12,507** | manifest-locked       |
| H6 | Lanthanide-ligand    | Same procedure as H1, applied to Ln-(non-TM) bond pairs (`ln_neighbors` extraction across 5 Ln-bearing verticals); shape (3,2,2,1,0) seed=190 | **2,589**  | locked (bond data, 5/5 Ln verticals) |
| H7 | Large systems        | `n_atoms > 250`                                                                                                | **18,200** | manifest-locked       |
| H8 | Weird charges        | `net_charge_abs > 4` (strict, equivalent to `|q| >= 5`)                                                        | **12,393** | manifest-locked       |

H2 (PDB-TM) was dropped: the public OMol25 4M release deliberately excludes metal-containing protein structures from the main training split, so PDB-family verticals contain zero `has_tm=True` structures in this slice. We document this as a limitation in §9; PDB-TM transfer becomes evaluable only against the full OMol25 release.

H4 (anions) and H5 (cations) reported as appendix-only distributional statistics. Charge sign is not orthogonal to the main split, so they are not standalone held-out tasks.

H1 and H6 follow the OMol25 convention of holding out specific M-L bond pairs rather than dropping all has_tm or has_lanthanide rows (which would lose 17.24% and 1.78% of the dataset respectively, and obliterate the vertical representation of `low_spin_23`, `tm_react`, `ml_mo`, etc.). For each held-out set, report the same per-task metrics as the main test split, plus the gap = `held_out_metric - main_test_metric`. The gap is the headline generalization signal.

##### 8.3.1 H1 metal-ligand pair selection procedure

Element-pair co-occurrence (859 TM pairs in `metal_nonmetal_pairs.csv`) was the prior estimate but is a coarse upper bound: a structure containing both Pt and N is counted as a Pt-N pair regardless of whether a Pt-N bond exists. The selection here uses true bonds (Mayer / fuzzy bond order >= 0.3) extracted from `bond.lmdb` by [`tm_neighbor_lists.py`](qtaim_gen/source/scripts/helpers/tm_neighbor_lists.py).

Pipeline:

1. **Per-vertical bond extraction**. For each of 15 TM-bearing verticals, walk `bond.lmdb` and emit one row per (TM atom, partner atom) bond with bond order >= 0.3, written to `tm_neighbors.{csv,parquet}` under `data/OMol4M_lmdbs/tm_bond_lists/<vertical>/{root,merged}/`.
2. **Global pair table**. Concatenate per-vertical files, key each structure by `(vertical, rel_path)`, group by pair `(tm_symbol, partner_symbol)`. Yields **1,101 unique (TM, partner) bond pairs** across **651,742 TM-bonded structures** (16.35% of dataset).
3. **Frequency bands**. Bin pairs by structure count into log-spaced bands B1 [1, 10), B2 [10, 100), B3 [100, 1k), B4 [1k, 10k), B5 [10k, 100k). Band populations: 529 / 93 / 233 / 203 / 43 pairs.
4. **Stratified sampling**. Shape `(3, 2, 2, 4, 0)` with `numpy.random.default_rng(seed=87)`: 3 pairs from B1, 2 from B2, 2 from B3, 4 from B4, none from B5. The head band B5 is excluded because its pairs are dominated by structurally common chemistry (Pt-C, Ti-H, etc.) that any descriptor model will see in training; including it inflates the holdout without adding generalization signal. The B4 count is kept low (4) so the realized holdout stays near 15k structures rather than the much larger numbers that picking many B4 pairs would produce.
5. **Membership materialization**. For each selected pair, take the union of its `(vertical, rel_path)` set. Realized H1 holdout: **11 pairs, 15,030 structures (2.31% of TM-bonded, 0.38% of dataset)**.

Selected pairs per band (seed=87):
- B1 [1, 10): Pt-Na, Rh-Fe, Ag-Pd
- B2 [10, 100): Sc-Ir, Cd-Sb
- B3 [100, 1k): Mn-Sb, Ir-Ge
- B4 [1k, 10k): Tc-Cl, Re-H, Sc-S, Hf-C

Reproducibility:
- Build script: [`qtaim_gen/source/analysis/build_holdout_csvs.py`](qtaim_gen/source/analysis/build_holdout_csvs.py) (CLI: `python -m qtaim_gen.source.analysis.build_holdout_csvs`).
- Outputs: `data/OMol4M_lmdbs/filter_csv_for_holdouts/h1_metal_ligand_pair_definitions.csv` (11 rows, the pair set with band/seed provenance) and `h1_metal_ligand_pairs.csv` (15,030 rows: `holdout_id, vertical, rel_path, matched_pairs, n_matched_pairs`).

Variance: across 30 seeds the realized holdout size for shape `(3, 2, 2, 4, 0)` is 15,227 +/- 4,871 structures (mean 2.34% of TM-bonded). Seed 87 was chosen because it lands within 200 structures of the 15k target. The variance is dominated by B4 pair-size heterogeneity (B4 pairs span 1k-10k structures each); picking a different seed with the same shape would shift the realized count by up to a few thousand structures.

##### 8.3.2 H6 lanthanide-ligand pair selection procedure

Same construction as H1, applied to (Ln, partner) bonds extracted by [`tm_neighbor_lists.py --element_class ln`](qtaim_gen/source/scripts/helpers/tm_neighbor_lists.py). The script is parameterized over element class (`tm`, `ln`, `an`); the file name is unchanged for backward compatibility, with class-specific output prefixes (`ln_neighbors.csv`, `ln_pair_counts.csv`) keeping H1 and H6 artifacts distinct.

Pipeline (anchored to the 2026-05-04 Ln extraction):

1. **Per-vertical extraction**. Five verticals contain lanthanide bonds: `omol`, `tm_react`, `ml_mo`, `electrolytes_reactivity`, `mo_hydrides`. Output written to `data/OMol4M_lmdbs/ln_bond_lists/<vertical>/{root,merged}/ln_neighbors.csv`.
2. **Global pair table**. **453 unique (Ln, partner) bond pairs** across **66,677 Ln-bonded structures** (1.67% of dataset). All 15 lanthanides (Ce, Dy, Er, Eu, Gd, Ho, La, Lu, Nd, Pm, Pr, Sm, Tb, Tm, Yb) are represented as the metal side of at least one bond.
3. **Frequency bands**. Same log-spaced edges as H1. Band populations: B1=207, B2=102, B3=86, B4=58, **B5=0** (max pair size = 4,439). H6 effectively samples from four bands.
4. **Stratified sampling**. Shape `(3, 2, 2, 1, 0)` with `numpy.random.default_rng(seed=190)`: 3 from B1, 2 from B2, 2 from B3, 1 from B4. The shape mirrors H1's (3, 2, 2, 4, 0) but tones the B4 contribution down to 1 since Ln B4 has fewer pairs (58 vs TM's 203) and the goal was a holdout sized at ~2.5k structures.
5. **Membership materialization**. Realized H6: **8 pairs, 2,589 structures (3.88% of Ln-bonded, 0.065% of dataset)**.

Selected pairs per band (seed=190):
- B1 [1, 10): Lu-Ni, Ce-La, Gd-Sb
- B2 [10, 100): Ho-F, Yb-Si
- B3 [100, 1k): Dy-S, Nd-S
- B4 [1k, 10k): Tm-N

Reproducibility:
- Build script: same as H1, [`qtaim_gen/source/analysis/build_holdout_csvs.py`](qtaim_gen/source/analysis/build_holdout_csvs.py); H6 path uses `--ln_bond_root` (default `data/OMol4M_lmdbs/ln_bond_lists`).
- Outputs: `h6_lanthanide_ligand_pair_definitions.csv` (8 rows) and `h6_lanthanide_ligand_pairs.csv` (2,589 rows: same schema as H1).

#### 8.4 Metrics (D3 folded)

Per task: MAE, RMSE, Pearson r. Stratifications:
- T1: per-method, per-element, per-vertical, per-charge-bucket.
- T2: per-element-pair, per-bond-type (covalent / metal-ligand / H-bond), per-vertical.
- T3: macro-F1, per-element-pair confusion, per-distance-bucket recall.

Held-out sets report the same metrics plus the gap to the main test number.

#### 8.5 Methodology and reporting (D5)

Train on the 8 train buckets. Tune on val (bucket 8). Final numbers reported once on test (bucket 9) and on each held-out set, with no further tuning. Held-out sets must not be used for hyperparameter selection or early stopping.

A reference scoring script is intentionally deferred to v2. Publishing scoring infrastructure for v1 without a reference model that produces non-trivial signal exposes the protocol to "garbage-in" adoption, where any number can be plugged into the script. We commit to releasing `score-predictions` in v2 alongside a reference model whose error sits between the cross-method noise floor and naive baselines, so the script ships with a calibrated reference point.

The floor numbers from Section 6 (cross-method noise floors) define a lower bound on achievable error. We argue that any model whose error is below the noise floor is overfitting noise rather than learning chemistry, and we ask future work using this benchmark to report the floor alongside model error.

Composition-ordered splitting is global by `formula_hill`, so verticals are NOT sampled uniformly across train / val / test. We commit to publishing a per-vertical post-split summary (notebook + appendix table) showing realized train / val / test counts and the implied vertical-level coverage gap. This is a transparency artifact: a reader can see which verticals are under-represented in test and weight conclusions accordingly. Implementation: a new analysis notebook driven by the existing manifest plus the `multi_vertical_merge` plan-phase output; no new core code required.

#### 8.6 Optional floor (D4)

Per-element-pair lookup table emitting the median descriptor value for the matching pair from the train bucket. Submitted as a floor, never a baseline. Page-budget contingent.

#### 8.7 What this protocol evaluates and what is out of scope for v1

Evaluates: descriptor regression generalization across composition; transfer to rare element-pair bond chemistry (M-L, Ln-L); robustness to system size and charge extremes; cross-method bond-classification agreement.

Out of scope for v1: force and energy prediction. The OMol25 release publishes forces and energies on the same structure keys; merging them into our descriptor LMDBs by canonical key is mechanically straightforward but pulls the dataset into the OMol25 model evaluation ecosystem (UMA, MACE-MP, etc.), and we want v1 to stand on descriptors alone. v2 will release a forces+energies+descriptors merged variant. Also out of scope: MD trajectory quality, conformational diversity beyond the source datasets, and level-of-theory transferability (functional fixed by OMol25).

#### 8.8 Section 8 punchlist

| Item | Type | Owner | Blocker | Status |
|---|---|---|---|---|
| Lock H8 size with strict cutoff `net_charge_abs > 4` | analysis | manifest pass | none | done: 12,393 |
| Element-pair co-occurrence analysis (TM + Ln) | analysis | manifest notebook | none | done: 859 TM pairs, 419 Ln pairs (`metal_nonmetal_pairs.csv`) |
| Sample TM M-L bond pairs for H1 (stratified shape (3,2,2,4,0), seed=87) | analysis | bond mining + notebook | none | done: 11 pairs, 15,030 structures (2.31% TM-bonded). Pair set in 8.3.1 |
| Build H3 composition-stratified reactivity CSV (`tm_react` 5k + `electrolytes_reactivity` 5k + `pmechdb` 2.5k) | analysis | manifest pass | none | done: 12,507 |
| Build H7 large-system CSV (`n_atoms > 250`) | analysis | manifest pass | none | done: 18,200 |
| Build H8 weird-charge CSV (`net_charge_abs > 4`, strict) | analysis | manifest pass | none | done: 12,393 |
| Drop H2 PDB-TM and document as a §9 limitation | writing | section 9 | none | done |
| Run `tm_neighbor_lists.py --element_class ln` and sample Ln-L pairs for H6 (stratified shape (3,2,2,1,0), seed=190) | analysis | bond mining + notebook | none | done: 8 pairs, 2,589 structures (3.88% Ln-bonded). Pair set in 8.3.2 |
| Mine T3 disagreement subset (QTAIM vs distance) | analysis | bond / QTAIM mining | needs bond.lmdb + qtaim.lmdb | open |
| Implement `score-predictions` reference script | code | new helper | requires reference model with non-trivial signal | deferred to v2 |
| Per-vertical post-split distribution notebook | analysis | manifest notebook | composition split applied to manifest (no LMDB build required) | open |
| Lock T5 (tasks x splits x metrics) | writing | section 8 | none | open (all H sizes locked: H1=15,030, H3=12,507, H6=2,589, H7=18,200, H8=12,393) |

Figures: optional F8 protocol illustration.
Tables: T5 tasks-by-splits-by-metrics.

### 9. Limitations and Scope

- Level-of-theory transferability: functional choice inherited from OMol25.
- Inherited OMol25 biases: chemistry coverage, geometry sampling.
- **PDB-TM transfer is not evaluable on this release.** OMol25 deliberately holds metal-containing protein structures out of the public 4M training split, with the explicit goal of letting users test learning transfer from metal-complex and electrolyte data into protein metal sites. Because we inherit that split, our manifest contains zero `has_tm=True` rows in any PDB-family vertical (`protein_*`, `pdb_pockets_*`, `pdb_fragments_*`, `ml_protein_interface`), which is why we omit a PDB-TM held-out (H2) from §8.3. PDB-TM transfer becomes evaluable only against the full OMol25 release; v2 of this dataset will incorporate PDB-TM structures and add H2 as a held-out set.
- Out of scope for v1: force and energy prediction (OMol25 publishes both on the same structure keys; v2 will release a merged forces+energies+descriptors variant and benchmark the OMol25 model ecosystem on it). Also out of scope for v1: MD, conformational sampling, time-dependent properties.
- Descriptor-method-specific limitations: ECP-related artifacts, basis-set sensitivity.
- Rare chemistry counts: lanthanide, actinide totals from manifest, with honest call-out if small.

### 10. Broader Impact

- **First multi-vertical descriptor dataset at this scale.** ~4M structures with 5+ partial-charge methods, 4+ bond-order methods, full QTAIM topology, fuzzy descriptors, and ORCA globals across 34 chemical verticals. No prior descriptor release has matched both the chemistry breadth and descriptor depth.
- **Open generation pipeline.** Users can rerun the full descriptor stack on their own DFT outputs (ORCA + Multiwfn) without rewriting any post-processing code, lowering the barrier to descriptor-based ML in domains that OMol25 (or any single source dataset) does not cover.
- **Substrate for physics-informed learning beyond forces and energies.** QTAIM topology, partial charges, and bond orders are physically interpretable handles. They support auxiliary supervision for physics-informed neural network potentials and structure-to-property models, providing a path to better generalization without requiring larger geometry-only datasets.
- **Robustness, sample efficiency, interpretability.** Three angles where descriptor-rich data is plausibly more useful than larger geometry-only releases: out-of-domain prediction robustness, sample efficiency on small downstream tasks, and reducing the black-box character of property prediction by giving models access to physically interpretable intermediate quantities.

Low dual-use risk profile.

### 11. Conclusion

Three sentences. Recap claims. State the dataset DOI.

## Figure budget (8 main figures, hard cap)

| # | Figure | Section | Source data |
|---|---|---|---|
| F1 | Pipeline diagram | 4 | hand-drawn |
| F2 | Dataset composition by vertical | 3 | manifest |
| F3 | Throughput / parallel scaling: local single-node sweep, concurrent workers x tier, wallclock per folder | 4 | local run logs (ORCA 6.0.0 + Multiwfn 3.8 pinned) |
| F4 | Structural UMAP: comparators vs QTAIM-OMol-4M subsample | 5 | structural fingerprints (Morgan / SOAP, locked at drafting) |
| F5 | Noise floor matrix (descriptor x element, includes dipole alignment row) | 6 | B1-B4, B6 |
| F6 | High-disagreement chemistry exemplars | 6 | B5 |
| F7 | Bond-order histograms or cross-vertical fingerprint | 7 | E5/E6 |
| F8 | Optional protocol illustration | 8 | hand-drawn |

If 8 is too many, drop F8 first, then F7 second.

## Table budget (5 main tables)

| # | Table | Section |
|---|---|---|
| T1 | Per-vertical breakdown (counts, corrupt rate, descriptor coverage) | 3 |
| T2 | DROPPED. HPC throughput out of scope for this submission. |  |
| T3 | Per-comparator structural overlap + descriptor breadth | 5 |
| T4 | Noise floors per descriptor | 6 |
| T5 | Tasks x splits x metrics | 8 |

## Held-out evaluation sets

Independent sets, overlaps allowed. Built from manifest queries (and bond / QTAIM data where flagged). Sizes anchored to the 2026-04-28 manifest where ready.

| ID | Set | Construction | Size | Source verticals (top 5) |
|---|---|---|---|---|
| H1 | Metal-ligand pairs | 11 (TM, partner) bond pairs sampled stratified across log-spaced frequency bands (shape (3,2,2,4,0), seed=87); see 8.3.1 | **15,030** | omol, tm_react, ml_mo, electrolytes_reactivity, low_spin_23 |
| H3 | Reactivity | Composition-stratified subsample of `tm_react` (5k) + `electrolytes_reactivity` (5k) + `pmechdb` (2.5k); per-vertical formulas chosen by deterministic blake2b hash | **12,507** | tm_react, electrolytes_reactivity, pmechdb |
| H6 | Lanthanide-ligand pairs | 8 (Ln, partner) bond pairs sampled stratified across log-spaced frequency bands (shape (3,2,2,1,0), seed=190); see 8.3.2 | **2,589** | tm_react, omol, ml_mo, electrolytes_reactivity, mo_hydrides |
| H7 | Large systems | `n_atoms > 250` | **18,200** | ml_elytes, pdb_pockets_300K, ml_protein_interface, omol, 5A_elytes |
| H8 | Weird charges | `net_charge_abs > 4` (strict, equivalent to `\|q\| >= 5`) | **12,393** | omol, tm_react, rna, ml_elytes, electrolytes_redox |

H2 (PDB-TM) is not a held-out set in this version; see §9 limitations.

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

A1, B1-B6, E5, E6 each become 30-line per_record_fn implementations on top of one streaming pass per LMDB. B6 (dipole alignment) is the only one that joins two LMDB types in a single pass (`charge.lmdb` + `orca.lmdb`). Implement as a separate plan when section 6 drafting begins.

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
| 4 | B3, B4, B5, B6 noise analyses | C4 if orca.lmdb ready |
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

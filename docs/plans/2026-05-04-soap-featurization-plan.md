---
title: SOAP featurization for OMol / comparator UMAP overlay
type: plan
date: 2026-05-04
parent: docs/plans/2026-05-03-analysis-implementation-plan.md
stream: G (foundation only - this is the descriptor layer that Stream G consumes)
references:
  - https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html
  - data/comparators/README.md
---

# Goal

Compute one SOAP fingerprint per molecular structure for OMol4M and the four
comparators (PCQM4Mv2, QMugs, QM7-X, SchNet4AIM) so that Stream G can run a
single UMAP per (OMol, comparator) pair and overlay the two clouds.

Plain SMILES + Morgan-r2 is already on disk for cheap 2D overlap. SOAP gives
us the 3D-aware overlay called out in the paper's section 5.

# Decisions to lock

1. **dscribe version** — install `dscribe==2.1.2`. Brings `numba>=0.49`,
   `llvmlite`, and `sparse` as transitive deps. ASE and scikit-learn already
   live in the `generator` env. Dry-run was clean (no conflicts).

2. **Element set policy (revised 2026-05-04)** — single global species set
   for all 5 sources (OMol + 4 comparators) so a unified UMAP plot is
   well-defined. Set = comparator union (>= 0.01% atoms in any comparator):

   ```
   GLOBAL_SPECIES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
                  =  H  B  C  N  O  F Si  P  S  Cl Br  I
   S = 12
   SOAP dim @ rcut=5.0 / n=8 / l=6 / sigma=0.5 = 32,592 (dscribe authoritative)
   ```

   Why not Z=1..83 (the OMol element range): SOAP power spectrum dim grows
   as `S(S+1)/2`. At S=83, dim = 1,545,460 -> 309 GB at 50k structures @ float32.
   Untenable. Even at n=4/l=3 (a much weaker descriptor) S=83 still gives
   221k dim / 44 GB. We have to pick a smaller species basis.

   Why the comparator union (not OMol's actual element distribution): the
   comparators are exclusively organic / drug-like / small-molecule chemistry,
   so heavy elements (transition metals, lanthanides) only appear in OMol.
   Including them in `species` adds dim with zero comparator signal. Dropping
   them on the OMol side (i.e. silently skipping OMol atoms whose Z is not
   in the comparator union) is the policy: metal-complex OMol structures
   project as their organic-ligand-only fingerprint, and the gap between
   OMol and comparators in the UMAP is itself the useful signal.

   Per-source histograms (kept here for reference; the species lock above
   is what the CLI defaults to and what every parquet records in metadata):
   - SOAP cost scales with `S * (S + 1) / 2`. OMol's full Z=1..83 set gives
     S=83 -> 3,486 pair channels and roughly 1.56 M descriptor dims at
     n_max=8 / l_max=6. Untenable.
   - For organic comparators we measured S in [4, 11]. With S=11 (PCQM4Mv2
     after trim) we land at 66 * 7 * 64 = 29,568 dims per structure.
     Manageable.
   - Atoms whose Z is outside `species` are silently dropped by dscribe. The
     overlay is meaningful only across the shared element set, which is
     exactly what the union policy enforces.
   - Per-comparator config means each (OMol, comparator) UMAP gets its own
     SOAP descriptor. We do not try to merge SOAP vectors across comparators.
   - Threshold of 0.01% (not 0.05%) keeps Br/Si/P/B in PCQM4Mv2 (each
     0.02-0.04%) because those are real drug-like / catalyst chemistry, not
     pathological singletons. It cuts only Be/Ga/Ge/As/Se in PCQM4Mv2 (each
     <=0.001%, total ~200 atoms across 10M) and Cl in QM7-X (230 atoms).

   | Source | Atom counts | Native Z list (>= 0.01%) | In GLOBAL_SPECIES? |
   |---|---|---|---|
   | pcqm4mv2 | 9,958,362 atoms / 337,861 structures | H,B,C,N,O,F,Si,P,S,Cl,Br | full coverage |
   | qmugs | 11,022,032 atoms / 199,298 SDFs | H,C,N,O,F,P,S,Cl,Br,I | full coverage |
   | qm7x | 7,057,305 atoms / 418,821 conformers | H,C,N,O,S | full coverage |
   | schnet4aim | 98,573 atoms / 4,881 records | H,C,N,O | full coverage |
   | omol (corpus) | TBD - 6 verticals locally, 41,034 structures | Z=1..83 expected | partial: drop heavy/transition metals |

   Dim confirmed via `dscribe.descriptors.SOAP(...).get_number_of_features()`.

   PCQM4Mv2 long-tail dropped at 0.01% threshold: Se (0.00134%),
   Ge (0.00045%), As (0.00026%), Be (0.00002%), Ga (0.00002%). Trim retains
   99.998% of atoms across 337k structures.

3. **SOAP hyperparameters** (lock unless calibration says otherwise):
   - `rcut = 5.0` Å (captures 1st + 2nd coordination shell)
   - `n_max = 8` (radial basis size)
   - `l_max = 6` (angular basis size)
   - `sigma = 0.5` Å (Gaussian width)
   - `rbf = "gto"`
   - `periodic = False`
   - `average = "inner"` (single per-structure descriptor; the standard
     choice for global similarity / UMAP)
   - `sparse = False` for UMAP feed-in (UMAP wants dense float32)

4. **Subsample sizes per source (revised 2026-05-04)**:
   - OMol: 10% per vertical (deterministic seed=42). With the 6 verticals
     locally available (5A_elytes, droplet, mo_hydrides, noble_gas,
     noble_gas_compounds, rmechdb) this is ~4,100 structures total.
     `geom_orca6`, `old`, `tm_bond_lists` lack `structure.lmdb` locally and
     are skipped.
   - PCQM4Mv2: full 337,861 in the on-disk subsample (already 10% of upstream).
   - QMugs: full 199,298 in the on-disk subsample.
   - QM7-X: full ~418,821 in the on-disk subsample.
   - SchNet4AIM: full ~4,884 records.

   Total expected ~960k SOAP descriptors. At 32,592 dims float32 each, the
   raw parquet bytes-on-disk would be ~125 GB. Compression mitigates;
   downstream UMAP needs at most ~50-100k points for a useful overlay so the
   plotting layer applies a final downsample. The featurization layer keeps
   everything; the plotting layer chooses what goes into UMAP.

5. **Output schema** (parquet; one file per source for comparators, one
   per vertical for OMol):
   - `structure_id` (str, dataset-native key)
   - `source` (str, e.g. "omol", "pcqm4mv2")
   - `n_atoms` (int32, atoms before species filter)
   - `n_atoms_kept` (int32, atoms inside GLOBAL_SPECIES; can be 0 -> skipped)
   - `soap` (fixed-size list<float32> of length 32,592, l2-normalized)
   - parquet kv-metadata: species_z, r_cut, n_max, l_max, sigma, rbf,
     average, l2_normalized, source

   File layout:
   ```
   data/comparators/<name>/soap.parquet                     # one per comparator
   data/comparators/omol/soap_<vertical>.parquet            # one per OMol vertical
   ```

6. **Iterative UMAP path** (for later, not in this plan's punchlist):
   - Default at plotting time: load all parquets into memory, downsample each
     source uniformly to a balanced N (e.g. 10k each), fit `umap.UMAP` once
     on the union, color by source.
   - For "add new comparator later" or "add an HPC OMol pass": persist the
     fitted UMAP via `joblib.dump(reducer, ...)`, project new SOAP vectors
     with `reducer.transform(X_new)`. umap-learn supports this natively.
   - Heavier alternative: `parametric_umap.ParametricUMAP` (Keras-backed,
     true online incremental fit). Useful only if we want to ship a reusable
     embedder, not for one-off plot generation.
   - GPU alternative: `cuml.UMAP` (drop-in, ~10x faster on >100k).

6. **Element-set strategy details**:
   - Run `analysis/comparator_embedding/discover_elements.py` once on the
     local data. Produces a json mapping `{dataset: sorted Z list}`.
   - For each (OMol-sample, comparator) pair, take the union and persist it
     in the parquet. SOAP species at compute time = that union, intersected
     with Z=1..83 (we are not benchmarking against beyond-Bi chemistry).
   - Sanity gate: if the comparator produces Z values outside [1, 83], log
     them and proceed with the intersection only. Document in stream G.

# Pipeline

```
analysis/comparator_embedding/
  discover_elements.py     # one-shot. Writes element_sets.json.
  load_omol.py             # iter (key, ase.Atoms) over an OMol vertical
                           # using structure.lmdb. Caps at N=50000.
  load_pcqm4mv2.py         # iter (record_idx, ase.Atoms) over the gzipped SDF.
  load_qmugs.py            # iter (chembl_id/conf, ase.Atoms) over the tarball.
  load_qm7x.py             # iter (mol/conf, ase.Atoms) decompressing shards.
  load_schnet4aim.py       # iter (name, ase.Atoms) over the json + xyz files.
  compute_soap.py          # given a loader + species set + path, write parquet.
  cli.py                   # `analysis-soap-featurize --source <name> --output <path>`
```

CLI entry point (added to `pyproject.toml [project.scripts]`):

```
analysis-soap-featurize = qtaim_gen.source.analysis.comparator_embedding.cli:main
```

Invocations:

```
analysis-soap-featurize --source omol --root data/OMol4M_lmdbs/droplet \
    --species 1,6,7,8,9,15,16,17 --n-sample 50000 --output data/comparators/omol/soap.parquet

analysis-soap-featurize --source pcqm4mv2 \
    --species 1,6,7,8,9,15,16,17 --n-sample 50000 \
    --output data/comparators/pcqm4mv2/soap.parquet
```

# Cost estimate

dscribe SOAP per molecule on CPU is roughly `O(N_atoms * n_max * l_max * S)`.
For an organic molecule (N=50, S=10, n=8, l=6): ~5 ms per structure on a
single core. With `n_jobs=-1` on an 8-core box: ~0.6 ms wall. Half a million
structures across all five sources -> ~5 minutes wall. Memory: 50k * 30k *
4 bytes = ~6 GB peak per source. Process one source at a time.

# Calibration / sanity

Before locking the parquet schema, run a small sanity test on
`tests/test_files/lmdb_tests/` (the four-folder OMol fixture) plus 100
structures from each comparator:

1. SOAP descriptor reproduces under repeated invocation (deterministic).
2. Cosine similarity within the same comparator is higher than across
   comparators (toy UMAP-equivalent check).
3. Empty output if `species` excludes every atom in the structure (with a
   logged warning).

Once those pass, scale to 50k samples per source.

# Open items

- Decide whether SOAP runs per-conformer (one descriptor per record) or
  per-molecule (avg over conformers). Default per-conformer; this matches
  Stream G's "one point per source structure" UMAP convention.
- Whether to z-score or l2-normalize the SOAP vector before UMAP. dscribe
  already normalizes inside the GTO basis; we will l2-normalize per vector
  before UMAP to put all sources on the same scale.
- If install of `numba` causes friction in the `generator` env (it pins
  llvmlite tightly), fall back to `dscribe[no-numba]` extras and document.

# Status of comparator data on disk

| Dataset | Subsample | Records | Notes for SOAP |
|---|---|---|---|
| pcqm4mv2 | 10% (seed=42) | 337,861 V2000 mol blocks | RDKit forward SDF supplier; preserves coords |
| qmugs | 10% (seed=42) | 199,298 SDFs | per-file SDF in tarball; need streaming |
| qm7x | 10%/molecule (seed=42-shard) | ~420k conformers across 6,950 mols | atNUM + atXYZ from HDF5 |
| schnet4aim | full | 3,865 + 1,016 + 3 trajs | ele + pos lists from JSON; ASE for XYZ |
| omol | 50k uniform on train split | per-vertical structure.lmdb | structure.lmdb -> ase.Atoms via existing helpers |

# Punchlist

| # | Task | Status |
|---|---|---|
| 1 | Confirm element sets per comparator (run discover script) | pending - background scan running |
| 2 | Install dscribe in `generator` env | pending |
| 3 | Implement five loaders + compute_soap | pending |
| 4 | Add `analysis-soap-featurize` CLI entry to pyproject | pending |
| 5 | Add sanity tests on small fixtures | pending |
| 6 | Run full 50k featurization per source | pending |

# DDEC6 P7: bond orders, overlap populations and radial moments

Branch `feat/engine-ddec6-bonds`, off `feat/charge-engine` eb946b8 (DDEC6 PR #79 and
IBSI/LBO PR #88 merged; ENGINE_VERSION 8 there). PR target: `feat/charge-engine`.
Status: implemented 2026-11-17, local parity done (section 7); cluster check pending.

## 1. What Chargemol does (09_26_2017, sourcecode_linux)
Called after the spin iterator when `compute_BOs` is true (the default), so every Chargemol
reference we hold already ran it (chargemol.f08:170-176):

| step | module (lines) | what it does |
|-|-|-|
| prepare_BO_density_grids | 199 | Up to 40 passes. Each pass rescales the 5x5x5 points around each nucleus, so each atom's shell-averaged density integrates to its DDEC6 population (core plus valence). It then rebuilds the shell averages from `corrected_total_density` and stops when sum \|dN\| < 1e-5 |
| compute_local_atomic_exchange_vectors | 149 | Per-atom exchange vectors (density, plus spin for open shells) mapped onto the grid; summed grid, min_density per shell, and the dot-product grid |
| initialize_bond_pair_matrix | 209 | Atom pairs within 2 x 500 pm. A pair is kept when determine_whether_pair_overlap_is_significant (59) passes. That test is an upper bound from nA_outside, then an overlap estimate, which uses single-precision `real()` sqrt and nint. Each kept pair gets its shared parallelepiped |
| compute_summed_contact_exchange | in integrate_bonding_terms | Per atom, over its cutoff sphere |
| integrate_bonding_terms | 250 | Per pair, over its parallelepiped: column 12 contact exchange, 13 spin term, 14 overlap population, 15 second-order exchange propensity |
| calculate_final_BOs | 142 | Coordination term (tanh), pairwise term, expansion, then the atom self-exchange constraint. Column 20 is the final bond order; SBO = summed contact exchange + correction |
| compute_atomic_radial_moments | 174 | r^2, r^3, r^4 per atom from `corrected_total_density`. The loop order is na outer, nc inner, unlike the other kernels; it has to stay that way for bitwise output |
| print_overlap_populations | 50 | Column 14 for every kept pair (f17.10, no cutoff) |

Cost: 1.2 s, 3.1 s and 11.7 s for perform_BO_analysis in Chargemol's serial logs for
rmechdb_652, XeCl4 and rmechdb_750. Those runs spent 358-3849 s on the density grids, so
bond orders add well under 1% (`data/ddec6_ref/dumps/*/input.output`).

Parity gotchas found while reading:
- **sum_points.** The collinear spin iterator overwrites the global `sum_points`
  (module_collinear_spin_moments_iterator.f08:60, 89), and prepare_BO_density_grids reads it.
  For open shells the port must pass the spin iterator's last point counts, not the valence
  iterator's. `spin_moments` must also keep `spherical_average_atomic_spin` (`saas`), which
  is local today.
- **Single-precision test.** The pair-significance test does its sqrt and nint in single
  precision; port it with float32.
- **Aligned coordinates.** integrate_bonding_terms uses `coords` and `origin`, so it needs
  the aligned coordinates (the P4 finding: align_periodic_atoms rebuilds them).
- **No pairs.** With zero significant pairs Chargemol returns before calculate_final_BOs, so
  there are no SBOs. The radial moments still run.
- **OpenMP.** Chargemol's OpenMP build reduces integrate_bonding_terms in arbitrary order.
  Bitwise parity is against the serial -O2 build only, as for the charges.

## 2. Port design (core/ddec6.py, new section "P7")
- `bond_orders(state)` runs after `spin_moments` and before `analyze` collects the output.
  It holds the steps in section 1 as numba kernels:
  - per atom in parallel where each atom writes only its own column (shell sums, summed
    contact exchange, radial moments);
  - one pair per task in integrate_bonding_terms (each pair has its own accumulators, so the
    serial order within a pair is kept);
  - serial where atoms share grid cells: the nucleus rescaling, and the exchange-vector
    grid, which is summed over atoms in serial order per cell, as total_pseudodensity is now.
- New state keys: `bond_pairs` (i, j, columns 12-20), `sbo`, `r2`, `r3`, `r4`,
  `overlap_population`, `bo_prep_passes`, `summed_contact_exchange_error` (Chargemol's MAD1
  consistency check).
- `stage_s` gains `bond_orders` and `radial_moments`.
- The memory estimate gains the new grids (corrected density, exchange vectors with 1 or 2
  components, dot product), measured on the 3 local fixtures as for P5. Some earlier grids
  are free by then, so the peak may change less than the grid count suggests.

## 3. Output (proposal, section 6 decisions)
- **bond.json** `ddec6_bond`: `{"i_El_to_j_El": BO}` with i < j (1-based, the existing
  format), for BO >= 0.001 (Chargemol's BO_print_cutoff, so the stored set equals Chargemol's
  printed set). `parse_bond_data` and `bond_list_definition: "ddec6"` work with this key
  unchanged.
- **bond.json** `ddec6_overlap_population` (optional, D2): column 14 for the same pairs.
- **fuzzy_full.json**: `ddec6_sbo`, `ddec6_r2`, `ddec6_r3`, `ddec6_r4` as
  `{atom: value, sum, abs_sum}` (fuzzy_entry), units Bohr^2/3/4.
- **_meta**: `n_bond_pairs`, `bo_prep_passes`, `summed_contact_exchange_error`, and the
  count of BOs the self-exchange constraint changed.
- **Open shells** (optional, D3): bonding spin polarization sqrt(max(c13/c12 - 1, 0)), as
  `ddec6_bond_spin_polarization` in bond.json.

## 4. Runner wiring (omol.py, validation.py)
- `_merge_ddec6`: add bond.json to the files it merges (bond.json is rebuilt on parse like
  charge.json, so the unconditional merge after each parse covers it).
- `_drop_ddec6`: strip `ddec6_*` keys from bond.json as well. fuzzy_full.json is already
  stripped by prefix.
- `validate_ddec6`: when DDEC6 is current, require `ddec6_bond` in bond.json (it may be
  empty, for example for a single atom).
- Engine version bump, `"changed": ["ddec6"]`. Existing DDEC6 results become not current
  and rerun on the next `--ddec6` pass, which forces a restart anyway.
- No new flag: bond orders always run with `--ddec6`, as in Chargemol (D1).

## 5. Validation
Local, bitwise (DDEC6_SLOW). `tests/test_files/ddec6/chargemol/{rmechdb_652_step10_0_2,
XeCl4_step2_0_1}.json` already hold `ddec6_bond`, `ddec6_sbo` and r2/r3/r4 at es25.16, plus
overlap populations at f17.10, from the serial -O2 precise build:
- bond orders: same pair set (5 and 8 pairs) and identical values;
- SBO identical; r2/r3/r4 identical;
- overlap populations equal after rounding to 10 decimals.
Fast unit tests: the float32 significance test, the zero-pair case, and runner merge, drop
and validation on stub folders.

Cluster: one run at the end, port only, no Chargemol reruns.
- The NERSC (96 openac) and LRC (90 OMol) campaigns wrote `ddec6_chargemol.json` per job in
  their work dirs. `run_chargemol.parse_outputs` puts bond orders, SBO, r2-r4 and overlap
  populations in those files. Check they are still there before relying on this.
- Rerun `campaign.py --skip_chargemol` on the same jobs and compare against those files.
  Extend campaign.py and summarize_campaign.py with max_dbo, max_dsbo, max_dr2-4 and pair-set
  agreement. Expected bar about 1e-10 (OpenMP reference).
- Full runner `--ddec6 --restart` on a copy of NERSC `ddec6_l1` (360 computed jobs) to check
  the bond.json merge, validation and data-unchanged counters.

## 6. Decisions (user, 2026-11-17)
- **D1** Bond orders always run with `--ddec6`; no separate flag.
- **D2** Overlap populations are stored in bond.json as `ddec6_overlap_population`, for the
  same pairs as `ddec6_bond`.
- **D3** No bonding spin polarization for now.
- **D4** Pair cutoff BO >= 0.001 (Chargemol's BO_print_cutoff).
- **D5** ENGINE_VERSION 9, `"changed": ["ddec6"]` (IBSI/LBO took 8 and is merged into the base).
- Output schema (bond.json and fuzzy_full.json keys in section 3) approved as proposed.

## 7. Local results (2026-11-17, uncommitted)
- `ddec6.bond_orders` (P7 section of core/ddec6.py) against Chargemol's serial -O2 precise build:
  - rmechdb_652 (open shell): 6 kept pairs, 5 printed;
  - XeCl4 (closed shell, ECP): 10 kept pairs, 8 printed;
  - on both, the printed pair sets are equal and bond orders, SBOs and r2/r3/r4 are bit-identical;
  - overlap populations are equal at Chargemol's 10 printed decimals (max 3.8e-11);
  - 2 nucleus-rescaling passes on both; Chargemol's summed contact exchange check is
    8.3e-14 and 2.2e-13.
- Cost with a warm numba cache: 0.18 s (rmechdb) and 0.15-0.19 s (XeCl4) of 10-17 s runs.
- Peak RSS, 16 threads: rmechdb 0.80 GB (estimate 0.99), XeCl4 0.67 GB (estimate 0.71, base
  0.65). The bond-order grids come after the iterators' peak, so the estimate is unchanged.
- Runner: `_merge_ddec6` merges `bond` into bond.json, `_drop_ddec6` strips `ddec6_*` from
  bond.json, `validate_ddec6(bond_json_loc=...)` requires `ddec6_bond`, ENGINE_VERSION 9.
- campaign.py: `_compare` adds max_dbo, bond-set agreement, max_dsbo, max_dr2-4 and
  max_doverlap; `--reuse_chargemol` compares against the `ddec6_chargemol.json` an earlier run
  left in each job's work dir, so the cluster check needs no Chargemol reruns.
  summarize_campaign.py prints the new columns and any pair-set mismatch.
- Not done: a test with a real single-atom wavefunction. The zero-pair path is covered by a
  synthetic one-atom state (`test_bond_orders_without_pairs_give_zero_sbo_and_radial_moments`).

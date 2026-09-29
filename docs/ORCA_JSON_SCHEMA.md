# orca.json Schema

Output of `parse_orca_output()` in `qtaim_gen/source/core/parse_orca.py`.
Written to `{job_folder}/orca.json` by `write_orca_json()`.

All keys are optional - absent when the corresponding section is not present in the `.out` file
(e.g. `mulliken_spins` only appears for UKS, `s_squared` only when ORCA prints the spin
contamination block). Exception: the orbital-energy keys below form a fixed-width set that is
always complete once an `ORBITAL ENERGIES` block was read.

## Versioning

| Key | Type | Notes |
|-----|------|-------|
| `orca_parser_version` | int | Stamped on every non-empty result. Absent means version 1. Bumped when a key is added or an existing key changes meaning. |

History:

- 1 (implicit, no key): only the `SPIN UP ORBITALS` block was read, so for UKS runs `n_electrons` was the alpha count and `homo_*`/`lumo_*` were alpha values. HOMO/LUMO were the last occupied and first virtual row in file order.
- 2: both spin blocks read; flat `homo_*`/`lumo_*` are the spin-agnostic frontier chosen by energy; per-spin keys always present; zero-energy placeholder rows skipped; `n_electrons_nel` and `hf_type` added.

Consumers should treat rows below the current version as stale. `GeneralConverter` does this by default
via `orca_min_parser_version` in the converter config (stale rows count as missing orca data, so
`missing_data_strategy` applies; set `0` to disable). `oact_utilities` caches `generator_metrics.json`
per job and should gate recompute on this key. Job folders parsed before version 2 need a re-run of
the ORCA parse; the mirrored `orca.tar.zst` is the source when `orca.out` has been cleaned.

## Energy

| Key | Type | Units | Notes |
|-----|------|-------|-------|
| `final_energy_eh` | float | Eh | Total DFT energy. Last occurrence wins (geometry opt). |
| `scf_converged` | bool | - | Always `true` when present. |
| `scf_cycles` | int | - | Number of SCF iterations. |
| `scf_convergence` | dict | - | Final SCF deltas - see below. |
| `energy_components` | dict | Eh | SCF energy decomposition - see below. |

### `scf_convergence` fields

| Key | Type | Units |
|-----|------|-------|
| `energy_change` | float | Eh |
| `max_density_change` | float | - |
| `rms_density_change` | float | - |
| `diis_error` | float | - |

### `energy_components` fields

| Key | Type | Units |
|-----|------|-------|
| `nuclear_repulsion_eh` | float | Eh |
| `electronic_energy_eh` | float | Eh |
| `one_electron_energy_eh` | float | Eh |
| `two_electron_energy_eh` | float | Eh |
| `virial_ratio` | float | - |
| `xc_energy_eh` | float | Eh |
| `nl_energy_eh` | float | Eh | Present when NL dispersion is active. |

## Orbital Energies

Selection is by energy, not file order: HOMO is the highest-energy row with `OCC > 0`, LUMO the
lowest-energy row with `OCC == 0`. Rows printed as `0.0000  0.000000  0.0000` (basis functions
removed for linear dependence) are ignored. For UKS runs both `SPIN UP` and `SPIN DOWN` blocks are
read. For geometry optimizations the last `ORBITAL ENERGIES` block wins.

| Key | Type | Units | Notes |
|-----|------|-------|-------|
| `homo_eh` | float | Eh | Spin-agnostic frontier: highest occupied orbital over both spin channels. |
| `homo_ev` | float | eV | eV of the same row as `homo_eh`. |
| `lumo_eh` | float | Eh | Spin-agnostic frontier: lowest virtual orbital over both spin channels. |
| `lumo_ev` | float | eV | eV of the same row as `lumo_eh`. |
| `homo_lumo_gap_eh` | float | Eh | `lumo_eh - homo_eh`. Negative = convergence to excited state. |
| `homo_eh_alpha`, `homo_ev_alpha`, `lumo_eh_alpha`, `lumo_ev_alpha`, `homo_lumo_gap_eh_alpha` | float | Eh / eV | `SPIN UP` channel. For a single-block run (RHF/RKS/ROHF/ROKS) equal to the flat keys. |
| `homo_eh_beta`, `homo_ev_beta`, `lumo_eh_beta`, `lumo_ev_beta`, `homo_lumo_gap_eh_beta` | float | Eh / eV | `SPIN DOWN` channel. For a single-block run equal to the flat keys. |
| `n_electrons` | float | e | Sum of orbital occupancies over both channels. Equals `n_electrons_nel` for a complete file. |
| `n_electrons_alpha` | float | e | Alpha electrons. Single block: doubly occupied rows count once, singly occupied rows count once. |
| `n_electrons_beta` | float | e | Beta electrons. Single block: doubly occupied rows only. |
| `n_electrons_nel` | int | e | `Number of Electrons NEL` from GENERAL SETTINGS. Exact SCF electron count; differs from the sum of Z under ECPs. |
| `n_orbitals` | int | - | Orbitals per spin channel, excluding placeholder rows. |
| `hf_type` | str | - | `RHF`, `UHF` or `ROHF` from GENERAL SETTINGS. |

A UKS file truncated before the `SPIN DOWN` block is reported as a single block: `hf_type == "UHF"`
with `n_electrons != n_electrons_nel` is the signature.

## Quality Filter Fields

These fields support the OMol25-style quality checks. See collaborator `quality_check()` for usage.

| Key | Type | Units | Notes |
|-----|------|-------|-------|
| `n_alpha` | float | e | Integrated alpha electron count from DFT grid (DFT components block). |
| `n_beta` | float | e | Integrated beta electron count. |
| `n_total` | float | e | Integrated total electron count. Should match `n_electrons_nel` to within ~0.001. |
| `s_squared` | float | - | `<S**2>` expectation value. UKS only. Ideal: `S*(S+1)`. Filter: < 0.5 for metal open-shell, < 1.1 otherwise. |
| `warnings` | list[str] | - | Warning messages from the ORCA `WARNINGS` header block. Each entry is one `WARNING:` message with continuation lines concatenated. Absent if no warnings. |
| `cosx_warning` | bool | - | `true` if `"final exchange deviates considerably"` appears anywhere in the file. Indicates RIJCOSX approximation failure. |

### Electron consistency check (filter 4)

```python
# n_electrons_nel is the exact SCF count; n_electrons_alpha/_beta are the
# per-channel occupation sums (requires orca_parser_version >= 2).
assert n_electrons_nel == n_electrons_alpha + n_electrons_beta
assert abs(n_alpha - n_electrons_alpha) < 0.001   # grid integration error
assert abs(n_beta  - n_electrons_beta)  < 0.001
assert abs(n_total - n_electrons_nel)   < 0.001
```

### COSX warning check (filter 5)

```python
# Via warnings list (matches cclib / collaborator approach):
cosx_fail = any("final exchange deviates considerably" in w.lower()
                for w in result.get("warnings", []))

# Or via convenience flag (also catches inline occurrences):
cosx_fail = result.get("cosx_warning", False)
```

## Partial Charges

Atom keys use the format `"{1-indexed}_{Element}"` (e.g. `"1_O"`, `"53_H"`).

| Key | Type | Notes |
|-----|------|-------|
| `mulliken_charges` | dict[str, float] | |
| `mulliken_spins` | dict[str, float] | UKS only. |
| `loewdin_charges` | dict[str, float] | |
| `loewdin_spins` | dict[str, float] | UKS only. |
| `mayer_charges` | dict[str, float] | From Mayer population analysis QA column. **Warning**: ORCA labels this column "Mulliken gross atomic charge" in its output; it is NOT an independent charge scheme. Values are identical to `mulliken_charges` up to 4-vs-6 digit truncation (observed MAR ~2.5e-5 e). Do not treat as a distinct method. |
| `mayer_population` | dict[str, dict] | Per-atom `{"va": float, "bva": float}`. |
| `hirshfeld_charges` | dict[str, float] | Requires `ALLPOP` or `HIRSHFELD` keyword. |
| `hirshfeld_spins` | dict[str, float] | |
| `mbis_charges` | dict[str, float] | Requires `MBIS` keyword. NaN if MBIS fitting diverged. |
| `mbis_populations` | dict[str, float] | |
| `mbis_spins` | dict[str, float] | NaN if MBIS fitting diverged. |
| `mbis_valence_populations` | dict[str, float] | |
| `mbis_valence_widths` | dict[str, float] | |

## Bond Orders

Bond keys use the format `"{i}_{Elem}_to_{j}_{Elem}"` (e.g. `"1_O_to_2_C"`).

| Key | Type | Notes |
|-----|------|-------|
| `loewdin_bond_orders` | dict[str, float] | |
| `mayer_bond_orders` | dict[str, float] | Only bonds above the Mayer threshold. |

## Gradient

| Key | Type | Units | Notes |
|-----|------|-------|-------|
| `gradient` | dict[str, list[float]] | Eh/Bohr | Per-atom `[gx, gy, gz]`. Present when `EnGrad` or geometry opt is run. |
| `gradient_norm` | float | Eh/Bohr | Norm of the full Cartesian gradient vector. |
| `gradient_rms` | float | Eh/Bohr | RMS gradient. |
| `gradient_max` | float | Eh/Bohr | Max gradient component. Filter: < 50 eV/A (~0.972 Eh/Bohr). |

## Multipole Moments & Rotational Constants

| Key | Type | Units | Notes |
|-----|------|-------|-------|
| `dipole_au` | list[float] | a.u. | `[x, y, z]` |
| `dipole_magnitude_au` | float | a.u. | |
| `quadrupole_au` | list[float] | a.u. | `[xx, yy, zz, xy, xz, yz]` |
| `rotational_constants_cm1` | list[float] | cm^-1 | `[A, B, C]` |

## Timing

| Key | Type | Units |
|-----|------|-------|
| `total_run_time_s` | float | seconds |

## Merge behavior

`write_orca_json()` writes all fields above to `orca.json`.

`merge_orca_into_charge_json()` adds ORCA charge methods into the existing Multiwfn
`charge.json` under new keys: `mulliken_orca`, `loewdin_orca`, `mayer_orca`.
**Note on `mayer_orca`**: this key is stored in charge.json for completeness but is
**not** an independent charge scheme. ORCA's Mayer population QA column is explicitly
labelled "Mulliken gross atomic charge" in the ORCA output and duplicates `mulliken_orca`
up to 4-vs-6 digit precision. It is excluded from all charge-agreement analyses
(`noise_floors.py` `CHARGE_SCHEMES`, `charge_alignment.py` `SCHEMES`).

`merge_orca_into_bond_json()` adds ORCA bond orders into `bond.json` under:
`mayer_orca`, `loewdin_orca`.

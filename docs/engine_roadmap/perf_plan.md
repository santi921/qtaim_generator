# Engine performance plan (full_set 0 and 1)

Branch `perf/engine-phase0` (off feat/charge-engine ff92ee3, ENGINE_VERSION 10). Scope: the
charge, surface and QTAIM engines and the per-job orchestration at full_set 0 and 1. Out of
scope for now: full_set 2 (bader, laplacian_bond, other_esp beyond what a shared kernel gives
it), DDEC6 (own handoff: ddec6_density_grids_perf_handoff.md on perf/ddec6-density-grids) and
GPU work (after phases 0-3, decision P5).

Research of 2026-11-24 (six agents, local workstation, loaded machine, single timings +/- 15-30%):
reports, scripts and raw results in `data/engine_test_2026-11/perf_2026-11-24/` on the
workstation (git-ignored; `reports/*.md`).

## Where the time goes

Size-weighted over the OMol manifest atom-count histogram (3,986,738 jobs, mean 55.3 atoms),
at 4 threads per job, core-seconds = wall x 4. [I] from a measured 10-job ladder (5-342 atoms)
fitted per stage.

| full_set | mean core-s per job | largest stages (share of core-s) |
|---|---|---|
| 0 | 230 | fukui 18.5%, density 17.2%, surface grid 16.1%, QTAIM Newton search 14.3%, fuzzy_bond 8.6%, becke 6.8%, surface bisect 4.8%, ESP at CPs 3.1%, fixed per-process overhead 2.3%, native convert 2.5% |
| 1 | 1068 | chelpg 65.4%, ELF/IBSI derivative pass 7.8%, fukui 4.0%, density 3.7%, mbis_charge 3.5%, surface grid 3.5%, QTAIM search 3.1% |

91% of core-seconds are in jobs above 50 atoms. Thread scaling 1 -> 4 at full_set 0: whole
worker process 1.6-2.9x; serial or poorly scaling: surface polygonize/eliminate, QTAIM bond
paths and Newton search, fuzzy_bond, fukui. At full_set 0 one thread per job gives 1.4-2.5x
more jobs per core-hour than four (derived from separate 1- and 4-thread runs).

## Expected gain [I]

Per-stage factors are prototype measurements; combining them assumes they are independent.

| | bitwise only | + roundoff-level | + value-changing (phase 3) |
|---|---|---|---|
| full_set 0, code | 1.31x | 1.53x | 1.63x |
| full_set 0, x 1 thread per job (1.4-1.6x) | 1.8-2.1x | 2.1-2.4x | 2.3-2.6x |
| full_set 1, code | about 1.05x | 3.5x | 3.75x |

The full_set 1 gain is the CHELPG ESP kernel (65% of the cost, 14-19x in a prototype).

## Decisions (2026-11-24)

| # | Question | Decision |
|---|---|---|
| P1 | Roundoff-level changes (summation order, GEMM partition; at most about 1e-12 relative on stored values) | Allowed without an ENGINE_VERSION `changed` entry; validation compares with that tolerance instead of exact equality. A change above it is value-changing and follows the usual rule |
| P1a | (2026-11-27) Quantities fitted to a roundoff-level input, e.g. CHELPG charges from an ESP within 1e-12 relative (the fit's condition number amplifies it) | Roundoff-level too: up to 1e-9 e on fitted charges, no ENGINE_VERSION entry (ESP kernel: 6.4e-11 e on the 99-job set) |
| P2 | Threads per job on the clusters | 1 thread per job, many jobs per node. Lands with phase 1: in the local layout the native conversion and ORCA parse of every job run in the one parent process under its GIL, which 128 single-thread jobs per node would saturate |
| P3 | Value-changing options (MBIS stalled-check stop, QTAIM seed thinning + capture radius, bond-path step 0.10 Bohr) | Evaluate each: measure how far stored values move, then decide per option |
| P4 | Surface narrow band: point-level (roundoff, about 2.5x on ALIE) or block-level (bitwise, 1.6x) | Measure how much the derived surface values differ under the point-level variant, then decide |
| P5 | GPU | After phases 0-3 |

## Phase 0: per-job overhead (bitwise, this branch)

| # | Item | Measured cost | Change |
|---|---|---|---|
| 0.1 | `esp.HBOUND_W` built at import by 40,010 Python-level kernel calls | 0.67 s per process that imports the engines (part of the 0.94 s `esp.py` import) | store the 11 values; a test recomputes them |
| 0.2 | The Fukui molden read: a header pass over the whole file, then a second line-by-line pass | 0.79 s (53 atoms), 2.1 s (110), 5.4 s (178), 19.7-20.0 s (227, 342), serial | one read; [MO] headers located by regex; coefficient blocks parsed only for the kept orbitals; same float() conversion. The native converter uses the same reader (0.42 s at 53 atoms, 10.9-12.1 s at 227-342) |
| 0.3 | orca.wfx parsed three times per job (QTAIM, charge, surface engines) | 2 extra reads: 0.03 s (36 atoms) to 2-2.8 s (227, 342) | one parse per process, shared read-only |
| 0.4 | Cold numba cache | 18-29 s per process that compiles | dropped: the cache sits next to the sources in a writable checkout, so it is cold once per checkout and CPU type (about 128 x 25 s, one core-hour, per node type and checkout); persistent workers (phase 1) compile once per worker. A checkout on a read-only filesystem would compile in every process: keep checkouts writable or set NUMBA_CACHE_DIR |

Phase 0 results (2026-11-24, local, measured):

- 0.1: `import qtaim_gen.source.core.esp` 1.11 s -> 0.23 s; the values are pinned (equal on this
  machine; test_esp checks them against `_hermite_bound_weights()` to 1e-12 relative).
- 0.2: the reader streams the file in 16 MB chunks (`_molden_scan`): sections ("[") and orbital
  headers ("=") are found with `find` (coefficient lines contain neither), and the byte span of
  each orbital's coefficient lines is recorded; the kept orbitals' spans are then read and parsed
  with the same `float()`/`int()` per token. A block whose lines do not all have two fields, or
  that repeats an index, takes the line-by-line rule. Memory stays at one chunk plus the kept
  coefficients (peak RSS 209 / 605 MB on a 120 / 1954 MB molden in window mode, the old streaming
  reader 175 / 591 MB); no memory map, which would turn a file truncated during the read into
  SIGBUS. Old (git HEAD) and new readers return identical dicts on all 106 distinct local moldens
  (0.1-1954 MB), in every mode (occupied, Fukui window, mwfn_h_compat, keep_virtuals below 200 MB),
  and identical `molden_mo_headers`, also at chunk sizes of 997 B to 64 MB on the fixtures and
  edited variants. Fukui window read: 1.45 -> 0.41 s (120 MB), 7.5 -> 1.6 s (649 MB), 24.5 -> 5.4 s
  (1954 MB). Converter (occupied orbitals): 0.79 -> 0.23 s (120 MB), 13.3 -> 2.8 s (1954 MB).
- 0.3: `omol._load_wfx` (one entry per thread, keyed on the file's path, inode, size and mtime;
  the basis is built on first use, after the QTAIM engine's ECP-without-EDF check);
  `charge_engine.run(load=...)`; `_run_engines` releases it after the in-process engines. The
  engines label the stage `load_shared` when they get the shared parse. A test runs the three
  engines in process and checks one parse, arrays equal to a fresh read, and the release.
- Engine outputs, trunk (c7672fc) vs this branch, in process at 4 threads: every JSON and
  CPprop.txt identical on 5 jobs (12-53 atoms, full_set 0, and 12 atoms at full_set 1; 79 files).
- Test suite: 1725 passed, 83 skipped, 2 failed (test_critic2 end-to-end tests that need the
  git-ignored data/ tree inside the checkout; they fail the same way on trunk).

## Phase 1: persistent workers and 1 thread per job (this branch)

Implemented:

- `--type_runner local` now runs `parsl_configs.local_process_config`: a HighThroughputExecutor on
  a LocalProvider with `n_threads // n_threads_per_job` worker processes, `cpu_affinity="block"`,
  `NUMBA_NUM_THREADS = n_threads_per_job` and single-threaded BLAS per worker. The engines run in
  the worker (no child per folder), and native conversion and the ORCA parse of different folders
  run in parallel instead of under one parent GIL. The interchange and worker pool start through
  the running interpreter, so an env that is not activated works. The old layout (threads of one
  process, an engine child per folder) stays as `--type_runner local_threads`.
- `--n_threads_per_job` defaults to 1 (decision P2). The earlier Multiwfn profiling study also had
  16 x 1 thread ahead of 4 x 4 (206-227 s vs 224-252 s for 16 droplet jobs).
- DDEC6 runs in a child of its own in every layout, also where the other engines run in process
  (its density grids are the likeliest step to be killed for memory, which would take the worker).
- A folder whose worker dies fails alone: the runner logs it, goes on, and exits 1 at the end
  (Parsl fails the task with WorkerLost and replaces the worker). Its `.processing.lock` stays
  until the 8 h stale-lock threshold, as after a preemption.
- Before an engine recomputes, its timings and provenance entries are dropped
  (`omol._drop_engine_records`; permanent skips stay) and written again on success, so a worker
  killed mid-engine leaves nothing that vouches for half-written outputs (the child-process
  layout had this cleanup in `_run_engines` only).
- Workers are capped at the CPUs the process may run on (parsl's block affinity stops every
  worker when there are more workers than CPUs); the runner prints a warning.
- The 1-thread default also applies to `--type_runner hpc` and `flux`, where it sets workers per
  node (for example 256 instead of 64 on a 256-thread ALCF node); batch scripts that want 4 threads
  per job pass `--n_threads_per_job 4`.
- Between jobs: the per-folder logger is torn down by `workflow.teardown_logger` (already), the
  shared wfx is released by `_run_engines` (phase 0). Workers are not recycled: Parsl's
  HighThroughputExecutor has no per-worker task limit, so memory growth is measured first.
- `scripts/nersc/actinide_test.slurm`: 128 jobs x 1 thread, `--cpus-per-task=256` (Perlmutter
  counts logical CPUs; 128 gave 64 physical cores).
- `scripts/perf/throughput_{nersc,lrc}.slurm` + `summarize_throughput.py`: one whole node per
  jobs-per-node x threads mix (NERSC 128x1, 64x2, 32x4, 256x1; LRC lr7 56x1, 28x2, 14x4) on the same
  job list, each into its own tree; jobs per hour over the run and its steady state, seconds and
  core-seconds per job, peak worker and node RSS from a 15 s sampler.

Local check (13 ORCA 6 jobs, 8 workers x 1 thread, full_set 0, both engines): every output identical
between `local_threads` and `local` except the `_meta.qtaim_engine.seconds` timing; wall 157.8 s vs
151.3 s, sum of per-job times 504 s vs 486 s, parent RSS 657 MB vs 108 MB. One 142-147 s job sets
the wall time, so this scale cannot show the parent-GIL limit; the cluster runs measure it.

NERSC Perlmutter CPU node (2026-11-25, array 61997706, branch at 7c9681a): the 440-job openac
sample (X2C actinide complexes), full_set 0, both engines, one whole node per mix. All 440 folders
completed and validated in every mix, and every output (3,520 json and CPprop.txt files) is identical
between the three mixes apart from timing fields.

| mix (workers x threads) | wall s | jobs per node-hour (run / steady) | median s per job | core-s per job (median / mean) | peak worker / node RSS GB |
|---|---|---|---|---|---|
| 128 x 1 | 4102 | 386 / 504 | 499 | 499 / 810 | 1.34 / 75.1 |
| 64 x 2 | 5647 | 281 / 316 | 399 | 798 / 1355 | 1.50 / 47.3 |
| 32 x 4 | 9943 | 159 / 175 | 371 | 1482 / 2524 | 1.75 / 30.7 |

128 x 1 gives 2.4x the jobs per node-hour of 32 x 4 over the whole run and 2.9x in the steady state
(the run figure includes the tail of the last long jobs); a job takes 1.35x longer at 1 thread than
at 4. Peak node memory is 75 GB of 512 GB. Steady state = completions between the 10th and 90th
percentile of completion times.

LRC lr7 node (2026-11-26, array 28217305, branch at 6632da2): the first 168 jobs of the tm_react
slice (engine_test/jobs.txt), full_set 0, both engines, one whole node per mix. All 168 folders
completed in every mix; all 168 validated in the 56 x 1 tree (cross-mix identity not checked on LRC).

| mix (workers x threads) | wall s | jobs per node-hour (run / steady) | median s per job | core-s per job (median / mean) | peak worker / node RSS GB |
|---|---|---|---|---|---|
| 56 x 1 | 1137 | 532 / 678 | 325 | 325 / 307 | 0.81 / 28.3 |
| 28 x 2 | 1704 | 355 / 437 | 264 | 528 / 482 | 0.92 / 17.3 |
| 14 x 4 | 2769 | 218 / 239 | 237 | 947 / 863 | 0.96 / 9.9 |

56 x 1 gives 2.4x the jobs per node-hour of 14 x 4 over the run and 2.8x in the steady state, the
same ratio as on NERSC; a job takes 1.37x longer at 1 thread than at 4.

## Phase 2: kernels (roundoff-level, one routine per PR, cluster validation once per routine)

| # | Item | Prototype | Parity |
|---|---|---|---|
| 2.1 | Exact ESP kernel: pairs outer over 32-64 Morton-sorted points, vectorized asymptotic Boys branch, table branch for near lanes, numba error_model="numpy"; also serves ESP at QTAIM CPs and other_esp | 14-19x on CHELPG ESP | ESP within 7e-14 a.u., CHELPG within 5e-12 e |
| 2.2 | Charge full_set 0 package: Fukui virtuals from the density pass's primitive block, two-phase Becke, column-major primitive kernel | 1.31x charge engine | bitwise |
| 2.3 | Charge full_set 1: one block pass for density, ELF/IBSI derivatives and Fukui (needs the open-shell reductions) | 1.31-1.39x on those stages | roundoff |
| 2.4 | Surface narrow band (P4 decides the variant) | ALIE 1.6x (block) / 2.0-2.85x (point) | bitwise / roundoff; seeding needs the 99-job mesh check |
| 2.5 | QTAIM: drop bitwise-duplicate seeds, one Newton batch across stages, rho/gradient/Hessian-only kernel | 1.28x QTAIM step | roundoff; CP sets identical on 89 jobs |
| 2.6 | MBIS: chord iteration with the analytic Jacobian for the fuzzy refit and for check fits converging to the default's fixed point | fuzzy refit about 10x | within 1.6e-7 e (inside the spread of the three guesses) |

Phase 2.1 results (branch perf/esp-kernel, 2026-11-27): `esp._esp_e_blocked` is the
`boys_mode="exact"` path of `electrostatic_potential` (CHELPG, ESP at QTAIM CPs, other_esp);
"reference" and "multiwfn" keep the per-point kernel. Orders 3-6 are straight-line code in
`core/esp_contract.py`, generated by `scripts/gen_esp_contract.py` (a test checks it is current).
The far-field thresholds `FAR_T` are pinned (a test recomputes them with scipy).

- Against the per-point exact kernel at the same points (local, 1 thread): ESP within 2.8e-14 a.u.,
  8.4e-16 relative, on the 2 fixtures, a 12-atom, a 36-atom and a 39-atom job; 18-26x faster; built
  for Zen 3 (`NUMBA_CPU_NAME=znver3`, no AVX-512) 18-25x and the same differences.
- 99-job set (`chelpg/list99.txt`: 87 wfx_pull, 12 nitrates), old and new kernel in one process:
  CHELPG charges max |dq| 6.35e-11 e (median over jobs 3.1e-13 e; the fit's condition number,
  1e4-1e6, amplifies the 1e-16-level ESP differences); CHELPG 5041 s -> 244 s summed at 1 thread
  (20.6x; 14.9-23.7x per job above 2 s). ESP surface (other_esp): at most 13 fields change per job,
  the largest by 1.07e-11 relative (a 1.0e-4 a.u. average moving 1.1e-15 a.u.), no positive or
  negative area changes; 7367 s -> 730 s (10.1x; the mesh is unchanged). Fields that are NaN (no
  negative or no positive surface) are NaN in both.
- Output independent of thread count and block size (bitwise, tested).
- QTAIM (cp_search.analyze) with the old and new kernel on 12, 36 and 39 atoms: identical CP sets and
  every other property, esp_e within 4.8e-16 relative (esp_total within 5.7e-13, where it nearly
  cancels); the 39-atom QTAIM step 5.1 s -> 3.9 s.

## Phase 3: value-changing options (P3)

| Option | Prototype gain | What to measure before deciding |
|---|---|---|
| MBIS: stop a stalled check fit at about evaluation 200 when its objective is below the best converged fit | selection 2.4-3.7x on jobs with stalled checks | selection flips on the 112-job set and the NERSC openac 440 (actinides untested) |
| QTAIM: drop seeds within 0.2 Bohr of an earlier seed; stop a seed within 0.03 Bohr of a CP of an earlier stage | 1.4-1.8x fewer Newton evaluations | CP-set changes on the LRC tier 2 set; the 0.0115 Bohr BCP-RCP case |
| Bond paths: RK2 step 0.10 Bohr (stop radius unchanged) | 3.3x fewer evaluations | end-CP changes; 0 of 1,671 BCPs so far |

Not pursued: angular pruning of inner shells (changes every stored value), ESP screening at
1e-10 a.u. (6-14% fewer pairs), killing vacuum seeds by density (3-6%), contracted-shell
evaluation (6% fewer primitives per block), SVML (numba dropped it in 0.62).

## Notes for later

- numpy's vector exp is 3.5x faster than numba's scalar exp only with AVX-512 (measured on the
  7950X: 1.18 vs 4.16 ns); without AVX-512 it is 4.78 ns. Perlmutter CPU nodes are Zen 3 (AVX2):
  kernel gains there must come from fewer exp/erf calls and vectorizable loops, not numpy's exp.
- GPU (P5): the orbital-value + density kernel feeds about 71% of full_set 0 today; NERSC GPU
  work goes on account gen0039_g. A Perlmutter GPU node has half the cores of a CPU node at the
  same charge, so the comparison is against the optimized CPU path.

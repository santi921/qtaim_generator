# NERSC actinide test (integration/nersc-actinide)

Multiwfn mode and charge-engine mode from one checkout of this branch, first at
full_set 0. The production env is not modified: the checkout is put ahead of
whatever `qtaim_gen` the env has installed, via PYTHONPATH.

Tools pinned by `actinide_test.slurm`:
- ORCA 6.1.1 `orca_2mkl`. The 6.0.1 one cannot read ORCA 6.1 gbw files: it prints
  "I/O OPERATION FAILED" and still exits 0.
- Multiwfn 3.8 dev.
- Account m3278, CPU nodes.

## 1. Checkout

```bash
module load python
conda activate /pscratch/sd/s/santiago/envs/generator
git clone -b integration/nersc-actinide https://github.com/santi921/qtaim_generator.git /pscratch/sd/s/santiago/qtaim_generator_nersc
python -c "import numba, threadpoolctl, rdkit; print('ok')"   # charge engine and CHELPG radii need these
```

## 2. Job sample (about 500 folders)

```bash
WORK=/pscratch/sd/s/santiago/actinide_test; mkdir -p $WORK && cd $WORK
PYTHONPATH=/pscratch/sd/s/santiago/qtaim_generator_nersc python \
  /pscratch/sd/s/santiago/qtaim_generator_nersc/scripts/nersc/sample_jobs.py \
  --roots <vertical root 1> <vertical root 2> ... --per_vertical 50 --actinide_first --out $WORK/sample.txt
```

Every root must sit under `SRC` (default `/pscratch/sd/s/santiago`); the result tree mirrors
the paths below it.

## 3. Runs

Each array task takes 50 lines, so 500 folders means `--array=0-9`. Each task runs 16 jobs at a
time at 8 threads, on one node.

```bash
cd $WORK
S=/pscratch/sd/s/santiago/qtaim_generator_nersc/scripts/nersc/actinide_test.slurm
sbatch -J mwfn   --export=ALL,LIST=$WORK/sample.txt,TREE=mwfn_l0          --array=0-9 $S
# after the Multiwfn run checks out:
sbatch -J engine --export=ALL,LIST=$WORK/sample.txt,TREE=engine_l0,ENGINE=1 --array=0-9 $S
```

`QTAIM_ENGINE=1` (with `ENGINE=1`) also runs the QTAIM step with the QTAIM engine (`--qtaim_engine`).
`summarize.py` then counts `_meta.qtaim_engine` records.

Add `FULL_SET=1` (and another TREE) for level 1, which adds CHELPG with the actinide radii and MBIS.
In Multiwfn mode, MBIS fails for Z > 86, so level-1 actinide folders do not validate there.

## 4. Check

```bash
PYTHONPATH=/pscratch/sd/s/santiago/qtaim_generator_nersc python \
  /pscratch/sd/s/santiago/qtaim_generator_nersc/scripts/nersc/summarize.py $WORK/mwfn_l0 \
  --list $WORK/sample.txt --src /pscratch/sd/s/santiago --failed $WORK/mwfn_l0_failed.txt
```

The summary gives, for actinide and other folders separately:
- folders and how many validated;
- Poincare-Hopf == 1;
- `far_heavy_nuclei` flags (heavy all-electron nuclei more than 15 Bohr from the origin, where
  Multiwfn can lose the nuclear CP);
- charge-engine stamps and CHELPG radius files;
- timings;
- failed Multiwfn steps by exit status;
- listed folders that produced no output.

## 5. Audit an engine tree against Multiwfn

```bash
PYTHONPATH=$CODE python $CODE/scripts/nersc/audit_prod.py $WORK/prod_l1 --ref $WORK/mwfn_l0 --tsv $WORK/prod_l1_audit.tsv
```

Reads both trees only. Per folder:
- charges summing to the net charge (hirshfeld, adch, cm5, becke; 0.02 e);
- fuzzy spin summing to mult - 1;
- fuzzy densities and the charge engine's grid summing to the all-electron count (1e-3 relative);
- Poincare-Hopf (kept and full set);
- an NCP on every actinide, plus the nucleus-frame and precision-limited counts;
- routines recorded as permanently skipped.

With `--ref`, for folders validated in both trees, it also reports:
- the largest per-atom difference per charge and fuzzy scheme and for fuzzy_bond, against the 0.03 e acceptance;
- the ALIE values;
- QTAIM atom and bond key sets, including actinide NCPs found by only one side.

The TSV has one row per folder with its flags.

## 6. Santiago's benchmark runs (santi_benchmarks)

`stage_benchmarks.py` selects the finished jobs and stages ORCA-opt outputs; it reads the sources only.
- **multi_spin:** each job's latest `multi_spin_jobs.sqlite3` row must have status 1.
- **wave_2:** only `CfK3C22H36OCl4-ZORA_PBE0`, the one job that isn't in multi_spin.
- **gbw:** a non-empty `orca.gbw` is required.
- **ORCA-opt outputs:** the newest `flux-*.out` is copied to `<results>/<rel>/orca.out`.

Results go to their own tree (`prod_l1_santi`), never merged into `prod_l1`.

```bash
SB=/pscratch/sd/s/santiago/santi_benchmarks
python $CODE/scripts/nersc/stage_benchmarks.py --raw $SB/raw --db $SB/raw/multi_spin/multi_spin/multi_spin_jobs.sqlite3 \
  --results $WORK/prod_l1_santi --out_dir $WORK [--dry_run]
sbatch -J santi --export=ALL,CODE=$CODE,SRC=$SB/raw,LIST=$WORK/santi_jobs.txt,TREE=prod_l1_santi,ENGINE=1,QTAIM_ENGINE=1,FULL_SET=0,N=50 \
  --array=0-$(( ($(wc -l < $WORK/santi_jobs.txt) + 49) / 50 - 1 )) $CODE/scripts/nersc/actinide_test.slurm
```

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

#!/bin/bash
# DDEC6 validation on NERSC Perlmutter: the port (core/ddec6.py) against Chargemol on raw
# OpenActinides jobs (campaign.py). Each array task takes N lines of $LIST and runs
# WORKERS jobs at a time with THREADS threads each on one CPU node.
#
#   sbatch -J ddec6 --export=ALL,CODE=$CODE,LIST=$WORK/ddec6_sample.txt --array=0-11 nersc_ddec6.slurm
#
# Optional: N (lines per task, default 8), WORKERS (default 8), THREADS (default 16),
# SRC (input root the list is under), CHARGEMOL (binary), DENSITIES (atomic_densities/),
# ORCA_2MKL, OUT (results JSONL directory), SKIP_CHARGEMOL=1 (port timings only).
#SBATCH -A m3278
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=santiagovargas@lbl.gov
#SBATCH --mail-type=END,FAIL

set -eo pipefail

module load python
conda activate /pscratch/sd/s/santiago/envs/generator
CODE=${CODE:?CODE is required (the feat/engine-ddec6 checkout)}
export PYTHONPATH=$CODE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_STACKSIZE=1G
ulimit -s unlimited

SRC=${SRC:-/pscratch/sd/s/santiago}
WORK=${WORK:-/pscratch/sd/s/santiago/actinide_test}
CHARGEMOL=${CHARGEMOL:-/pscratch/sd/s/santiago/chargemol/bin/chargemol_omp_precise}
DENSITIES=${DENSITIES:-/pscratch/sd/s/santiago/chargemol/chargemol_09_26_2017/atomic_densities/}
ORCA_2MKL=${ORCA_2MKL:-/pscratch/sd/s/santiago/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca_2mkl}
OUT=${OUT:-$WORK/ddec6}
N=${N:-8}
WORKERS=${WORKERS:-8}
THREADS=${THREADS:-16}
LIST=${LIST:?LIST is required}
ID=${SLURM_ARRAY_TASK_ID}
EXTRA=""
[ "${SKIP_CHARGEMOL:-0}" = "1" ] && EXTRA="--skip_chargemol"

python -c "import qtaim_gen; assert qtaim_gen.__file__.startswith('$CODE'), qtaim_gen.__file__"
echo "branch: $(git -C $CODE rev-parse --abbrev-ref HEAD) $(git -C $CODE rev-parse --short HEAD)"
[ -x "$ORCA_2MKL" ] || { echo "missing $ORCA_2MKL"; exit 1; }
[ -n "$EXTRA" ] || [ -x "$CHARGEMOL" ] || { echo "missing $CHARGEMOL"; exit 1; }

mkdir -p $OUT/slices logs
SLICE=$OUT/slices/${SLURM_JOB_NAME}_${ID}.txt
sed -n "$(( ID * N + 1 )),$(( (ID + 1) * N ))p" $LIST > $SLICE
if [ ! -s $SLICE ]; then
  echo "task $ID: empty slice, nothing to do"
  exit 0
fi

export NUMBA_NUM_THREADS=$THREADS
python $CODE/scripts/ddec6_reference/campaign.py \
  --job_file $SLICE --root_omol_inputs $SRC/ --work $OUT/work --out $OUT/${SLURM_JOB_NAME}_${ID}.jsonl \
  --orca_2mkl_cmd $ORCA_2MKL --chargemol_bin $CHARGEMOL --atomic_densities $DENSITIES \
  --workers $WORKERS --threads $THREADS $EXTRA

#!/bin/bash
# NERSC Perlmutter test of the integration/nersc-actinide branch: Multiwfn mode and
# charge-engine mode from the same checkout. Each array task takes a disjoint slice of
# N lines of $LIST and runs them on one CPU node, N_PAR jobs at a time.
#
#   sbatch -J mwfn   --export=ALL,LIST=$WORK/sample.txt,TREE=mwfn_l0         --array=0-9 actinide_test.slurm
#   sbatch -J engine --export=ALL,LIST=$WORK/sample.txt,TREE=engine_l0,ENGINE=1 --array=0-9 actinide_test.slurm
# Optional: FULL_SET (default 0), N (lines per task, default 50), N_PAR (concurrent jobs per
# node, default 16), THREADS (per job, default 8), SRC (input root the list is under),
# MULTIWFN and ORCA_2MKL (tool paths, defaults below).
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
# the branch checkout, imported ahead of whatever qtaim_gen the env has installed
CODE=${CODE:-/pscratch/sd/s/santiago/qtaim_generator_nersc}
export PYTHONPATH=$CODE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_STACKSIZE=4G
export KMP_STACKSIZE=200M
ulimit -s unlimited

ORCA_2MKL=${ORCA_2MKL:-/pscratch/sd/s/santiago/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca_2mkl}
MULTIWFN=${MULTIWFN:-/pscratch/sd/s/santiago/multiwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI}
SRC=${SRC:-/pscratch/sd/s/santiago}
WORK=${WORK:-/pscratch/sd/s/santiago/actinide_test}
N=${N:-50}
N_PAR=${N_PAR:-16}
THREADS=${THREADS:-8}
TREE=${TREE:?TREE is required}
LIST=${LIST:?LIST is required}
ID=${SLURM_ARRAY_TASK_ID}
EXTRA=""
[ "${ENGINE:-0}" = "1" ] && EXTRA="--charge_engine"

python -c "import qtaim_gen, sys; print('qtaim_gen from', qtaim_gen.__file__); assert qtaim_gen.__file__.startswith('$CODE')"
[ "${ENGINE:-0}" = "1" ] && python -c "import numba, threadpoolctl; print('numba', numba.__version__)"
[ -x "$ORCA_2MKL" ] && [ -x "$MULTIWFN" ] || { echo "missing $ORCA_2MKL or $MULTIWFN"; exit 1; }
echo "orca_2mkl: $ORCA_2MKL"
echo "multiwfn:  $MULTIWFN"
echo "branch:    $(git -C $CODE rev-parse --abbrev-ref HEAD) $(git -C $CODE rev-parse --short HEAD)"

mkdir -p $WORK/slices $WORK/$TREE logs
SLICE=$WORK/slices/${TREE}_${ID}.txt
sed -n "$(( ID * N + 1 )),$(( (ID + 1) * N ))p" $LIST > $SLICE
if [ ! -s $SLICE ]; then
  echo "task $ID: empty slice, nothing to do"
  exit 0
fi

python -m qtaim_gen.source.scripts.full_runner_parsl_alcf \
  --num_folders $N \
  --orca_2mkl_cmd $ORCA_2MKL \
  --multiwfn_cmd $MULTIWFN \
  --clean --patch_timings \
  --full_set ${FULL_SET:-0} \
  --n_threads $(( N_PAR * THREADS )) \
  --n_threads_per_job $THREADS \
  --safety_factor 1.0 \
  --move_results --wfx --check_orca \
  --prevalidate $EXTRA \
  --timeout_hr 4 \
  --n_nodes 1 \
  --type_runner local \
  --job_file $SLICE \
  --preprocess_compressed \
  --root_omol_results $WORK/$TREE/ \
  --root_omol_inputs $SRC/

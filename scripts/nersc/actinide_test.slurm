#!/bin/bash
# NERSC Perlmutter test of the integration/nersc-actinide branch: Multiwfn mode and
# charge-engine mode from the same checkout. Each array task takes a disjoint slice of
# N lines of $LIST and runs them on one CPU node, N_PAR jobs at a time.
#
#   sbatch -J mwfn   --export=ALL,CODE=$CODE,LIST=$WORK/sample.txt,TREE=mwfn_l0         --array=0-9 actinide_test.slurm
#   sbatch -J engine --export=ALL,CODE=$CODE,LIST=$WORK/sample.txt,TREE=engine_l0,ENGINE=1 --array=0-9 actinide_test.slurm
#   sbatch -J qeng   --export=ALL,CODE=$CODE,LIST=$WORK/sample.txt,TREE=qengine_l0,ENGINE=1,QTAIM_ENGINE=1 --array=0-9 actinide_test.slurm
# Optional: FULL_SET (default 0), N (lines per task, default 50), N_PAR (concurrent jobs per
# node, default 128), THREADS (per job, default 1), SRC (input root the list is under),
# MULTIWFN and ORCA_2MKL (tool paths, defaults below), NATIVE_CONVERT (the runner's default is
# fixed, core/wavefunction.py; multiwfn_h for parity checks; off runs Multiwfn's convert step,
# needed to continue a tree Multiwfn converted, e.g. prod_l1), FLAGS (extra runner flags, e.g.
# "--restart" on a copy of a finished tree).
#SBATCH -A m3278
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --cpus-per-task=256
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=santiagovargas@lbl.gov
#SBATCH --mail-type=END,FAIL

set -eo pipefail

module load python
conda activate /pscratch/sd/s/santiago/envs/generator
# the branch checkout, imported ahead of whatever qtaim_gen the env has installed
CODE=${CODE:?CODE is required: the branch checkout to import, e.g. /pscratch/sd/s/santiago/qtaim_generator_<name>}
export PYTHONPATH=$CODE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_STACKSIZE=4G
export KMP_STACKSIZE=200M
# one BLAS thread per job (Multiwfn does not use OpenBLAS); --type_runner local runs N_PAR worker
# processes, each pinned to THREADS cores (parsl_configs.local_process_config)
export OPENBLAS_NUM_THREADS=1
ulimit -s unlimited

ORCA_2MKL=${ORCA_2MKL:-/pscratch/sd/s/santiago/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca_2mkl}
MULTIWFN=${MULTIWFN:-/pscratch/sd/s/santiago/multiwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI}
SRC=${SRC:-/pscratch/sd/s/santiago}
WORK=${WORK:-/pscratch/sd/s/santiago/actinide_test}
N=${N:-50}
N_PAR=${N_PAR:-128}
THREADS=${THREADS:-1}
TREE=${TREE:?TREE is required}
LIST=${LIST:?LIST is required}
ID=${SLURM_ARRAY_TASK_ID}
EXTRA=""
[ "${ENGINE:-0}" = "1" ] && EXTRA="--charge_engine"
[ "${QTAIM_ENGINE:-0}" = "1" ] && EXTRA="$EXTRA --qtaim_engine"
[ -n "${NATIVE_CONVERT:-}" ] && EXTRA="$EXTRA --native_convert $NATIVE_CONVERT"
EXTRA="$EXTRA ${FLAGS:-}"

python -c "import qtaim_gen, sys; print('qtaim_gen from', qtaim_gen.__file__); assert qtaim_gen.__file__.startswith('$CODE')"
if [ "${ENGINE:-0}" = "1" ] || [ "${QTAIM_ENGINE:-0}" = "1" ]; then
  python -c "import numba, threadpoolctl; print('numba', numba.__version__)"
fi
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

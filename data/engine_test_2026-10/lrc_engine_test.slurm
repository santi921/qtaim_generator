#!/bin/bash
#SBATCH --account=ac_blau
#SBATCH --partition=lr7
#SBATCH --qos=lr_lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/global/scratch/users/santiagovargas/engine_test/slices/logs/%x_%A_%a.out

# Charge-engine LRC test. MODE selects the tree (all share $WORK/jobs.txt, built by setup.sh):
#   merge   test A: full-runner-engine --restart on copies of finished Multiwfn folders
#   scratch test B: full-runner-engine from the raw inputs into an empty tree
#   ref     same settings, stock full-runner-parsl-alcf (Multiwfn) on jobs_ref.txt, for timing
# Each array task takes a disjoint slice of N lines. Submit, e.g.:
#   sbatch -J merge   --export=ALL,MODE=merge   --array=0-199%50 lrc_engine_test.slurm
#   sbatch -J scratch --export=ALL,MODE=scratch --array=0-199%50 lrc_engine_test.slurm
#   sbatch -J ref     --export=ALL,MODE=ref     --array=0-3      lrc_engine_test.slurm

source ~/.bashrc
module load gcc
conda activate ${CONDA_ENV:-qtaim_engine}   # separate env + checkout, see README
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1   # ignore ~/.local site-packages; only the env decides what is imported
export OMP_STACKSIZE=4G
export KMP_STACKSIZE=200M
ulimit -s unlimited

SRC=/global/scratch/users/santiagovargas/OMol4M_raw
WORK=/global/scratch/users/santiagovargas/engine_test
N=${N:-50}
ID=${SLURM_ARRAY_TASK_ID}
case $MODE in
  merge)   RUNNER=full-runner-engine;     LIST=$WORK/jobs.txt;     EXTRA="--restart" ;;
  scratch) RUNNER=full-runner-engine;     LIST=$WORK/jobs.txt;     EXTRA="" ;;
  ref)     RUNNER=full-runner-parsl-alcf; LIST=$WORK/jobs_ref.txt; EXTRA="" ;;
  *) echo "MODE must be merge, scratch or ref"; exit 1 ;;
esac
SLICE=$WORK/slices/${MODE}_${ID}.txt
sed -n "$(( ID * N + 1 )),$(( (ID + 1) * N ))p" $LIST > $SLICE
if [ ! -s $SLICE ]; then
  echo "task $ID: empty slice, nothing to do"
  exit 0
fi

$RUNNER \
  --num_folders $N \
  --orca_2mkl_cmd /global/scratch/users/santiagovargas/orca_6_0_1_linux_x86-64_shared_openmpi416/orca_2mkl \
  --multiwfn_cmd /global/scratch/users/santiagovargas/Multiwfn/Multiwfn_3.8_dev_bin_Linux/Multiwfn \
  --clean --patch_timings \
  --full_set 0 \
  --n_threads 1 \
  --n_threads_per_job ${SLURM_CPUS_PER_TASK} \
  --safety_factor 1.0 \
  --move_results --wfx --check_orca \
  --prevalidate $EXTRA \
  --timeout_hr 5 \
  --queue workq-route \
  --n_nodes 1 \
  --type_runner local \
  --job_file $SLICE \
  --preprocess_compressed \
  --root_omol_results $WORK/$MODE/ \
  --root_omol_inputs $SRC/

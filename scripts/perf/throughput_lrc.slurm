#!/bin/bash
# Jobs per node-hour for --type_runner local at several jobs-per-node x threads-per-job mixes
# (docs/engine_roadmap/perf_plan.md, phase 1). Each array task is one mix on one whole CPU node,
# running the first N lines of $LIST from the raw inputs into its own tree $WORK/<TAG>_<mix>.
#
#   sbatch -J tput --export=ALL,CODE=$CODE,LIST=$WORK/jobs.txt,TAG=tput --array=0-2 throughput_lrc.slurm
#   python $CODE/scripts/perf/summarize_throughput.py $WORK/tput_*
#
# LRC lr7 (2 x 28 cores). Mixes (array index): 0 = 56 x 1, 1 = 28 x 2, 2 = 14 x 4. Optional: N (lines,
# default 168), FULL_SET (default 0), SRC, WORK, ORCA_2MKL, MULTIWFN, FLAGS (extra runner flags).
#SBATCH --account=pc_aditik
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --cpus-per-task=56
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# never on a blau recharge account, whatever the header or command line says
case "${SLURM_JOB_ACCOUNT:-}" in *blau*) echo "refusing to run on account $SLURM_JOB_ACCOUNT"; exit 1 ;; esac
# ~/.bashrc can end on a command that returns non-zero, so set -e comes after the env setup
source ~/.bashrc
module load gcc
conda activate ${CONDA_ENV:-qtaim_engine}
set -eo pipefail
CODE=${CODE:?CODE is required: the branch checkout to import}
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_STACKSIZE=4G KMP_STACKSIZE=200M
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited
python -c "import sys, parsl, qtaim_gen; print(sys.executable, 'parsl', parsl.__version__, 'qtaim_gen from', qtaim_gen.__file__)"

MIXES=("56 1" "28 2" "14 4")
read N_PAR THREADS <<< "${MIXES[$SLURM_ARRAY_TASK_ID]}"
ORCA_2MKL=${ORCA_2MKL:-/global/scratch/users/santiagovargas/orca_6_0_1_linux_x86-64_shared_openmpi416/orca_2mkl}
MULTIWFN=${MULTIWFN:-/global/scratch/users/santiagovargas/Multiwfn/Multiwfn_3.8_dev_bin_Linux/Multiwfn}
SRC=${SRC:-/global/scratch/users/santiagovargas/OMol4M_raw}
WORK=${WORK:-/global/scratch/users/santiagovargas/engine_test}
N=${N:-168}
LIST=${LIST:?LIST is required}
TREE=$WORK/${TAG:-tput}_${N_PAR}x${THREADS}
mkdir -p $TREE logs
SLICE=$TREE/jobs.txt
head -n $N $LIST > $SLICE
echo "branch: $(git -C $CODE rev-parse --abbrev-ref HEAD) $(git -C $CODE rev-parse --short HEAD); mix ${N_PAR}x${THREADS}; $(wc -l < $SLICE) jobs"
python -c "import numba; print('numba', numba.__version__, 'USING_SVML', numba.config.USING_SVML)"

# RSS of every process of this user every 15 s (epoch pid ppid kB command), for peak memory per
# worker (with its children, e.g. orca_2mkl and Multiwfn) and per node
( while true; do ps -u $USER -o pid=,ppid=,rss=,comm= | awk -v t=$(date +%s) '{print t, $1, $2, $3, $4}'; sleep 15; done ) >> $TREE/rss.log &
SAMPLER=$!
START=$(date +%s.%N)
python -m qtaim_gen.source.scripts.full_runner_parsl_alcf \
  --num_folders $N --orca_2mkl_cmd $ORCA_2MKL --multiwfn_cmd $MULTIWFN \
  --clean --patch_timings --full_set ${FULL_SET:-0} \
  --n_threads $(( N_PAR * THREADS )) --n_threads_per_job $THREADS \
  --move_results --wfx --check_orca --charge_engine --qtaim_engine ${FLAGS:-} \
  --type_runner local --job_file $SLICE --preprocess_compressed \
  --root_omol_results $TREE/ --root_omol_inputs $SRC/ || echo "runner exited with $?"
END=$(date +%s.%N)
kill $SAMPLER
python -c "import json; json.dump({'n_par': $N_PAR, 'threads': $THREADS, 'listed': $(wc -l < $SLICE), 'start': $START, 'end': $END}, open('$TREE/throughput.json', 'w'))"
python $CODE/scripts/perf/summarize_throughput.py $TREE

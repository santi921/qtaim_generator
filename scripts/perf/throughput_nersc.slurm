#!/bin/bash
# Jobs per node-hour for --type_runner local at several jobs-per-node x threads-per-job mixes
# (docs/engine_roadmap/perf_plan.md, phase 1). Each array task is one mix on one whole CPU node,
# running the first N lines of $LIST from the raw inputs into its own tree $WORK/<TAG>_<mix>.
#
#   sbatch -J tput --export=ALL,CODE=$CODE,LIST=$WORK/sample.txt,TAG=tput --array=0-2 throughput_nersc.slurm
#   python $CODE/scripts/perf/summarize_throughput.py $WORK/tput_*
#
# Mixes (array index): 0 = 128 x 1, 1 = 64 x 2, 2 = 32 x 4, 3 = 256 x 1 (both hardware threads of
# every core; 2 GB per job). Optional: N (lines, default 384), FULL_SET (default 0), SRC, WORK,
# ORCA_2MKL, MULTIWFN, FLAGS (extra runner flags).
#SBATCH -A m3278
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --cpus-per-task=256
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -eo pipefail
module load python
conda activate /pscratch/sd/s/santiago/envs/generator
CODE=${CODE:?CODE is required: the branch checkout to import}
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 OMP_STACKSIZE=4G KMP_STACKSIZE=200M
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ulimit -s unlimited

MIXES=("128 1" "64 2" "32 4" "256 1")
read N_PAR THREADS <<< "${MIXES[$SLURM_ARRAY_TASK_ID]}"
ORCA_2MKL=${ORCA_2MKL:-/pscratch/sd/s/santiago/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca_2mkl}
MULTIWFN=${MULTIWFN:-/pscratch/sd/s/santiago/multiwfn/Multiwfn_3.8_dev_bin_Linux_noGUI/Multiwfn_noGUI}
SRC=${SRC:-/pscratch/sd/s/santiago}
WORK=${WORK:-/pscratch/sd/s/santiago/actinide_test}
N=${N:-384}
LIST=${LIST:?LIST is required}
TREE=$WORK/${TAG:-tput}_${N_PAR}x${THREADS}
mkdir -p $TREE logs
SLICE=$TREE/jobs.txt
head -n $N $LIST > $SLICE
echo "branch: $(git -C $CODE rev-parse --abbrev-ref HEAD) $(git -C $CODE rev-parse --short HEAD); mix ${N_PAR}x${THREADS}; $(wc -l < $SLICE) jobs"

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

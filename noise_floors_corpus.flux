#!/bin/sh
#flux: -N 1
#flux: -n 32
#flux: -q pbatch
#flux: -B dnn-sim
#flux: -t 480m

source ~/.bashrc
conda activate generator

export PYTHONUNBUFFERED=1
ulimit -s unlimited

ROOT=/p/lustre5/vargas58/converters/converters_final
OUT_DIR=/p/lustre5/vargas58/converters/noise_floors

JOBS=16 ROOT=$ROOT OUT_DIR=$OUT_DIR \
    bash "$(dirname "$0")/noise_floors_corpus.sh"

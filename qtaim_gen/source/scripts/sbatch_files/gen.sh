#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
conda activate tf_gpu
python qtaim.py


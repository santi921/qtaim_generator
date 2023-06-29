#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
module load AI/anaconda3-tf2.2020.11
conda activate base
module load orca/5.0.1
python controller.py 

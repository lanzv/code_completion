#!/bin/sh
#SBATCH -J pyphi
#SBATCH -o scripts/slurm_outputs/pyphi.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3



python3 run_experiment.py --model phi --dataset python

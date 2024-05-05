#!/bin/sh
#SBATCH -J pystar
#SBATCH -o scripts/slurm_outputs/pystar.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3



python3 run_experiment.py --model starcoder2 --dataset python

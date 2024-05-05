#!/bin/sh
#SBATCH -J pystab
#SBATCH -o scripts/slurm_outputs/pystab.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu3



python3 run_experiment.py --model stablecode --dataset python

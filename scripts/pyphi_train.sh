#!/bin/sh
#SBATCH -J tpyphi
#SBATCH -o scripts/slurm_outputs/pyphi_train.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu2



python3 run_experiment.py --model phi --dataset python --disable_tqdm True --train True

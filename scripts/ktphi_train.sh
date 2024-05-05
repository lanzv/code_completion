#!/bin/sh
#SBATCH -J tktphi
#SBATCH -o scripts/slurm_outputs/ktphi_train.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu1



python3 run_experiment.py --model phi --dataset kotlin --disable_tqdm True --train True

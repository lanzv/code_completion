#!/bin/sh
#SBATCH -J tktphi
#SBATCH -o scripts/slurm_outputs/ktphi_train.out
#SBATCH -p gpu-troja
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=tdll-3gpu4



python3 run_experiment.py --model phi --dataset kotlin --disable_tqdm True --train True

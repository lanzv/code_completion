#!/bin/sh
#SBATCH -J 2tpktphi
#SBATCH -o scripts/slurm_outputs/ktphi_train_python2.out
#SBATCH -p gpu-ms
#SBATCH -G 1
#SBATCH --mem-per-gpu=50G
#SBATCH --nodelist=dll-3gpu4



python3 run_experiment.py --model phi --dataset kotlin --disable_tqdm True --train True --evaluate_on_python_data True

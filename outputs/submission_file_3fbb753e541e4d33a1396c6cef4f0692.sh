#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=/home/guevel/OT4D/cell_similarity/outputs/%j_0_log.err
#SBATCH --gpus-per-node=8
#SBATCH --job-name=dinov2:train
#SBATCH --mem=0GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/home/guevel/OT4D/cell_similarity/outputs/%j_0_log.out
#SBATCH --partition=learnlab
#SBATCH --signal=USR2@120
#SBATCH --time=2800
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/guevel/OT4D/cell_similarity/outputs/%j_%t_log.out --error /home/guevel/OT4D/cell_similarity/outputs/%j_%t_log.err /home/guevel/.conda/envs/dinov2/bin/python -u -m submitit.core._submit /home/guevel/OT4D/cell_similarity/outputs

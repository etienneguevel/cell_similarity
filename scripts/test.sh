#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=memorytest
#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=15

#SBATCH --mail-type=ALL

#SBATCH --mail-user=etienneguev@gmail.com

#SBATCH --output=/home/guevel/logs/outs/%x-%j.out

#SBATCH --error=/home/guevel/logs/errs/%x-%j.err
source activate /home/guevel/.conda/envs/dinov2
echo $SLURM_NTASKS

python /home/guevel/OT4D/cell_similarity/scripts/test.py

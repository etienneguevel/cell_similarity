#!/bin/bash

#SBATCH --partition=hard

#SBATCH --nodelist=aerosmith

#SBATCH --job-name=memorytest1

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5

#SBATCH --mail-type=ALL

#SBATCH --mail-user=etienneguev@gmail.com

#SBATCH --output=/home/guevel/logs/outs/%x-%j.out

#SBATCH --error=/home/guevel/logs/errs/%x-%j.err
conda activate /home/guevel/.conda/envs/cellsim
python3 /home/guevel/src/cell_similarity/scripts/memory_test1.py

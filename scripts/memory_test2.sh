#!/bin/bash

#SBATCH --partition=electronic
#SBATCH --job-name=memorytest2
#SBATCH --nodes=1

#SBATCH --gpus-per-node=2

#SBATCH --time=15

#SBATCH --mail-type=ALL

#SBATCH --mail-user=etienneguev@gmail.com

#SBATCH --output=/home/guevel/logs/outs/%x-%j.out

#SBATCH --error=/home/guevel/logs/errs/%x-%j.err
source activate /home/guevel/.conda/envs/dinov2
nvidia-smi
python3 /home/guevel/OT4D/cell_similarity/scripts/test.py
python3 /home/guevel/OT4D/cell_similarity/src/cell_similarity/scripts/memory_test2.py --config-file /home/guevel/OT4D/cell_similarity/src/cell_similarity/scripts/config_test2.yaml

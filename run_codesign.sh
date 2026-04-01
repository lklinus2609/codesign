#!/bin/bash
#SBATCH -J codesign
#SBATCH -o codesign_%j.out
#SBATCH -e codesign_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -A IRI25030
#SBATCH --mail-type=all
#SBATCH --mail-user=lkim23@utexas.edu

source ~/.bashrc
cd /work/11297/lklinus0926/ls6/codesign_workspace
conda activate codesign

echo "=== $(date) ==="
nvidia-smi

python codesign/codesign_g1_unified.py \
    --num-train-envs 16392 \
    --outer-iters 5 \
    --max-inner-iters 10000 \
    "$@"

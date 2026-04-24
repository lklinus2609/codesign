#!/bin/bash
#SBATCH -J codesign-seeds
#SBATCH -o codesign-seeds_%j.out
#SBATCH -e codesign-seeds_%j.err
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=lkim23@utexas.edu

# Run several co-design jobs back-to-back with different master seeds.
# Default: seeds 7, 11, 13. Override via env var:
#   SEEDS="3 5 7 11 13" sbatch run_seeds.sh
# Extra CLI args pass through to codesign_g1_unified.py, e.g.:
#   sbatch run_seeds.sh --design-scope full
#   SEEDS="7 11" sbatch run_seeds.sh --outer-iters 50

source ~/.bashrc
cd /work/11297/lklinus0926/ls6/codesign_workspace
conda activate codesign

SEEDS=(${SEEDS:-7 11 13})

echo "=== $(date) ==="
echo "Partition: $SLURM_JOB_PARTITION"
echo "Seeds:     ${SEEDS[*]}"
echo "Extra args: $*"
nvidia-smi

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "========================================"
    echo "=== Seed $seed -- start $(date) ==="
    echo "========================================"
    python codesign/codesign_g1_unified.py \
        --num-train-envs 16392 \
        --outer-iters 100 \
        --max-inner-iters 10000 \
        --seed "$seed" \
        "$@"
    status=$?
    echo "=== Seed $seed -- exit $status at $(date) ==="
    if [ "$status" -ne 0 ]; then
        echo "[WARN] Seed $seed exited non-zero; continuing to next seed."
    fi
done

echo ""
echo "=== All seeds complete at $(date) ==="

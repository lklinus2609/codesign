#!/bin/bash

# --- Job body (runs inside SLURM when passed --run-job) ---
if [[ "$1" == "--run-job" ]]; then
    shift
    source ~/.bashrc
    cd /work/11297/lklinus0926/ls6/codesign_workspace
    conda activate codesign

    echo "=== $(date) ==="
    echo "Partition: $SLURM_JOB_PARTITION"
    nvidia-smi

    python codesign/codesign_g1_unified.py \
        --num-train-envs 16392 \
        --outer-iters 5 \
        --max-inner-iters 10000 \
        "$@"
    exit $?
fi

# --- Submission wrapper (runs on login node) ---
# Usage: bash run_codesign.sh [--partition PARTITION] [extra python args...]
# Partitions: gpu-a100, gpu-h100 (ls6) | amd-rtx, h100 (stampede3)
PARTITION="gpu-a100"
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        *) PYTHON_ARGS+=("$1"); shift ;;
    esac
done

# Map partition to account
case "$PARTITION" in
    gpu-a100|gpu-h100) ACCOUNT="IRI25030" ;;
    amd-rtx|h100)      ACCOUNT="IRI26004" ;;
    *) echo "Error: invalid partition '$PARTITION'"
       echo "Valid: gpu-a100, gpu-h100 (ls6) | amd-rtx, h100 (stampede3)"
       exit 1 ;;
esac

echo "Submitting to partition: $PARTITION (account: $ACCOUNT)"
exec sbatch \
    -J codesign \
    -o "codesign_%j.out" \
    -e "codesign_%j.err" \
    -p "$PARTITION" \
    -N 1 \
    -t "12:00:00" \
    -A "$ACCOUNT" \
    --mail-type=all \
    --mail-user=lkim23@utexas.edu \
    "$0" --run-job "${PYTHON_ARGS[@]}"

#!/bin/bash

#SBATCH -A shakeri-lab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -o ./slurm_logs/%x_%A_%a.out
#SBATCH -e ./slurm_logs/%x_%A_%a.err

set -euo pipefail

PYTHON_EX="/apps/software/standard/core/miniforge/24.11.3-py3.12/bin/python"

: "${SHARD_SIZE:=3}"
NAME="Comp_v6_KLD01"

now=$(date -Iseconds)
wallclock=$(squeue -h -j "${SLURM_JOB_ID}" -o %l || true)
gpuname=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)

echo "======================================"
echo "[JOB ID]: ${SLURM_ARRAY_JOB_ID}"
echo "[ARRAY ID]: ${SLURM_ARRAY_TASK_ID}"
echo "[JOB NAME]: ${SLURM_JOB_NAME}"
echo "[WALLCLOCK]: ${wallclock}"
echo "[START TIME]: ${now}"
echo "[GPU]: ${gpuname}"
echo "======================================"

$PYTHON_EX sample.py --name "$NAME" --shard_size "$SHARD_SIZE" --shard_index "$SLURM_ARRAY_TASK_ID" --descriptions

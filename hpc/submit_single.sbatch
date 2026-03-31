#!/bin/bash
# =============================================================
# submit_single.sbatch - Run one RLBench task evaluation
#                        NYU Torch HPC (overlay + official SIF)
#
# Usage:
#   sbatch hpc/submit_single.sbatch
#   TASK_NAME=open_box sbatch hpc/submit_single.sbatch
# =============================================================

#SBATCH --job-name=ip-sim
#SBATCH --account=<YOUR_ACCOUNT>          # ← Replace: run `my_slurm_accounts`
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p "${HOME}/instant_policy/logs"

export PROJECT_DIR="${HOME}/instant_policy"
export TASK_NAME="${TASK_NAME:-plate_out}"
export NUM_DEMOS="${NUM_DEMOS:-2}"
export NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "GPU:    $CUDA_VISIBLE_DEVICES"
echo "Task:   $TASK_NAME"

singularity exec --nv --fakeroot \
    --overlay /scratch/${USER}/instant-policy/overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    bash "${PROJECT_DIR}/hpc/run_sim.sh"

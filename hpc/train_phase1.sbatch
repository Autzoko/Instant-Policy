#!/bin/bash
# =============================================================
# Phase 1: Pre-train geometry encoder (occupancy network)
# Input:  ShapeNet meshes
# Output: checkpoints_ip/geo_encoder.pt
# Time:   ~2-4 hours on 1 GPU
# =============================================================

#SBATCH --job-name=ip-phase1
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err

mkdir -p ~/instant_policy/logs

SHAPENET_ROOT="${SHAPENET_ROOT:-/scratch/${USER}/ShapeNetCore.v2}"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "ShapeNet:  $SHAPENET_ROOT"
echo "Start:     $(date)"

singularity exec --nv \
    --overlay /scratch/${USER}/instant-policy/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
micromamba activate ip_env
cd ~/instant_policy

python -m ip.train \
    --shapenet_root ${SHAPENET_ROOT} \
    --save_dir ./checkpoints_ip \
    --phase 1 \
    --device cuda

echo 'Phase 1 complete: $(date)'
"

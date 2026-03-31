#!/bin/bash
# =============================================================
# Phase 2: Train full graph diffusion model
# Input:  ShapeNet meshes + Phase 1 encoder
# Output: checkpoints_ip/model_final.pt
# Time:   ~5 days on 1 GPU (2.5M steps)
#
# Resume: set RESUME env var
#   RESUME=./checkpoints_ip/model_step100000.pt sbatch ...
# =============================================================

#SBATCH --job-name=ip-phase2
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=168:00:00
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err

mkdir -p ~/instant_policy/logs

SHAPENET_ROOT="${SHAPENET_ROOT:-/scratch/${USER}/ShapeNetCore.v2}"
RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
    echo "Resuming from: $RESUME"
fi

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "ShapeNet:  $SHAPENET_ROOT"
echo "Start:     $(date)"

singularity exec --nv --fakeroot \
    --overlay /scratch/${USER}/instant-policy/overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
micromamba activate ip_env
cd ~/instant_policy

python -m ip.train \
    --shapenet_root ${SHAPENET_ROOT} \
    --save_dir ./checkpoints_ip \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 \
    --device cuda \
    ${RESUME_ARG}

echo 'Phase 2 complete: $(date)'
"

#!/bin/bash
# =============================================================
# Bimanual Phase 2: Train bimanual graph diffusion model
# Input:  ShapeNet meshes + Phase 1 encoder (shared with single-arm)
# Output: checkpoints_bimanual/bimanual_final.pt
# Time:   ~7 days on 1 GPU (2.5M steps, ~40% slower than single-arm)
#
# Resume: set RESUME env var
#   RESUME=./checkpoints_bimanual/bimanual_step100000.pt sbatch ...
# =============================================================

#SBATCH --job-name=ip-biman-p2
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=168:00:00
#SBATCH --output=logs/bimanual_phase2_%j.out
#SBATCH --error=logs/bimanual_phase2_%j.err

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

python -m ip.bimanual.train \
    --shapenet_root ${SHAPENET_ROOT} \
    --save_dir ./checkpoints_bimanual \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 \
    --device cuda \
    ${RESUME_ARG}

echo 'Bimanual Phase 2 complete: $(date)'
"

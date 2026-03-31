#!/bin/bash
# =============================================================
# Phase 3: Train language transfer module (phi_lang)
# Input:  Phase 2 model + language-annotated data
# Output: checkpoints_lang/phi_lang_final.pt
# Time:   ~6-12 hours on 1 GPU
# =============================================================

#SBATCH --job-name=ip-phase3
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase3_%j.out
#SBATCH --error=logs/phase3_%j.err

mkdir -p ~/instant_policy/logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"

singularity exec --nv \
    --overlay /scratch/${USER}/instant-policy/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
micromamba activate ip_env
cd ~/instant_policy

python -m ip.lang.train_lang \
    --ip_checkpoint ./checkpoints_ip/model_final.pt \
    --data_dir ./lang_data \
    --save_dir ./checkpoints_lang \
    --device cuda

echo 'Phase 3 complete: $(date)'
"

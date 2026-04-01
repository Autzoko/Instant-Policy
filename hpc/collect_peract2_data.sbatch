#!/bin/bash
# =============================================================
# Collect bimanual demonstrations from PerAct2/RLBench
#
# This script:
#   1. Installs PerAct2's fork of RLBench (bimanual support)
#   2. Collects demos for bimanual tasks
#   3. Converts to npz format for Instant Policy
#
# Prerequisites:
#   - CoppeliaSim installed in overlay (setup_overlay.sh)
#   - PyRep installed in overlay
#
# Output: /scratch/$USER/peract2_data/
# Time:   ~2-4 hours per task (13 tasks total)
# =============================================================

#SBATCH --job-name=collect-biman
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/collect_peract2_%j.out
#SBATCH --error=logs/collect_peract2_%j.err

mkdir -p ~/instant_policy/logs

SAVE_DIR="${SAVE_DIR:-/scratch/${USER}/peract2_data}"
NUM_EPISODES="${NUM_EPISODES:-20}"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Save dir:  $SAVE_DIR"
echo "Episodes:  $NUM_EPISODES per task"
echo "Start:     $(date)"

singularity exec --nv --fakeroot \
    --overlay /scratch/${USER}/instant-policy/overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
micromamba activate ip_env

# ── Start virtual display ──
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99
sleep 2

cd ~/instant_policy

python scripts/collect_bimanual_demos.py \
    --save_dir ${SAVE_DIR} \
    --num_episodes ${NUM_EPISODES}

echo 'Collection complete: $(date)'
"

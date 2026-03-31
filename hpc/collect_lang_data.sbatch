#!/bin/bash
# =============================================================
# Collect language-annotated demonstrations from RLBench
# Requires: CoppeliaSim + RLBench + virtual display
# Output:   ~/instant_policy/lang_data/
# Time:     ~1-2 hours
#
# NOTE: This requires CoppeliaSim installed in the overlay.
#       If not available, collect data on a machine with
#       CoppeliaSim and upload to HPC.
# =============================================================

#SBATCH --job-name=ip-collect
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err

mkdir -p ~/instant_policy/logs

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"

singularity exec --nv --fakeroot \
    --overlay /scratch/${USER}/instant-policy/overlay-25GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
micromamba activate ip_env

# Start virtual display for CoppeliaSim
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99
sleep 2

cd ~/instant_policy

python -c \"
from ip.lang.lang_dataset import collect_rlbench_lang_data
collect_rlbench_lang_data(
    save_dir='./lang_data',
    demos_per_task=20,
    headless=True,
)
\"

echo 'Data collection complete: \$(date)'
"

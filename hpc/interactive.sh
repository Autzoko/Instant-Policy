#!/bin/bash
# =============================================================
# interactive.sh - Interactive debugging inside container
#                  NYU Torch HPC (overlay + official SIF)
#
# Step 1: Get a GPU node:
#   srun --account=<YOUR_ACCOUNT> --gres=gpu:1 --cpus-per-task=4 \
#        --mem=32GB --time=01:00:00 --pty /bin/bash
#
# Step 2: Run this script:
#   bash ~/instant_policy/hpc/interactive.sh
# =============================================================

singularity exec --nv \
    --overlay /scratch/${USER}/instant-policy/overlay-15GB-500K.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c '
        source /ext3/env.sh
        conda activate ip_env

        Xvfb :99 -screen 0 1280x1024x24 &
        export DISPLAY=:99
        sleep 2

        echo "================================================"
        echo " Container ready."
        echo " CoppeliaSim + ip_env activated."
        echo " cd ~/instant_policy && python deploy_sim.py ..."
        echo "================================================"

        exec /bin/bash
    '

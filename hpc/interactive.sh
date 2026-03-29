#!/bin/bash
# =============================================================
# interactive.sh - Quick-start for interactive debugging
#
# Step 1: Get an interactive GPU node:
#   srun --account=<YOUR_ACCOUNT> --gres=gpu:1 --cpus-per-task=4 \
#        --mem=32GB --time=01:00:00 --pty /bin/bash
#
# Step 2: Run this script on the compute node:
#   bash ~/instant_policy/hpc/interactive.sh
#
# Step 3: You are now inside the container with everything ready.
# =============================================================

singularity exec --nv \
    --bind "${HOME}:${HOME}" \
    --bind "/scratch/${USER}:/scratch/${USER}" \
    "/scratch/${USER}/instant-policy.sif" \
    /bin/bash -c '
        export COPPELIASIM_ROOT=/opt/CoppeliaSim
        export LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}
        export QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}

        Xvfb :99 -screen 0 1280x1024x24 &
        export DISPLAY=:99
        sleep 2

        source /opt/conda/etc/profile.d/conda.sh
        conda activate ip_env

        echo "================================================"
        echo " Container ready. CoppeliaSim + ip_env activated."
        echo " cd ~/instant_policy && python deploy_sim.py ..."
        echo "================================================"

        exec /bin/bash
    '

#!/bin/bash
# =============================================================
# run_sim.sh - Run Instant Policy RLBench simulation inside
#              Singularity container on NYU Torch HPC
# =============================================================

set -e

# ---------- CoppeliaSim ----------
export COPPELIASIM_ROOT=/opt/CoppeliaSim
export LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}
export QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}

# ---------- Start virtual display ----------
echo "[INFO] Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

# Verify display
if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "[ERROR] Xvfb failed to start"
    exit 1
fi
echo "[INFO] Xvfb started (PID=$XVFB_PID)"

# ---------- Activate conda ----------
source /opt/conda/etc/profile.d/conda.sh
conda activate ip_env

# ---------- Print environment info ----------
echo "[INFO] Python: $(python --version)"
echo "[INFO] PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "[INFO] CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "[INFO] GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"

# ---------- Navigate to project ----------
# PROJECT_DIR is passed via SLURM script, defaults to current user's home
PROJECT_DIR="${PROJECT_DIR:-$HOME/instant_policy}"
cd "$PROJECT_DIR"
echo "[INFO] Working directory: $(pwd)"

# ---------- Run simulation ----------
TASK_NAME="${TASK_NAME:-plate_out}"
NUM_DEMOS="${NUM_DEMOS:-2}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"
RESTRICT_ROT="${RESTRICT_ROT:-1}"

echo "[INFO] Task: $TASK_NAME | Demos: $NUM_DEMOS | Rollouts: $NUM_ROLLOUTS"
echo "========================================================"

python deploy_sim.py \
    --task_name="$TASK_NAME" \
    --num_demos="$NUM_DEMOS" \
    --num_rollouts="$NUM_ROLLOUTS" \
    --restrict_rot="$RESTRICT_ROT"

echo "========================================================"
echo "[INFO] Finished."

# ---------- Cleanup ----------
kill $XVFB_PID 2>/dev/null || true

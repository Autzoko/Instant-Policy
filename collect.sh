#!/bin/bash
# =============================================================
# collect.sh - Collect RLBench language-annotated demonstrations
#
# Usage (Docker on Windows/Linux x86):
#   docker build -t instant-policy .
#   docker run -v "$(pwd)":/workspace instant-policy bash /workspace/collect.sh
#
# Windows CMD:
#   docker run -v "%cd%":/workspace instant-policy bash /workspace/collect.sh
#
# Windows PowerShell:
#   docker run -v "${PWD}":/workspace instant-policy bash /workspace/collect.sh
#
# Output: ./lang_data/ (~2GB, 24 tasks x 20 demos each)
# Time:   ~1-2 hours on CPU
# =============================================================

set -e

echo "============================================"
echo " RLBench Language Data Collection"
echo "============================================"

# Activate conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate ip_env

# Install compatible sentence-transformers
echo "[1/4] Installing sentence-transformers..."
pip install -q sentence-transformers==3.0.1 transformers==4.42.4

# Force NVIDIA GPU rendering instead of Mesa software renderer
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export MESA_GL_VERSION_OVERRIDE=3.3
# Remove Mesa's swrast so CoppeliaSim doesn't try to use it
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export NVIDIA_DRIVER_CAPABILITIES=all

# Start virtual display
echo "[2/4] Starting virtual display..."
Xvfb :1 -screen 0 1280x1024x24 &
XVFB_PID=$!
export DISPLAY=:1
sleep 2

if ! kill -0 $XVFB_PID 2>/dev/null; then
    echo "[ERROR] Xvfb failed to start"
    exit 1
fi
echo "  Xvfb started (PID=$XVFB_PID)"

# Collect data
echo "[3/4] Collecting demonstrations (20 per task, 24 tasks)..."
echo "  This takes ~1-2 hours. Be patient."
cd /workspace

python -c "
from ip.lang.lang_dataset import collect_rlbench_lang_data
collect_rlbench_lang_data(
    save_dir='./lang_data',
    demos_per_task=20,
    headless=True,
)
"

# Cleanup
echo "[4/4] Cleaning up..."
kill $XVFB_PID 2>/dev/null || true

echo ""
echo "============================================"
echo " Collection complete!"
echo " Data saved to: ./lang_data/"
echo ""
echo " Next: upload to HPC with:"
echo "   scp -r lang_data ll5582@greene.hpc.nyu.edu:~/instant_policy/"
echo "============================================"

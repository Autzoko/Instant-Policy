#!/bin/bash
# =============================================================
# setup_peract2.sh - Install PerAct2 bimanual RLBench fork
#
# Run INSIDE a compute node with the overlay activated:
#   srun --account=<YOUR_ACCOUNT> --cpus-per-task=4 --mem=32GB \
#        --time=02:00:00 --gres=gpu:1 --pty /bin/bash
#
#   singularity exec --nv --fakeroot \
#       --overlay /scratch/${USER}/instant-policy/overlay-25GB-500K.ext3:rw \
#       /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
#       bash ~/instant_policy/hpc/setup_peract2.sh
#
# What this does:
#   1. Clones PerAct2's bimanual RLBench fork
#   2. Installs it into the existing ip_env
#   3. Verifies bimanual task availability
# =============================================================

set -e

source /ext3/env.sh
micromamba activate ip_env

echo "============================================"
echo " PerAct2 Bimanual RLBench Installation"
echo "============================================"

# ── Step 1: Clone bimanual RLBench fork ────────────────────
echo "[1/3] Cloning PerAct2 bimanual RLBench fork..."

cd /tmp

# Remove old clone if exists
rm -rf /tmp/RLBench_bimanual

# PerAct2 uses a modified RLBench with bimanual support
# Main repo: https://github.com/markusgrotz/peract_bimanual
# Their RLBench fork: https://github.com/markusgrotz/RLBench
git clone --depth 1 https://github.com/markusgrotz/RLBench.git RLBench_bimanual

# ── Step 2: Install ────────────────────────────────────────
echo "[2/3] Installing bimanual RLBench..."
cd /tmp/RLBench_bimanual

# Install, replacing the existing RLBench
pip install --no-cache-dir -e . --force-reinstall --no-deps
# Install dependencies that might be missing
pip install --no-cache-dir pyquaternion natsort

echo "  Installed successfully."

# ── Step 3: Verify ─────────────────────────────────────────
echo "[3/3] Verifying installation..."

python -c "
import rlbench
print(f'RLBench version: {rlbench.__version__}')
print(f'RLBench path: {rlbench.__file__}')

# Check if bimanual action modes exist
try:
    from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
    print('BimanualMoveArmThenGripper: OK')
except ImportError:
    print('BimanualMoveArmThenGripper: NOT FOUND (may use standard action modes)')

# List available tasks
import os
task_dir = os.path.join(os.path.dirname(rlbench.__file__), 'tasks')
if os.path.isdir(task_dir):
    tasks = [f[:-3] for f in os.listdir(task_dir)
             if f.endswith('.py') and not f.startswith('_')]
    bimanual = [t for t in tasks if 'bimanual' in t or 'coordinated' in t
                or 'handover' in t or 'dual' in t]
    print(f'Total tasks: {len(tasks)}')
    print(f'Bimanual tasks found: {len(bimanual)}')
    for t in sorted(bimanual):
        print(f'  - {t}')
"

# Cleanup
rm -rf /tmp/RLBench_bimanual

echo ""
echo "============================================"
echo " PerAct2 setup complete!"
echo " Now run data collection:"
echo "   sbatch ~/instant_policy/hpc/collect_peract2_data.sbatch"
echo "============================================"

#!/bin/bash
# =============================================================
# setup_overlay.sh - One-time environment setup on NYU HPC
#
# Run this on a COMPUTE NODE (not login node) for faster I/O:
#   srun --account=<YOUR_ACCOUNT> --cpus-per-task=4 --mem=32GB \
#        --time=02:00:00 --pty /bin/bash
#   bash ~/instant_policy/hpc/setup_overlay.sh
#
# What this does:
#   1. Creates a 15GB overlay filesystem
#   2. Installs Micromamba (fast conda replacement)
#   3. Creates ip_env with all dependencies
#   4. Takes ~20-30 minutes total
# =============================================================

set -e

OVERLAY_DIR="/scratch/${USER}/instant-policy"
OVERLAY_FILE="${OVERLAY_DIR}/overlay-15GB-500K.ext3"
SIF="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"

echo "============================================"
echo " Instant Policy - HPC Environment Setup"
echo " User: ${USER}"
echo " Overlay: ${OVERLAY_FILE}"
echo "============================================"

# ── Step 1: Create overlay ──────────────────────────────────
mkdir -p "${OVERLAY_DIR}"

if [ ! -f "${OVERLAY_FILE}" ]; then
    echo "[1/4] Creating overlay filesystem..."
    cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz "${OVERLAY_DIR}/"
    cd "${OVERLAY_DIR}"
    gunzip overlay-15GB-500K.ext3.gz
    echo "  Done."
else
    echo "[1/4] Overlay already exists, skipping."
fi

# ── Step 2-4: Install everything inside the container ───────
echo "[2/4] Launching container to install packages..."

singularity exec --nv \
    --overlay "${OVERLAY_FILE}" \
    "${SIF}" \
    /bin/bash << 'CONTAINER_SCRIPT'

set -e

# ── Install Micromamba (instant, no unpacking) ──────────────
echo "[2/4] Installing Micromamba..."
mkdir -p /ext3/bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /ext3 bin/micromamba 2>/dev/null
echo "  Micromamba installed."

# ── Create env.sh activation script ─────────────────────────
cat > /ext3/env.sh << 'ENVEOF'
#!/bin/bash
export MAMBA_ROOT_PREFIX=/ext3/micromamba_envs
export PATH=/ext3/bin:$PATH
eval "$(micromamba shell hook -s bash)"
ENVEOF

source /ext3/env.sh

# ── Create ip_env ───────────────────────────────────────────
echo "[3/4] Creating ip_env and installing packages..."
echo "  This takes 15-20 minutes. Be patient."

micromamba create -n ip_env python=3.10 -c conda-forge -y -q

# Activate
micromamba activate ip_env

# PyTorch + CUDA 11.8
echo "  Installing PyTorch..."
micromamba install -n ip_env -y -q \
    pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -c conda-forge

# PyTorch Geometric
echo "  Installing PyTorch Geometric..."
micromamba install -n ip_env -y -q \
    pyg==2.5.0 pytorch-scatter pytorch-cluster \
    -c pyg -c pytorch -c nvidia -c conda-forge

# Scientific stack
echo "  Installing scientific packages..."
micromamba install -n ip_env -y -q \
    numpy==1.26.4 scipy scikit-learn \
    pyyaml tqdm pillow \
    -c conda-forge

# Pip packages
echo "  Installing pip packages..."
pip install --no-cache-dir -q \
    open3d==0.18.0 \
    sentence-transformers \
    diffusers==0.31.0 \
    transformers==4.46.2 \
    accelerate \
    pyquaternion \
    trimesh \
    wandb \
    gdown

# pyg-lib
echo "  Installing pyg-lib..."
pip install --no-cache-dir -q pyg-lib \
    -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# ── Verify ──────────────────────────────────────────────────
echo "[4/4] Verifying installation..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA compiled: {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
import torch_geometric
print(f'  PyG: {torch_geometric.__version__}')
import open3d
print(f'  Open3D: {open3d.__version__}')
from sentence_transformers import SentenceTransformer
print('  sentence-transformers: OK')
import trimesh
print('  trimesh: OK')
"

echo ""
echo "============================================"
echo " Environment setup complete!"
echo " Activate with:"
echo "   source /ext3/env.sh"
echo "   micromamba activate ip_env"
echo "============================================"

CONTAINER_SCRIPT

echo ""
echo "All done. Overlay saved at: ${OVERLAY_FILE}"

FROM nvidia/cudagl:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ============================================================
# 1. System dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git ca-certificates \
        build-essential cmake \
        software-properties-common \
        xvfb x11-utils mesa-utils \
        libgl1-mesa-glx libgl1-mesa-dri libglu1-mesa \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
        libxkbcommon-x11-0 libfontconfig1 libxi6 \
        libavcodec-dev libavformat-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. Install Miniconda
#    Pin a specific version to avoid TOS/breaking changes.
# ============================================================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Bypass Anaconda TOS prompt (non-interactive Docker build)
# and configure conda for non-interactive use
RUN conda config --set auto_activate_base false \
    && conda init bash

# ============================================================
# 3. Install CoppeliaSim V4.1.0
# ============================================================
ENV COPPELIASIM_ROOT=/opt/CoppeliaSim
ENV LD_LIBRARY_PATH=${COPPELIASIM_ROOT}:${LD_LIBRARY_PATH}
ENV QT_QPA_PLATFORM_PLUGIN_PATH=${COPPELIASIM_ROOT}

RUN wget -q https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz \
        -O /tmp/coppeliasim.tar.xz \
    && mkdir -p ${COPPELIASIM_ROOT} \
    && tar -xf /tmp/coppeliasim.tar.xz -C ${COPPELIASIM_ROOT} --strip-components 1 \
    && rm /tmp/coppeliasim.tar.xz

# ============================================================
# 4. Create conda environment (ip_env)
#    Use conda-forge as default channel to avoid Anaconda TOS.
# ============================================================
RUN conda create -n ip_env python=3.10 -y -c conda-forge --override-channels

# Helper: all subsequent conda/pip commands run inside ip_env.
# We use CONDA_PREFIX + shell source pattern for reliability.
SHELL ["/bin/bash", "-c"]

# ---- PyTorch + CUDA 11.8 ----
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    conda install -y pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 \
        -c pytorch -c nvidia -c conda-forge --override-channels

# ---- PyTorch Geometric + extensions ----
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    conda install -y pyg==2.5.0 pytorch-scatter pytorch-cluster \
        -c pyg -c pytorch -c nvidia -c conda-forge --override-channels

# ---- Scientific stack ----
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    conda install -y \
        numpy==1.26.4 scipy scikit-learn \
        pyyaml tqdm pillow \
        pytorch-lightning lightning \
        -c conda-forge --override-channels

# ---- Pip packages ----
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    pip install --no-cache-dir \
        open3d==0.18.0 \
        diffusers==0.31.0 \
        transformers==4.46.2 \
        accelerate==1.1.1 \
        huggingface-hub \
        safetensors \
        pyquaternion \
        plotly matplotlib \
        gymnasium==1.0.0 \
        wandb \
        gdown \
        natsort pandas

# ---- pyg-lib (needs matching torch+cuda wheel) ----
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    pip install --no-cache-dir pyg-lib \
        -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# ============================================================
# 5. Install PyRep (Python bindings for CoppeliaSim)
# ============================================================
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    cd /tmp && git clone https://github.com/stepjam/PyRep.git && \
    cd PyRep && pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/PyRep

# ============================================================
# 6. Install RLBench
# ============================================================
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    pip install --no-cache-dir git+https://github.com/stepjam/RLBench.git

# ============================================================
# 7. Verify installation
# ============================================================
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
    python -c "\
import torch; print('PyTorch:', torch.__version__); \
print('CUDA compiled:', torch.version.cuda); \
import open3d; print('Open3D:', open3d.__version__); \
print('torch-geometric package present:', __import__('importlib').util.find_spec('torch_geometric') is not None); \
print('NOTE: CUDA extensions (pyg-lib, torch-scatter, torch-cluster) require GPU at runtime.'); \
"

# ============================================================
# 8. Environment defaults
# ============================================================
ENV DISPLAY=:99

# Reset to default shell
SHELL ["/bin/sh", "-c"]

WORKDIR /workspace
CMD ["/bin/bash"]

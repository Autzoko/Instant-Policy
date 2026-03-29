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
# 2. Install Python 3.10 from deadsnakes PPA
# ============================================================
RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-distutils python3.10-venv \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# ============================================================
# 3. Install Miniconda
# ============================================================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash

# ============================================================
# 4. Install CoppeliaSim V4.1.0
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
# 5. Create conda environment (ip_env) with core dependencies
#    Using explicit installs instead of environment.yml to avoid
#    platform-specific build hash mismatches.
# ============================================================
RUN conda create -n ip_env python=3.10 -y

# Install PyTorch + CUDA 11.8
RUN /bin/bash -c "source activate ip_env && \
    conda install -y pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 \
        -c pytorch -c nvidia"

# Install PyTorch Geometric + extensions
RUN /bin/bash -c "source activate ip_env && \
    conda install -y pyg==2.5.0 pytorch-scatter pytorch-cluster \
        -c pyg -c pytorch -c nvidia"

# Install other conda packages
RUN /bin/bash -c "source activate ip_env && \
    conda install -y \
        numpy==1.26.4 scipy scikit-learn \
        pyyaml tqdm pillow \
        pytorch-lightning lightning \
    -c pytorch -c conda-forge"

# Install pip packages
RUN /bin/bash -c "source activate ip_env && \
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
        natsort pandas"

# Install pyg-lib
RUN /bin/bash -c "source activate ip_env && \
    pip install --no-cache-dir pyg-lib \
        -f https://data.pyg.org/whl/torch-2.2.0+cu118.html"

# ============================================================
# 6. Install PyRep (Python bindings for CoppeliaSim)
# ============================================================
RUN /bin/bash -c "source activate ip_env && \
    cd /tmp && git clone https://github.com/stepjam/PyRep.git && \
    cd PyRep && pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir . && \
    rm -rf /tmp/PyRep"

# ============================================================
# 7. Install RLBench
# ============================================================
RUN /bin/bash -c "source activate ip_env && \
    pip install --no-cache-dir git+https://github.com/stepjam/RLBench.git"

# ============================================================
# 8. Default environment variables
# ============================================================
ENV DISPLAY=:99

# Default working directory
WORKDIR /workspace

CMD ["/bin/bash"]

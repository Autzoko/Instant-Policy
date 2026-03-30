# Instant Policy

Code for the paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion",
[Project Webpage](https://www.robot-learning.uk/instant-policy)

**Extended with language-guided modality transfer** (Appendix J): use natural language descriptions instead of demonstrations to control the robot.

<p align="center">
<img src="./media/rollout_roll.gif" alt="drawing" width="700"/>
</p>

## Repository Structure

```
instant_policy/
├── deploy.py / deploy_sim.py       # Original deployment scripts (pre-trained .so model)
├── sim_utils.py / utils.py         # Simulation and point cloud utilities
├── instant_policy.so / .pyi        # Pre-trained compiled model (inference only)
│
├── ip/                             # Full reimplementation (trainable PyTorch)
│   ├── config.py                   # All hyperparameters (Appendix C/E)
│   ├── se3_utils.py                # SE(3) math: logmap, expmap, SVD alignment
│   ├── pos_encoding.py             # NeRF-like positional encoding
│   ├── geometry_encoder.py         # φ_e: PointNet++ Set Abstraction (Appendix A)
│   ├── graph_transformer.py        # Heterogeneous Graph Transformer (Eq. 3)
│   ├── graph_builder.py            # Local / context / action graph construction
│   ├── networks.py                 # σ, φ, ψ sub-networks (Eq. 6, Appendix C)
│   ├── diffusion.py                # SE(3) diffusion: DDPM forward + DDIM reverse
│   ├── model.py                    # GraphDiffusionPolicy (full model)
│   ├── pseudo_demo.py              # Pseudo-demonstration generator (Appendix D)
│   ├── dataset.py                  # Datasets: pseudo-demo / RLBench
│   ├── train.py                    # Two-phase training pipeline
│   ├── deploy_lang.py              # Language-guided deployment
│   └── lang/                       # Language transfer module (Appendix J)
│       ├── encoder.py              # Sentence-BERT + projection MLP
│       ├── phi_lang.py             # φ_lang: language-conditioned graph transformer
│       ├── lang_dataset.py         # Language-annotated dataset + RLBench collector
│       └── train_lang.py           # Contrastive + MSE bottleneck alignment training
│
├── hpc/                            # NYU HPC / SLURM deployment scripts
│   ├── submit_single.sbatch        # Submit single task evaluation
│   ├── submit_all_tasks.sh         # Batch submit all RLBench tasks
│   ├── run_sim.sh                  # Simulation runner (inside container)
│   └── interactive.sh              # Interactive debugging session
│
├── Dockerfile                      # Container: CUDA + CoppeliaSim + RLBench
├── environment.yml                 # Conda environment specification
└── download_weights.sh             # Download pre-trained weights
```

## Setup

### Option A: Local Machine (with GPU)

```bash
git clone https://github.com/Autzoko/Instant-Policy.git
cd Instant-Policy

# Create conda environment
conda env create -f environment.yml
conda activate ip_env
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Install sentence-transformers (for language transfer)
pip install sentence-transformers
```

### Option B: Docker (recommended for reproducibility)

```bash
docker build -t instant-policy .
docker run --gpus all -it -v $(pwd):/workspace instant-policy bash

# Inside container:
source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env
pip install sentence-transformers
cd /workspace
```

### Option C: NYU HPC (Greene / Torch clusters)

See [HPC Deployment](#hpc-deployment) section below.

## Quick Start: Pre-trained Model (Inference Only)

```bash
# Download pre-trained weights
./download_weights.sh

# Run RLBench simulation evaluation
python deploy_sim.py --task_name='plate_out' --num_demos=2 --num_rollouts=10
```

## Training from Scratch

Training has three phases. All commands assume you are in the project root.

### Phase 1: Pre-train Geometry Encoder

Pre-train the PointNet++ geometry encoder as an occupancy network on ShapeNet objects.

```bash
# Download ShapeNet: https://shapenet.org/ (ShapeNetCore.v2)
# Extract to e.g. /data/ShapeNetCore.v2/

python -m ip.train \
    --shapenet_root /data/ShapeNetCore.v2 \
    --save_dir ./checkpoints_ip \
    --phase 1 \
    --device cuda
```

This saves `./checkpoints_ip/geo_encoder.pt`. Takes ~2-4 hours on a single GPU.

### Phase 2: Train Graph Diffusion Model

Train the full σ → φ → ψ pipeline on pseudo-demonstrations.

```bash
python -m ip.train \
    --shapenet_root /data/ShapeNetCore.v2 \
    --save_dir ./checkpoints_ip \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 \
    --device cuda
```

Training parameters (Appendix E):
- AdamW, lr=1e-5, weight_decay=1e-4
- 2.5M steps + 50K cosine cooldown
- float16 mixed precision
- ~5 days on a single RTX 3080-ti

Checkpoints saved every 10K steps. Resume with `--resume ./checkpoints_ip/model_step100000.pt`.

### Phase 3: Train Language Transfer Module

Train φ_lang to replace demonstrations with language commands.

**Step 3a: Collect language-annotated demonstrations from RLBench**

```bash
python -c "
from ip.lang.lang_dataset import collect_rlbench_lang_data
collect_rlbench_lang_data(
    save_dir='./lang_data',
    demos_per_task=20,
    headless=True
)
"
```

This requires RLBench + CoppeliaSim. Collects ~20 demos per task with language descriptions.

**Step 3b: Train φ_lang**

```bash
python -m ip.lang.train_lang \
    --ip_checkpoint ./checkpoints_ip/model_final.pt \
    --data_dir ./lang_data \
    --save_dir ./checkpoints_lang \
    --device cuda
```

Training parameters:
- Only φ_lang parameters trained (~45M); IP model fully frozen
- AdamW, lr=1e-4
- ~100K steps
- Loss: InfoNCE contrastive + MSE bottleneck alignment

### Language-Guided Deployment

After training, use natural language instead of demonstrations:

```bash
python -m ip.deploy_lang \
    --ip_checkpoint ./checkpoints_ip/model_final.pt \
    --lang_checkpoint ./checkpoints_lang/phi_lang_final.pt \
    --task "close the microwave"
```

In code:

```python
from ip.deploy_lang import LanguageGuidedPolicy

policy = LanguageGuidedPolicy(
    ip_checkpoint='./checkpoints_ip/model_final.pt',
    lang_checkpoint='./checkpoints_lang/phi_lang_final.pt',
    device='cuda'
)

# No demonstrations needed — just language + current observation
actions, grips = policy.predict_actions(
    task_description="close the microwave",
    pcd=current_point_cloud,    # (N, 3) segmented point cloud, world frame
    T_w_e=gripper_pose,         # (4, 4) end-effector pose
    grip=1,                     # 0=closed, 1=open
)
# actions: (8, 4, 4) relative SE(3) transforms
# grips:   (8,) gripper commands (0=close, 1=open)
```

## HPC Deployment

### NYU HPC Setup (Greene / Torch)

**1. Build the Singularity overlay (one-time)**

```bash
# On the HPC login node:
cd /scratch/$USER
mkdir -p instant-policy && cd instant-policy

# Create overlay filesystem (15GB, 500K inodes)
cp -rp /scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz

# Launch interactive container to install packages
singularity exec --nv \
    --overlay overlay-15GB-500K.ext3 \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash

# Inside the container:
# Install Miniconda into /ext3
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -p /ext3/miniconda3
rm Miniconda3-py310_24.1.2-0-Linux-x86_64.sh

# Create activation script
cat > /ext3/env.sh << 'EOF'
#!/bin/bash
export PATH=/ext3/miniconda3/bin:$PATH
source /ext3/miniconda3/etc/profile.d/conda.sh
EOF

source /ext3/env.sh

# Create environment and install packages
conda create -n ip_env python=3.10 -y
conda activate ip_env

# PyTorch + CUDA
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y

# PyTorch Geometric
conda install pyg==2.5.0 pytorch-scatter pytorch-cluster \
    -c pyg -c pytorch -c nvidia -y
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# Scientific stack
conda install numpy==1.26.4 scipy scikit-learn pytorch-lightning -c conda-forge -y

# Remaining packages
pip install open3d==0.18.0 diffusers==0.31.0 transformers==4.46.2 \
    sentence-transformers pyquaternion wandb gdown

# Exit container
exit
```

**2. Upload project code**

```bash
# From your local machine:
rsync -avz --exclude='*.tar.gz' --exclude='*.pdf' --exclude='*.so' \
    --exclude='__pycache__' --exclude='checkpoints/' \
    . $USER@greene.hpc.nyu.edu:~/instant_policy/
```

**3. Download ShapeNet on the HPC**

```bash
# ShapeNet requires registration: https://shapenet.org/
# After getting access, download ShapeNetCore.v2 to /scratch/$USER/
```

**4. Submit training jobs**

```bash
# Edit hpc/submit_single.sbatch: set --account=<YOUR_ACCOUNT>

# Phase 1: Geometry encoder pre-training (~2h, 1 GPU)
cat > hpc/train_phase1.sbatch << 'SBATCH'
#!/bin/bash
#SBATCH --job-name=ip-geo
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/geo_%j.out

mkdir -p logs
singularity exec --nv \
    --overlay /scratch/$USER/instant-policy/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    bash -c "
source /ext3/env.sh && conda activate ip_env && \
cd ~/instant_policy && \
python -m ip.train --shapenet_root /scratch/$USER/ShapeNetCore.v2 \
    --save_dir ./checkpoints_ip --phase 1 --device cuda
"
SBATCH

sbatch hpc/train_phase1.sbatch

# Phase 2: Full model training (~5 days, 1 GPU)
cat > hpc/train_phase2.sbatch << 'SBATCH'
#!/bin/bash
#SBATCH --job-name=ip-train
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=168:00:00
#SBATCH --output=logs/train_%j.out

mkdir -p logs
singularity exec --nv \
    --overlay /scratch/$USER/instant-policy/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    bash -c "
source /ext3/env.sh && conda activate ip_env && \
cd ~/instant_policy && \
python -m ip.train --shapenet_root /scratch/$USER/ShapeNetCore.v2 \
    --save_dir ./checkpoints_ip \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 --device cuda
"
SBATCH

sbatch hpc/train_phase2.sbatch

# Phase 3: Language transfer (~6h, 1 GPU)
# Requires Phase 2 checkpoint + language-annotated data
cat > hpc/train_lang.sbatch << 'SBATCH'
#!/bin/bash
#SBATCH --job-name=ip-lang
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/lang_%j.out

mkdir -p logs
singularity exec --nv \
    --overlay /scratch/$USER/instant-policy/overlay-15GB-500K.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    bash -c "
source /ext3/env.sh && conda activate ip_env && \
cd ~/instant_policy && \
python -m ip.lang.train_lang \
    --ip_checkpoint ./checkpoints_ip/model_final.pt \
    --data_dir ./lang_data \
    --save_dir ./checkpoints_lang \
    --device cuda
"
SBATCH

sbatch hpc/train_lang.sbatch
```

**5. Monitor jobs**

```bash
squeue -u $USER                    # Check job status
tail -f logs/train_*.out           # Watch training progress
sacct -j <JOB_ID> --format=...    # Job resource usage
```

### Generic Server (non-HPC)

```bash
# With Docker:
docker build -t instant-policy .
docker run --gpus all -d --name ip-train \
    -v /data/ShapeNetCore.v2:/data/ShapeNet \
    -v $(pwd):/workspace \
    instant-policy bash -c "
source /opt/conda/etc/profile.d/conda.sh && conda activate ip_env && \
pip install sentence-transformers && \
cd /workspace && \
python -m ip.train --shapenet_root /data/ShapeNet --save_dir ./checkpoints_ip --device cuda
"

# Monitor:
docker logs -f ip-train

# Without Docker (bare metal):
conda activate ip_env
python -m ip.train --shapenet_root /path/to/ShapeNet --save_dir ./checkpoints_ip --device cuda
```

## Deploy on Your Robot

Collect demonstrations as `demo = {'pcds': [], 'T_w_es': [], 'grips': []}` where:
- `pcds`: list of segmented point clouds (numpy arrays, world frame)
- `T_w_es`: list of end-effector poses (4x4 numpy arrays, world frame)
- `grips`: list of gripper states (0=closed, 1=open)

For the original pre-trained model, see `deploy.py`. For language-guided control, see above.

## Notes on Performance

- Objects of interest should be well segmented.
- Tasks should follow the Markovian assumption (no observation history).
- Demonstrations should be short and consistent.
- Model uses point clouds in the end-effector frame.
- To avoid gripper oscillation, execute until gripper state changes then re-query.

## Citing

```
@inproceedings{vosylius2024instant,
  title={Instant Policy: In-Context Imitation Learning via Graph Diffusion},
  author={Vosylius, Vitalis and Johns, Edward},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

# Instant Policy

Code for the paper: "Instant Policy: In-Context Imitation Learning via Graph Diffusion",
[Project Webpage](https://www.robot-learning.uk/instant-policy)

**Extended with:**
- Language-guided modality transfer (Appendix J)
- **Bimanual (dual-arm) manipulation** via cross-arm heterogeneous graph edges

<p align="center">
<img src="./media/rollout_roll.gif" alt="drawing" width="700"/>
</p>

## Repository Structure

```
instant_policy/
├── deploy.py / deploy_sim.py       # Original deployment scripts (pre-trained .so model)
├── eval_bimanual.py                # Bimanual evaluation CLI (PerAct2 tasks)
├── bimanual_sim_utils.py           # Bimanual RLBench environment + rollout utilities
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
│   ├── lang/                       # Language transfer module (Appendix J)
│   │   ├── encoder.py              # Sentence-BERT + projection MLP
│   │   ├── phi_lang.py             # φ_lang: language-conditioned graph transformer
│   │   ├── lang_dataset.py         # Language-annotated dataset + RLBench collector
│   │   └── train_lang.py           # Contrastive + MSE bottleneck alignment training
│   │
│   └── bimanual/                   # Bimanual (dual-arm) extension
│       ├── config.py               # BimanualIPConfig (cross-arm edge toggles)
│       ├── graph_builder.py        # Bimanual local / context / action graph builders
│       ├── networks.py             # BimanualSigma/Phi/PsiNetwork
│       ├── model.py                # BimanualGraphDiffusionPolicy
│       ├── pseudo_demo.py          # Dual-arm pseudo-demo generation
│       ├── dataset.py              # BimanualPseudoDemoDataset + PerAct2Dataset
│       └── train.py                # Bimanual training pipeline
│
├── scripts/
│   └── collect_bimanual_demos.py   # Collect PerAct2 bimanual demonstrations
│
├── hpc/                            # NYU HPC / SLURM deployment scripts
│   ├── setup_overlay.sh            # One-time environment setup (Micromamba)
│   ├── setup_peract2.sh            # Install PerAct2 bimanual RLBench fork
│   ├── train_phase1.sbatch         # Phase 1: geometry encoder pre-training
│   ├── train_phase2.sbatch         # Phase 2: single-arm model training
│   ├── train_bimanual_phase2.sbatch# Phase 2: bimanual model training
│   ├── collect_peract2_data.sbatch # Collect PerAct2 bimanual demos
│   ├── collect_lang_data.sbatch    # Collect RLBench language-annotated data
│   ├── train_phase3.sbatch         # Phase 3: language transfer training
│   ├── submit_single.sbatch        # Submit single task evaluation
│   ├── submit_all_tasks.sh         # Batch submit all RLBench tasks
│   ├── run_sim.sh                  # Simulation runner (inside container)
│   └── interactive.sh              # Interactive debugging session
│
├── Dockerfile                      # Container: CUDA + CoppeliaSim + RLBench
├── environment.yml                 # Conda environment specification
└── download_weights.sh             # Download pre-trained weights
```

---

## Quick Start: Pre-trained Model (Inference Only)

```bash
./download_weights.sh
python deploy_sim.py --task_name='plate_out' --num_demos=2 --num_rollouts=10
```

---

## Single-Arm Training

### Environment Setup

> Python 3.10 is required (PyTorch 2.2.0 does not support 3.12+).

```bash
conda create -n ip_env python=3.10 -y
conda activate ip_env

# PyTorch + CUDA 11.8
conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 \
    -c pytorch -c nvidia -y

# PyTorch Geometric
conda install pyg==2.5.0 pytorch-scatter pytorch-cluster \
    -c pyg -c pytorch -c nvidia -y

# Scientific stack
conda install numpy==1.26.4 scipy scikit-learn pyyaml tqdm pillow -c conda-forge -y

# Pip packages
pip install open3d==0.18.0 trimesh pyquaternion sentence-transformers \
    diffusers==0.31.0 transformers==4.46.2 accelerate wandb gdown

# pyg-lib (optional, skip if it fails)
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

Verify:

```bash
python -c "
import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
from ip import IPConfig, GraphDiffusionPolicy; print('Single-arm model OK')
"
```

### Data: ShapeNet

Download [ShapeNetCore.v2](https://shapenet.org/) (~30GB, registration required). The directory should contain category folders (e.g. `02691156/`, `03001627/`, ...) with `.obj` meshes inside.

### Phase 1: Geometry Encoder (~2-4 hours)

```bash
python -m ip.train \
    --shapenet_root /path/to/ShapeNet \
    --save_dir ./checkpoints_ip \
    --phase 1 \
    --device cuda
```

Output: `checkpoints_ip/geo_encoder.pt`

### Phase 2: Full Model (~5 days)

```bash
python -m ip.train \
    --shapenet_root /path/to/ShapeNet \
    --save_dir ./checkpoints_ip \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 \
    --device cuda
```

Resume: `--resume ./checkpoints_ip/model_step100000.pt`

Output: `checkpoints_ip/model_final.pt`

### Phase 3: Language Transfer (~6-12 hours)

Requires CoppeliaSim + RLBench. See [CoppeliaSim Setup](#coppelasim-setup-for-rlbench--peract2).

```bash
# Collect language-annotated demos first
python -c "
from ip.lang.lang_dataset import collect_rlbench_lang_data
collect_rlbench_lang_data('./lang_data', demos_per_task=20, headless=True)
"

# Train
python -m ip.lang.train_lang \
    --ip_checkpoint ./checkpoints_ip/model_final.pt \
    --data_dir ./lang_data \
    --save_dir ./checkpoints_lang \
    --device cuda
```

---

## Bimanual (Dual-Arm) Training

The bimanual extension adds cross-arm coordination edges to the heterogeneous graph at all three levels (sigma/phi/psi). No existing single-arm code is modified.

### Environment Setup

Same as single-arm — no additional dependencies required.

```bash
# Verify bimanual imports
conda activate ip_env
python -c "
from ip.bimanual import BimanualIPConfig, BimanualGraphDiffusionPolicy
cfg = BimanualIPConfig()
model = BimanualGraphDiffusionPolicy(cfg)
total = sum(p.numel() for p in model.parameters())
print(f'Bimanual model: {total:,} params — OK')
"
```

### Phase 1: Geometry Encoder (shared with single-arm)

The geometry encoder is task-agnostic — reuse the same `geo_encoder.pt` from single-arm training. If you have already trained it, skip this step.

```bash
python -m ip.train \
    --shapenet_root /path/to/ShapeNet \
    --save_dir ./checkpoints_ip \
    --phase 1 \
    --device cuda
```

### Phase 2: Bimanual Model (~5-7 days)

This is the main training phase. It uses **ShapeNet pseudo-demonstrations** generated on-the-fly (no PerAct2 data needed yet).

```bash
python -m ip.bimanual.train \
    --shapenet_root /path/to/ShapeNet \
    --save_dir ./checkpoints_bimanual \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 \
    --device cuda
```

Run in background (prevents SSH disconnect from killing training):

```bash
mkdir -p logs
nohup python -m ip.bimanual.train \
    --shapenet_root /path/to/ShapeNet \
    --save_dir ./checkpoints_bimanual \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 \
    --device cuda \
    > logs/bimanual_phase2.log 2>&1 &

echo "PID: $!"
```

Monitor:

```bash
tail -f logs/bimanual_phase2.log     # training loss
watch -n 10 nvidia-smi               # GPU usage
```

Resume from checkpoint:

```bash
python -m ip.bimanual.train \
    --shapenet_root /path/to/ShapeNet \
    --save_dir ./checkpoints_bimanual \
    --encoder_ckpt ./checkpoints_ip/geo_encoder.pt \
    --phase 2 \
    --device cuda \
    --resume ./checkpoints_bimanual/bimanual_step100000.pt
```

Ablation flags:

| Flag | Effect |
|------|--------|
| `--no_coordinate_edges` | Disable cross-arm edges in sigma (local subgraph) |
| `--no_bimanual_edges` | Disable cross-arm edges in phi (context graph) |
| `--no_sync_edges` | Disable cross-arm edges in psi (action graph) |
| `--scene_frame world` | Use world frame instead of midpoint frame for scene encoding |

Output: `checkpoints_bimanual/bimanual_final.pt` (checkpoint every 10K steps)

### Collecting PerAct2 Data (for fine-tuning / evaluation)

PerAct2 bimanual demonstrations are used for fine-tuning or evaluating the trained model on specific bimanual tasks. This requires CoppeliaSim.

#### CoppeliaSim Setup (for RLBench / PerAct2)

```bash
# Download CoppeliaSim V4.1.0
cd /path/to/tools
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Add to ~/.bashrc
export COPPELIASIM_ROOT=/path/to/tools/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

#### Install PyRep + PerAct2 RLBench Fork

```bash
conda activate ip_env

# PyRep
cd /tmp && git clone https://github.com/stepjam/PyRep.git
cd PyRep && pip install -r requirements.txt && pip install . && cd ..

# PerAct2's bimanual RLBench fork
git clone https://github.com/markusgrotz/RLBench.git
cd RLBench && pip install -e . && cd ..
```

#### Collect Bimanual Demonstrations

```bash
# Start virtual display (if headless server)
Xvfb :99 -screen 0 1280x1024x24 &
export DISPLAY=:99

cd /path/to/Instant-Policy

# Collect 20 episodes for all 13 bimanual tasks
python scripts/collect_bimanual_demos.py \
    --save_dir ./peract2_data \
    --num_episodes 20

# Or collect specific tasks only
python scripts/collect_bimanual_demos.py \
    --save_dir ./peract2_data \
    --num_episodes 20 \
    --tasks coordinated_push_box coordinated_lift_tray handover_item
```

Output:

```
peract2_data/
├── coordinated_push_box/
│   ├── episode_0000.npz      # pcds, T_w_es_left, T_w_es_right, grips_left, grips_right
│   ├── episode_0001.npz
│   └── ...
├── coordinated_lift_tray/
│   └── ...
└── ... (13 bimanual tasks)
```

### Evaluation on PerAct2 Bimanual Tasks

The evaluation protocol is aligned with the Instant Policy paper (Section 4.1) and PerAct2 (Section 4.1):

- **Metric**: Task success rate (binary: succeeded or not).
- **Context**: N=2 live demonstrations collected at the start of evaluation.
- **Randomisation**: Object poses randomised at each rollout via `task.reset()`.
- **Execution**: Closed-loop — observe, predict T=8 actions, execute, repeat (up to 30 cycles).
- **Success**: Determined by RLBench's built-in task completion conditions (`terminate and reward > 0`).

#### Evaluate on a single task

```bash
python eval_bimanual.py \
    --checkpoint ./checkpoints_bimanual/bimanual_final.pt \
    --tasks coordinated_push_box \
    --num_rollouts 100 \
    --num_demos 2
```

#### Evaluate on all 13 PerAct2 tasks

```bash
python eval_bimanual.py \
    --checkpoint ./checkpoints_bimanual/bimanual_final.pt \
    --num_rollouts 100 \
    --save_results ./results/bimanual_eval.json
```

#### Quick test (fewer rollouts)

```bash
python eval_bimanual.py \
    --checkpoint ./checkpoints_bimanual/bimanual_final.pt \
    --tasks coordinated_push_box coordinated_lift_tray \
    --num_rollouts 10
```

#### Evaluation parameters

| Parameter | Default | Paper value | Description |
|-----------|---------|-------------|-------------|
| `--num_demos` | 2 | 2 | Demonstrations as in-context examples (N) |
| `--num_rollouts` | 100 | 100 | Rollouts per task |
| `--execution_horizon` | 8 | 8 | Actions executed per cycle (T) |
| `--num_traj_wp` | 10 | 10 | Demo waypoints after downsampling (L) |
| `--max_steps` | 30 | 30 | Max observe-predict-execute cycles |

#### Output format

Results are printed as a table and optionally saved as JSON:

```
==================================================
Bimanual Evaluation Results
==================================================
Task                                       SR
--------------------------------------------------
  coordinated_push_box                  57/100 (57%)
  coordinated_lift_ball                 50/100 (50%)
  coordinated_lift_tray                  4/100 (4%)
  ...
--------------------------------------------------
  Average                               16.8%
  Tasks evaluated: 13/13
==================================================
```

### Inference API

For custom integration (outside RLBench), use the model directly:

```python
import torch
from ip.bimanual import BimanualIPConfig, BimanualGraphDiffusionPolicy

# Load model
ckpt = torch.load('./checkpoints_bimanual/bimanual_final.pt')
model = BimanualGraphDiffusionPolicy(ckpt.get('cfg', BimanualIPConfig()))
model.load_state_dict(ckpt['model'])
model.cuda().eval()

# Build input: demonstrations + current observation
sample = {
    'demos': [demo_dict_1, demo_dict_2],   # each with pcds, T_w_es_left/right, grips_left/right
    'current': {
        'pcd': pcd_tensor,                  # (2048, 3) world frame
        'T_w_e_left':  T_left_tensor,       # (4, 4)
        'T_w_e_right': T_right_tensor,      # (4, 4)
        'grip_left': 1, 'grip_right': 1,
    },
}

# Predict
actions_l, grips_l, actions_r, grips_r = model.predict_actions(sample)
# actions_{l,r}: (T, 4, 4) relative SE(3) transforms
# grips_{l,r}:   (T,) binary gripper commands
```

---

## Training Summary

| Phase | Command | Input | Output | Time | GPU Memory |
|-------|---------|-------|--------|------|------------|
| Phase 1 (shared) | `python -m ip.train --phase 1` | ShapeNet | `geo_encoder.pt` | 2-4h | ~4 GB |
| Phase 2 (single-arm) | `python -m ip.train --phase 2` | ShapeNet + Phase 1 | `model_final.pt` | ~5 days | ~10 GB |
| Phase 2 (bimanual) | `python -m ip.bimanual.train --phase 2` | ShapeNet + Phase 1 | `bimanual_final.pt` | ~7 days | ~18 GB |
| Phase 3 (language) | `python -m ip.lang.train_lang` | Phase 2 + RLBench | `phi_lang_final.pt` | 6-12h | ~12 GB |

---

## HPC Deployment (NYU Greene)

See `hpc/` directory for SLURM scripts. Key commands:

```bash
# One-time setup
bash hpc/setup_overlay.sh

# Single-arm training
sbatch hpc/train_phase1.sbatch
sbatch hpc/train_phase2.sbatch

# Bimanual training
sbatch hpc/train_bimanual_phase2.sbatch

# PerAct2 data collection
bash hpc/setup_peract2.sh    # install bimanual RLBench fork
sbatch hpc/collect_peract2_data.sbatch
```

---

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `No module named 'torch'` | Python 3.12+ | Recreate env with `python=3.10` |
| `torch-scatter` install fails | PyTorch/CUDA mismatch | Use `%2B` instead of `+` in wheel URLs |
| `pyg-lib` install fails | Not required | Skip it |
| Training loss stays flat | Missing `geo_encoder.pt` | Run Phase 1 first, check `--encoder_ckpt` path |
| CUDA OOM on bimanual | Batch too large or too many demos | Reduce `max_demos` in config or use fewer demo waypoints |
| PerAct2 task not found | Wrong RLBench fork | Install PerAct2's fork: `markusgrotz/RLBench` |
| `Xvfb` fails | No display for CoppeliaSim | `Xvfb :99 -screen 0 1280x1024x24 & export DISPLAY=:99` |

## Citing

```
@inproceedings{vosylius2024instant,
  title={Instant Policy: In-Context Imitation Learning via Graph Diffusion},
  author={Vosylius, Vitalis and Johns, Edward},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

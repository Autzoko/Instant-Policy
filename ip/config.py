"""
Instant Policy configuration.
All hyperparameters from the paper (Appendix C, E) centralised here.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class IPConfig:
    # ── Geometry encoder (Appendix A) ──────────────────────────────
    geo_feat_dim: int = 512          # output feature dim of encoder
    num_scene_nodes: int = 16        # M centroids from FPS
    num_pcd_points: int = 2048       # input point cloud size
    geo_freq_bands: int = 10         # NeRF position encoding bands (2^0..2^9)

    # SA layer 1
    sa1_npoint: int = 128
    sa1_radius: float = 0.1
    sa1_nsample: int = 32
    sa1_mlp: List[int] = field(default_factory=lambda: [64, 128, 256])

    # SA layer 2
    sa2_npoint: int = 16             # == num_scene_nodes
    sa2_radius: float = 0.3
    sa2_nsample: int = 64
    sa2_mlp: List[int] = field(default_factory=lambda: [256, 256, 512])

    # Occupancy decoder (Appendix A)
    occ_decoder_layers: int = 8
    occ_decoder_dim: int = 256

    # ── Gripper ────────────────────────────────────────────────────
    num_gripper_keypoints: int = 6
    gripper_feat_dim: int = 64       # node-distinction embedding dim
    gripper_state_embed_dim: int = 64

    # ── Graph transformer (Appendix C, Eq. 3) ─────────────────────
    hidden_dim: int = 1024
    num_heads: int = 16
    head_dim: int = 64               # hidden_dim // num_heads
    num_layers: int = 2              # layers per network (sigma/phi/psi)
    edge_freq_bands: int = 10       # D for edge positional encoding
    edge_dim: int = 60               # 3 * 2 * D

    # ── Diffusion (Section 3.3) ────────────────────────────────────
    num_diffusion_steps_train: int = 20   # K_train (higher for training)
    num_diffusion_steps_infer: int = 4    # K at test time
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # ── Trajectory ─────────────────────────────────────────────────
    num_traj_waypoints: int = 10     # L (demo length after downsampling)
    pred_horizon: int = 8           # T (number of future actions)
    max_demos: int = 5               # N max during training

    # ── Normalisation (Appendix E) ─────────────────────────────────
    max_translation: float = 0.01    # 1 cm between consecutive steps
    max_rotation_deg: float = 3.0    # 3 degrees
    max_rotation_rad: float = 0.05236  # 3 deg in radians

    # ── Training (Appendix E) ──────────────────────────────────────
    lr: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_optim_steps: int = 2_500_000
    cooldown_steps: int = 50_000
    use_fp16: bool = True
    grad_clip: float = 1.0

    # ── Data augmentation (Appendix D, E) ──────────────────────────
    gripper_flip_prob: float = 0.10  # flip gripper state probability
    local_perturb_prob: float = 0.30 # local disturbance probability

    # ── Language transfer (Appendix J) ─────────────────────────────
    sbert_model: str = "all-MiniLM-L6-v2"
    sbert_dim: int = 384             # output dim of all-MiniLM-L6-v2
    lang_lr: float = 1e-4
    lang_temperature: float = 0.07   # InfoNCE temperature
    lang_lambda_mse: float = 1.0     # weight of MSE auxiliary loss
    lang_train_steps: int = 100_000

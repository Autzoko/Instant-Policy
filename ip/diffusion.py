"""
SE(3) graph diffusion process (Section 3.3, Appendix B).

Training (forward process):
  1. Project T_EA to se(3) via Logmap → normalise → add DDPM noise → Expmap back.
  2. Construct graph with noisy actions.
  3. Network predicts flow (denoising direction) for each action gripper node.

Inference (reverse process, DDIM):
  1. Sample noisy actions from N(0, I).
  2. Iteratively denoise using predicted flow + SVD alignment (Eq. 4-5).
  3. Extract final SE(3) actions.
"""
import torch
import torch.nn as nn
import math
from typing import Tuple

from .se3_utils import (
    se3_log_map, se3_exp_map,
    normalize_se3, unnormalize_se3,
    svd_align, transform_points,
)
from .config import IPConfig


# ──────────────────────────────────────────────────────────────────────
# Noise schedule (linear β schedule, Ho et al. 2020)
# ──────────────────────────────────────────────────────────────────────

class NoiseSchedule:
    """Linear variance schedule for DDPM."""

    def __init__(self, num_steps: int, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device='cpu'):
        self.num_steps = num_steps
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        # For convenience
        self.sqrt_alpha_cumprod = alpha_cumprod.sqrt()
        self.sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod).sqrt()

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.sqrt_alpha_cumprod = self.sqrt_alpha_cumprod.to(device)
        self.sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod.to(device)
        return self


# ──────────────────────────────────────────────────────────────────────
# Forward diffusion: add noise to SE(3) actions  (Appendix B)
# ──────────────────────────────────────────────────────────────────────

def forward_diffusion_se3(
    T_EA: torch.Tensor,          # (B, T, 4, 4)  ground truth actions
    grip: torch.Tensor,          # (B, T)         gripper states {0, 1}
    k: int,                      # diffusion step (1..K)
    schedule: NoiseSchedule,
    max_rot: float,
    max_trans: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add noise at diffusion step k.

    Returns:
        T_EA_k:     (B, T, 4, 4) noisy SE(3) actions
        grip_k:     (B, T)       noisy gripper states (continuous)
        noise_se3:  (B, T, 6)    the noise added in se(3) space (target for flow)
        noise_grip: (B, T)       noise added to gripper
    """
    B, T_pred = T_EA.shape[:2]
    device = T_EA.device

    # 1. T_EA → se(3)
    xi = se3_log_map(T_EA.reshape(-1, 4, 4)).reshape(B, T_pred, 6)  # (B,T,6)

    # 2. Normalise to [-1, 1]
    xi_norm = normalize_se3(xi, max_rot, max_trans)

    # 3. DDPM forward: x_k = √ᾱ_k x_0 + √(1-ᾱ_k) ε
    sqrt_alpha = schedule.sqrt_alpha_cumprod[k - 1]
    sqrt_one_minus = schedule.sqrt_one_minus_alpha_cumprod[k - 1]
    noise_se3 = torch.randn_like(xi_norm)
    xi_noisy = sqrt_alpha * xi_norm + sqrt_one_minus * noise_se3

    # 4. Unnormalise and Expmap back
    xi_noisy_unnorm = unnormalize_se3(xi_noisy, max_rot, max_trans)
    T_EA_k = se3_exp_map(xi_noisy_unnorm.reshape(-1, 6)).reshape(B, T_pred, 4, 4)

    # 5. Gripper noise (treat as continuous, same schedule)
    grip_float = grip.float() * 2 - 1  # map {0,1} → {-1,1}
    noise_grip = torch.randn_like(grip_float)
    grip_k = sqrt_alpha * grip_float + sqrt_one_minus * noise_grip

    return T_EA_k, grip_k, noise_se3, noise_grip


# ──────────────────────────────────────────────────────────────────────
# Compute ground-truth flow targets  (Section 3.3)
# ──────────────────────────────────────────────────────────────────────

def compute_flow_targets(
    T_EA_0: torch.Tensor,        # (B, T, 4, 4) clean actions
    T_EA_k: torch.Tensor,        # (B, T, 4, 4) noisy actions
    grip_0: torch.Tensor,        # (B, T)       clean gripper
    grip_k: torch.Tensor,        # (B, T)       noisy gripper (continuous)
    kp: torch.Tensor,            # (K, 3)       gripper keypoints in EE frame
) -> torch.Tensor:
    """
    Compute per-node flow predictions as targets for training.

    Flow decomposition (Section 3.3):
      ∇p_t = t_EA^0 - t_EA^k          (translation flow)
      ∇p_r = R_EA^0 × p_kp - R_EA^k × p_kp  (rotation flow)
      ∇p   = ∇p_t + ∇p_r

    Returns: (B, T, K, 7)  [∇p_t(3) + ∇p_r(3) + ∇grip(1)]
    """
    B, T_pred = T_EA_0.shape[:2]
    K = kp.shape[0]

    # Translation flow: t_0 - t_k
    t_0 = T_EA_0[..., :3, 3]     # (B, T, 3)
    t_k = T_EA_k[..., :3, 3]     # (B, T, 3)
    dp_t = t_0 - t_k              # (B, T, 3)
    dp_t = dp_t.unsqueeze(-2).expand(-1, -1, K, -1)  # (B, T, K, 3)

    # Rotation flow: R_0 × p_kp - R_k × p_kp
    R_0 = T_EA_0[..., :3, :3]    # (B, T, 3, 3)
    R_k = T_EA_k[..., :3, :3]
    kp_exp = kp.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 3)
    rot_0 = torch.einsum('btij,btkj->btki', R_0, kp_exp.expand(B, T_pred, -1, -1))
    rot_k = torch.einsum('btij,btkj->btki', R_k, kp_exp.expand(B, T_pred, -1, -1))
    dp_r = rot_0 - rot_k          # (B, T, K, 3)

    # Gripper gradient
    grip_0_cont = grip_0.float() * 2 - 1  # {0,1} → {-1,1}
    dg = (grip_0_cont - grip_k).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 1)
    # (B, T, K, 1)

    return torch.cat([dp_t, dp_r, dg], dim=-1)  # (B, T, K, 7)


# ──────────────────────────────────────────────────────────────────────
# DDIM reverse step  (Eq. 4, Song et al. 2020)
# ──────────────────────────────────────────────────────────────────────

def ddim_reverse_step(
    gripper_pos_k: torch.Tensor,     # (T, K, 3) noisy action keypoint positions
    predicted_flow: torch.Tensor,    # (T, K, 7) [dp_t(3), dp_r(3), dg(1)]
    grip_k: torch.Tensor,           # (T,) noisy gripper
    k: int,                          # current diffusion step index in schedule
    k_prev: int,                     # previous (target) step index in schedule
    schedule: NoiseSchedule,
    kp: torch.Tensor,               # (K, 3) reference keypoints
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One DDIM denoising step (Eq. 4, Song et al. 2020).

    k and k_prev are indices into the schedule's alpha_cumprod array.
    When subsampling steps (e.g. [19,14,9,4,0]), k_prev is the NEXT
    index in that subsampled sequence, NOT simply k-1.

    Returns:
        T_step:          (T, 4, 4)  incremental SE(3) transform this step
        gripper_pos_km1: (T, K, 3)  updated keypoint positions
        grip_km1:        (T,)       updated gripper
    """
    dp_t = predicted_flow[..., :3]   # (T, K, 3)
    dp_r = predicted_flow[..., 3:6]  # (T, K, 3)
    dg = predicted_flow[..., 6]      # (T, K)

    # 1. Estimate clean positions
    p_hat_0 = gripper_pos_k + dp_t + dp_r   # (T, K, 3)

    # 2. DDIM step (Eq. 4)
    alpha_k = schedule.alpha_cumprod[k]
    if k_prev >= 0:
        alpha_km1 = schedule.alpha_cumprod[k_prev]
    else:
        alpha_km1 = torch.tensor(1.0, device=gripper_pos_k.device)

    coeff_pred = alpha_km1.sqrt()
    coeff_noise = ((1 - alpha_km1) / (1 - alpha_k)).sqrt()
    coeff_orig = alpha_k.sqrt()

    gripper_pos_km1 = coeff_pred * p_hat_0 + \
        coeff_noise * (gripper_pos_k - coeff_orig * p_hat_0)

    # 3. SVD alignment: find T mapping p_k → p_{k-1}
    T_step = svd_align(gripper_pos_km1, gripper_pos_k)  # (T, 4, 4)

    # 4. Gripper denoising (simple interpolation)
    dg_mean = dg.mean(dim=-1)  # (T,) average over keypoints
    grip_hat_0 = grip_k + dg_mean
    grip_km1 = coeff_pred * grip_hat_0 + \
        coeff_noise * (grip_k - coeff_orig * grip_hat_0)

    return T_step, gripper_pos_km1, grip_km1


# ──────────────────────────────────────────────────────────────────────
# Full denoising loop
# ──────────────────────────────────────────────────────────────────────

def full_denoise(
    initial_pos: torch.Tensor,       # (T, K, 3) sampled from N(0,σ)
    initial_grip: torch.Tensor,      # (T,) sampled from N(0,1)
    predict_fn,                      # callable(pos, grip, k) → flow (T,K,7)
    schedule: NoiseSchedule,
    kp: torch.Tensor,
    num_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the full DDIM denoising loop.

    predict_fn: given current action positions (T,K,3), grip (T,),
                and step k, returns predicted flow (T,K,7).
                This encapsulates the full σ→φ→ψ pipeline.

    Returns:
        actions: (T, 4, 4) final SE(3) actions
        grips:   (T,)       final gripper commands
    """
    pos = initial_pos
    grip = initial_grip
    # Accumulate SE(3) transformations
    T_accum = torch.eye(4, device=pos.device, dtype=pos.dtype).unsqueeze(0)
    T_accum = T_accum.expand(pos.shape[0], -1, -1).clone()

    step_indices = torch.linspace(
        schedule.num_steps - 1, 0, num_steps, device=pos.device
    ).long()

    for i, k in enumerate(step_indices):
        k_int = k.item()
        k_prev_int = step_indices[i + 1].item() if i + 1 < len(step_indices) else -1
        flow = predict_fn(pos, grip, k_int)
        T_step, pos, grip = ddim_reverse_step(
            pos, flow, grip, k_int, k_prev_int, schedule, kp
        )
        T_accum = T_step @ T_accum

    # Discretise gripper: > 0 → open (1), ≤ 0 → closed (0)
    grips = (grip > 0).float()

    return T_accum, grips


# ──────────────────────────────────────────────────────────────────────
# Training loss  (MSE on flow predictions)
# ──────────────────────────────────────────────────────────────────────

def diffusion_loss(
    predicted_flow: torch.Tensor,    # (B, T, K, 7)
    target_flow: torch.Tensor,       # (B, T, K, 7)
) -> torch.Tensor:
    """MSE loss on per-node flow predictions (Appendix B)."""
    return ((predicted_flow - target_flow) ** 2).mean()

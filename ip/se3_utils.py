"""
SE(3) utilities for Instant Policy.
Provides logmap / expmap between SE(3) and se(3), SVD-based alignment,
and noise injection on the SE(3) manifold (Appendix B).
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────
# SO(3) operations
# ──────────────────────────────────────────────────────────────────────

def skew(v: torch.Tensor) -> torch.Tensor:
    """v: (..., 3) -> skew-symmetric matrix (..., 3, 3)."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        z,        -v[..., 2],  v[..., 1],
        v[..., 2],  z,        -v[..., 0],
        -v[..., 1], v[..., 0],  z,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def so3_exp_map(omega: torch.Tensor) -> torch.Tensor:
    """Rodrigues: omega (..., 3) -> R (..., 3, 3)."""
    theta = omega.norm(dim=-1, keepdim=True).unsqueeze(-1)  # (...,1,1)
    theta = theta.clamp(min=1e-8)
    K = skew(omega)                                          # (...,3,3)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    R = I + (torch.sin(theta) / theta) * K + \
        ((1 - torch.cos(theta)) / (theta ** 2)) * (K @ K)
    return R


def so3_log_map(R: torch.Tensor) -> torch.Tensor:
    """R (..., 3, 3) -> omega (..., 3)."""
    # cos(theta) = (tr(R) - 1) / 2
    cos_theta = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)  # (...,)
    # sin(theta) near 0 → use Taylor expansion
    safe_theta = theta.clamp(min=1e-8)
    factor = theta / (2 * torch.sin(safe_theta))
    # extract skew part
    omega = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    return factor.unsqueeze(-1) * omega


# ──────────────────────────────────────────────────────────────────────
# SE(3) log / exp
# ──────────────────────────────────────────────────────────────────────

def se3_log_map(T: torch.Tensor) -> torch.Tensor:
    """T (..., 4, 4) -> xi (..., 6)  [omega(3), v(3)]."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    omega = so3_log_map(R)           # (..., 3)
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    K = skew(omega)
    # V^{-1} for translational component
    half_theta = theta / 2
    I = torch.eye(3, device=T.device, dtype=T.dtype)
    V_inv = I - 0.5 * K + \
        (1 / (theta ** 2).unsqueeze(-1)) * \
        (1 - (theta * torch.cos(half_theta) /
              (2 * torch.sin(half_theta.clamp(min=1e-8)))).unsqueeze(-1)) * (K @ K)
    v = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([omega, v], dim=-1)


def se3_exp_map(xi: torch.Tensor) -> torch.Tensor:
    """xi (..., 6) [omega(3), v(3)] -> T (..., 4, 4)."""
    omega = xi[..., :3]
    v = xi[..., 3:]
    R = so3_exp_map(omega)           # (..., 3, 3)
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    K = skew(omega)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype)
    V = I + ((1 - torch.cos(theta)) / (theta ** 2)).unsqueeze(-1) * K + \
        ((theta - torch.sin(theta)) / (theta ** 3)).unsqueeze(-1) * (K @ K)
    t = (V @ v.unsqueeze(-1)).squeeze(-1)
    T = torch.zeros(*xi.shape[:-1], 4, 4, device=xi.device, dtype=xi.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# ──────────────────────────────────────────────────────────────────────
# Normalise / unnormalise se(3) vectors to [-1, 1]  (Appendix E)
# ──────────────────────────────────────────────────────────────────────

def normalize_se3(xi: torch.Tensor,
                  max_rot: float, max_trans: float) -> torch.Tensor:
    """xi (...,6) [omega(3), v(3)] -> normalised to [-1,1]."""
    out = xi.clone()
    out[..., :3] = out[..., :3] / max_rot
    out[..., 3:] = out[..., 3:] / max_trans
    return out.clamp(-1, 1)


def unnormalize_se3(xi_norm: torch.Tensor,
                    max_rot: float, max_trans: float) -> torch.Tensor:
    """Inverse of normalize_se3."""
    out = xi_norm.clone()
    out[..., :3] = out[..., :3] * max_rot
    out[..., 3:] = out[..., 3:] * max_trans
    return out


# ──────────────────────────────────────────────────────────────────────
# SVD rigid alignment  (Eq. 5, Arun et al. 1987)
# ──────────────────────────────────────────────────────────────────────

def svd_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Find T ∈ SE(3) s.t.  P ≈ T @ Q  (least-squares).
    P, Q: (..., N, 3)  — corresponding point sets.
    Returns T: (..., 4, 4).
    """
    p_mean = P.mean(dim=-2, keepdim=True)
    q_mean = Q.mean(dim=-2, keepdim=True)
    P_c = P - p_mean
    Q_c = Q - q_mean
    H = Q_c.transpose(-1, -2) @ P_c   # (..., 3, 3)
    U, S, Vh = torch.linalg.svd(H)
    # Handle reflection
    d = torch.det(Vh.transpose(-1, -2) @ U.transpose(-1, -2))
    sign = torch.ones_like(d)
    sign[d < 0] = -1
    diag = torch.zeros(*d.shape, 3, 3, device=P.device, dtype=P.dtype)
    diag[..., 0, 0] = 1
    diag[..., 1, 1] = 1
    diag[..., 2, 2] = sign
    R = Vh.transpose(-1, -2) @ diag @ U.transpose(-1, -2)
    t = p_mean.squeeze(-2) - (R @ q_mean.squeeze(-2).unsqueeze(-1)).squeeze(-1)
    T = torch.zeros(*d.shape, 4, 4, device=P.device, dtype=P.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def transform_points(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Apply T (...,4,4) to points pts (...,N,3) -> (...,N,3)."""
    R = T[..., :3, :3]   # (..., 3, 3)
    t = T[..., :3, 3]    # (..., 3)
    return (pts @ R.transpose(-1, -2)) + t.unsqueeze(-2)


def identity_se3(batch_shape, device='cpu', dtype=torch.float32):
    """Return batch of identity 4x4 matrices."""
    T = torch.eye(4, device=device, dtype=dtype)
    return T.expand(*batch_shape, 4, 4).contiguous()


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """Invert SE(3) matrix. T (...,4,4) -> T_inv (...,4,4)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv

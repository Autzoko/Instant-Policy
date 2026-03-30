"""
Graph construction for Instant Policy (Section 3.2).

Builds three types of graph structures:
  1. Local graph  G_l^t  — scene nodes + gripper nodes for a single timestep
  2. Context graph G_c   — links demo gripper nodes to current gripper nodes
  3. Action graph  G_l^a — virtual local graphs for predicted future actions

All edge attributes use NeRF-like positional encoding of relative positions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

from .pos_encoding import nerf_positional_encoding
from .config import IPConfig

# Type aliases
NodeType = str
EdgeTriple = Tuple[str, str, str]


# ──────────────────────────────────────────────────────────────────────
# Default gripper keypoints (6 points, Robotiq 2F-85 style, in metres)
# These are fixed in the end-effector frame.
# ──────────────────────────────────────────────────────────────────────

DEFAULT_GRIPPER_KEYPOINTS = torch.tensor([
    [0.000,  0.000,  0.000],    # palm centre
    [0.000,  0.000,  0.050],    # fingertip centre
    [0.000,  0.020,  0.050],    # left fingertip
    [0.000, -0.020,  0.050],    # right fingertip
    [0.000,  0.020,  0.020],    # left finger base
    [0.000, -0.020,  0.020],    # right finger base
], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────
# Gripper node feature builder
# ──────────────────────────────────────────────────────────────────────

class GripperNodeEncoder(nn.Module):
    """
    Builds gripper node features:
      F_g^i = [f_g^i, φ_g(s_g)]
    where f_g^i is a learned per-keypoint distinction embedding and
    φ_g embeds the binary gripper state.
    """

    def __init__(self, num_kp: int = 6,
                 distinction_dim: int = 64,
                 state_embed_dim: int = 64):
        super().__init__()
        # Learnable distinction embedding per keypoint index
        self.distinction_embed = nn.Embedding(num_kp, distinction_dim)
        # Gripper state embedding: 0=closed, 1=open
        self.state_embed = nn.Embedding(2, state_embed_dim)
        self.out_dim = distinction_dim + state_embed_dim

    def forward(self, grip_state: torch.Tensor) -> torch.Tensor:
        """
        grip_state: (B,) integer 0 or 1.
        Returns:    (B, num_kp, out_dim)
        """
        B = grip_state.shape[0]
        device = grip_state.device
        kp_idx = torch.arange(self.distinction_embed.num_embeddings, device=device)
        f_g = self.distinction_embed(kp_idx)         # (K, dist_dim)
        f_g = f_g.unsqueeze(0).expand(B, -1, -1)    # (B, K, dist_dim)
        s_g = self.state_embed(grip_state.long())    # (B, state_dim)
        s_g = s_g.unsqueeze(1).expand(-1, f_g.shape[1], -1)  # (B, K, state_dim)
        return torch.cat([f_g, s_g], dim=-1)         # (B, K, out_dim)


# ──────────────────────────────────────────────────────────────────────
# Edge construction helpers
# ──────────────────────────────────────────────────────────────────────

def _make_edge_attr(pos_src: torch.Tensor, pos_dst: torch.Tensor,
                    src_idx: torch.Tensor, dst_idx: torch.Tensor,
                    freq_bands: int) -> torch.Tensor:
    """
    Compute NeRF-encoded relative position for each edge.
    pos_src: (N_src, 3),  pos_dst: (N_dst, 3)
    src_idx, dst_idx: (E,)
    Returns: (E, 3*2*freq_bands)
    """
    rel = pos_dst[dst_idx] - pos_src[src_idx]  # dst relative to src
    return nerf_positional_encoding(rel, freq_bands)


def _fully_connected_edges(n_src: int, n_dst: int,
                           src_offset: int = 0,
                           dst_offset: int = 0,
                           device='cpu') -> torch.Tensor:
    """
    Returns (2, n_src * n_dst) fully-connected edge index.
    """
    src = torch.arange(n_src, device=device) + src_offset
    dst = torch.arange(n_dst, device=device) + dst_offset
    grid = torch.cartesian_prod(src, dst)  # (n_src*n_dst, 2)
    return grid.T  # (2, E)


# ──────────────────────────────────────────────────────────────────────
# Local graph builder
# ──────────────────────────────────────────────────────────────────────

def build_local_graph(
    scene_pos: torch.Tensor,       # (M, 3)   centroid positions
    scene_feat: torch.Tensor,      # (M, feat_dim)
    gripper_pos: torch.Tensor,     # (K, 3)   keypoint positions (EE frame)
    gripper_feat: torch.Tensor,    # (K, gripper_feat_dim)
    freq_bands: int = 10,
) -> dict:
    """
    Build a single local subgraph G_l.
    Returns dict with node_feats, node_pos, edge_index, edge_attr.
    """
    M = scene_pos.shape[0]
    K = gripper_pos.shape[0]
    device = scene_pos.device

    # Edges: every scene node → every gripper node
    ei = _fully_connected_edges(M, K, device=device)  # (2, M*K)
    ea = _make_edge_attr(scene_pos, gripper_pos, ei[0], ei[1], freq_bands)

    return {
        'node_feats': {'scene': scene_feat, 'gripper': gripper_feat},
        'node_pos': {'scene': scene_pos, 'gripper': gripper_pos},
        'edge_index': {('scene', 'observe', 'gripper'): ei},
        'edge_attr': {('scene', 'observe', 'gripper'): ea},
    }


# ──────────────────────────────────────────────────────────────────────
# Context graph builder  (Section 3.2: Context Representation)
# ──────────────────────────────────────────────────────────────────────

def build_context_edges(
    demo_gripper_pos: List[List[torch.Tensor]],
    current_gripper_pos: torch.Tensor,
    freq_bands: int = 10,
) -> Tuple[Dict, Dict]:
    """
    Build edges for the context graph φ.

    demo_gripper_pos: list of N demos, each a list of L tensors of shape (K, 3)
                      representing gripper keypoint world-frame positions at
                      each waypoint.
    current_gripper_pos: (K, 3)  current gripper keypoint world-frame positions.

    All gripper nodes (from demos + current) are flattened into a single list.
    Layout:  [demo_0_wp_0 ... demo_0_wp_{L-1}, demo_1_wp_0 ..., ..., current]
    Each block has K=6 nodes.

    Returns: (edge_index_dict, edge_attr_dict)
    """
    device = current_gripper_pos.device
    K = current_gripper_pos.shape[0]
    N = len(demo_gripper_pos)

    # Flatten all positions into one tensor
    all_pos_list = []
    for demo in demo_gripper_pos:
        for wp_pos in demo:
            all_pos_list.append(wp_pos)  # (K, 3)
    all_pos_list.append(current_gripper_pos)  # current is last block
    all_pos = torch.cat(all_pos_list, dim=0)  # (total_nodes, 3)

    L = len(demo_gripper_pos[0]) if N > 0 else 0
    num_demo_nodes = N * L * K
    current_offset = num_demo_nodes

    edge_index_dict = {}
    edge_attr_dict = {}
    temporal_src, temporal_dst = [], []
    context_src, context_dst = [], []

    for n in range(N):
        demo_base = n * L * K
        # ── Red edges: temporal within demo (consecutive waypoints) ──
        for l in range(L - 1):
            wp_offset = demo_base + l * K
            next_offset = demo_base + (l + 1) * K
            ei = _fully_connected_edges(K, K, wp_offset, next_offset, device)
            temporal_src.append(ei[0])
            temporal_dst.append(ei[1])

        # ── Grey edges: every demo gripper node → current gripper ──
        for l in range(L):
            wp_offset = demo_base + l * K
            ei = _fully_connected_edges(K, K, wp_offset, current_offset, device)
            context_src.append(ei[0])
            context_dst.append(ei[1])

    if temporal_src:
        t_src = torch.cat(temporal_src)
        t_dst = torch.cat(temporal_dst)
        ei_temporal = torch.stack([t_src, t_dst])
        ea_temporal = _make_edge_attr(all_pos, all_pos, t_src, t_dst, freq_bands)
        edge_index_dict[('gripper', 'temporal', 'gripper')] = ei_temporal
        edge_attr_dict[('gripper', 'temporal', 'gripper')] = ea_temporal

    if context_src:
        c_src = torch.cat(context_src)
        c_dst = torch.cat(context_dst)
        ei_context = torch.stack([c_src, c_dst])
        ea_context = _make_edge_attr(all_pos, all_pos, c_src, c_dst, freq_bands)
        edge_index_dict[('gripper', 'context', 'gripper')] = ei_context
        edge_attr_dict[('gripper', 'context', 'gripper')] = ea_context

    return edge_index_dict, edge_attr_dict


# ──────────────────────────────────────────────────────────────────────
# Action graph builder  (Section 3.2: Action Representation)
# ──────────────────────────────────────────────────────────────────────

def build_action_edges(
    current_gripper_pos: torch.Tensor,     # (K, 3)
    action_gripper_pos: List[torch.Tensor], # T tensors of (K, 3)
    freq_bands: int = 10,
    current_offset: int = 0,
    action_offset: int = 0,
) -> Tuple[Dict, Dict]:
    """
    Build temporal edges from current gripper to action gripper nodes,
    and between consecutive action steps.

    Layout: action nodes are at [action_offset .. action_offset + T*K)
    current nodes are at [current_offset .. current_offset + K)

    Returns: (edge_index_dict, edge_attr_dict)
    """
    K = current_gripper_pos.shape[0]
    T = len(action_gripper_pos)
    device = current_gripper_pos.device

    # Flatten all positions
    all_pos = torch.cat([current_gripper_pos] + action_gripper_pos, dim=0)
    # current: [0..K), action_0: [K..2K), action_1: [2K..3K), ...

    act_src, act_dst = [], []

    # Current → first action
    ei = _fully_connected_edges(K, K, 0, K, device)
    act_src.append(ei[0])
    act_dst.append(ei[1])

    # Action t → action t+1
    for t in range(T - 1):
        offset_t = (t + 1) * K
        offset_t1 = (t + 2) * K
        ei = _fully_connected_edges(K, K, offset_t, offset_t1, device)
        act_src.append(ei[0])
        act_dst.append(ei[1])

    a_src = torch.cat(act_src)
    a_dst = torch.cat(act_dst)
    ei_action = torch.stack([a_src, a_dst])
    ea_action = _make_edge_attr(all_pos, all_pos, a_src, a_dst, freq_bands)

    edge_index_dict = {('gripper', 'action', 'gripper'): ei_action}
    edge_attr_dict = {('gripper', 'action', 'gripper'): ea_action}
    return edge_index_dict, edge_attr_dict


# ──────────────────────────────────────────────────────────────────────
# Convenience: get gripper world-frame keypoint positions
# ──────────────────────────────────────────────────────────────────────

def gripper_keypoints_world(T_WE: torch.Tensor,
                            kp: torch.Tensor = None) -> torch.Tensor:
    """
    T_WE: (..., 4, 4)
    kp:   (K, 3) keypoints in EE frame (default: DEFAULT_GRIPPER_KEYPOINTS)
    Returns: (..., K, 3) keypoints in world frame.
    """
    if kp is None:
        kp = DEFAULT_GRIPPER_KEYPOINTS.to(T_WE.device, T_WE.dtype)
    R = T_WE[..., :3, :3]  # (..., 3, 3)
    t = T_WE[..., :3, 3]   # (..., 3)
    return (kp @ R.transpose(-1, -2)) + t.unsqueeze(-2)


def gripper_keypoints_ee(kp: torch.Tensor = None,
                         device='cpu', dtype=torch.float32) -> torch.Tensor:
    """Return the default keypoints in EE frame."""
    if kp is None:
        return DEFAULT_GRIPPER_KEYPOINTS.to(device=device, dtype=dtype)
    return kp.to(device=device, dtype=dtype)

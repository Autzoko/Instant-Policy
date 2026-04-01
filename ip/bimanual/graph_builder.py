"""
Bimanual graph construction for Instant Policy.

Extends the single-arm graph builders with dual-arm node types and
cross-arm coordination edges at every level:

  1. Local graph  G_l  -- scene + gripper_left + gripper_right
     Edge types: observe (scene->each gripper), coordinate (left<->right)

  2. Context graph G_c  -- demo gripper nodes (left & right) -> current
     Edge types: temporal (within-arm consecutive), context (demo->current),
                 bimanual (left<->right at same timestep)

  3. Action graph  G_a  -- current + future action nodes for both arms
     Edge types: action (within-arm temporal), sync (left<->right at same step)
"""
import torch
from typing import Dict, List, Tuple

from ..graph_builder import (
    _make_edge_attr,
    _fully_connected_edges,
    gripper_keypoints_world,
    gripper_keypoints_ee,
    DEFAULT_GRIPPER_KEYPOINTS,
    GripperNodeEncoder,
)

# Re-export for convenience
__all__ = [
    'build_bimanual_local_graph',
    'build_bimanual_context_edges',
    'build_bimanual_action_edges',
    'gripper_keypoints_world',
    'gripper_keypoints_ee',
    'DEFAULT_GRIPPER_KEYPOINTS',
    'GripperNodeEncoder',
]


# ──────────────────────────────────────────────────────────────────────
# Bimanual local graph builder
# ──────────────────────────────────────────────────────────────────────

def build_bimanual_local_graph(
    scene_pos: torch.Tensor,             # (M, 3)
    scene_feat: torch.Tensor,            # (M, feat_dim)
    gripper_left_pos: torch.Tensor,      # (K, 3)
    gripper_left_feat: torch.Tensor,     # (K, gripper_feat_dim)
    gripper_right_pos: torch.Tensor,     # (K, 3)
    gripper_right_feat: torch.Tensor,    # (K, gripper_feat_dim)
    freq_bands: int = 10,
    enable_coordinate_edges: bool = True,
) -> dict:
    """
    Build a bimanual local subgraph G_l.

    Node types:
      'scene'         -- M centroid nodes from the geometry encoder
      'gripper_left'  -- K keypoint nodes for the left arm
      'gripper_right' -- K keypoint nodes for the right arm

    Edge types:
      ('scene', 'observe', 'gripper_left')  -- scene informs left gripper
      ('scene', 'observe', 'gripper_right') -- scene informs right gripper
      ('gripper_left',  'coordinate', 'gripper_right') -- cross-arm (optional)
      ('gripper_right', 'coordinate', 'gripper_left')  -- cross-arm (optional)

    Returns dict with node_feats, node_pos, edge_index, edge_attr.
    """
    M = scene_pos.shape[0]
    K = gripper_left_pos.shape[0]
    device = scene_pos.device

    edge_index = {}
    edge_attr = {}

    # Scene -> left gripper (fully connected M*K edges)
    ei_sl = _fully_connected_edges(M, K, device=device)
    ea_sl = _make_edge_attr(scene_pos, gripper_left_pos, ei_sl[0], ei_sl[1], freq_bands)
    edge_index[('scene', 'observe', 'gripper_left')] = ei_sl
    edge_attr[('scene', 'observe', 'gripper_left')] = ea_sl

    # Scene -> right gripper (fully connected M*K edges)
    ei_sr = _fully_connected_edges(M, K, device=device)
    ea_sr = _make_edge_attr(scene_pos, gripper_right_pos, ei_sr[0], ei_sr[1], freq_bands)
    edge_index[('scene', 'observe', 'gripper_right')] = ei_sr
    edge_attr[('scene', 'observe', 'gripper_right')] = ea_sr

    # Cross-arm coordination edges (optional)
    if enable_coordinate_edges:
        # Left -> right
        ei_lr = _fully_connected_edges(K, K, device=device)
        ea_lr = _make_edge_attr(gripper_left_pos, gripper_right_pos,
                                ei_lr[0], ei_lr[1], freq_bands)
        edge_index[('gripper_left', 'coordinate', 'gripper_right')] = ei_lr
        edge_attr[('gripper_left', 'coordinate', 'gripper_right')] = ea_lr

        # Right -> left
        ei_rl = _fully_connected_edges(K, K, device=device)
        ea_rl = _make_edge_attr(gripper_right_pos, gripper_left_pos,
                                ei_rl[0], ei_rl[1], freq_bands)
        edge_index[('gripper_right', 'coordinate', 'gripper_left')] = ei_rl
        edge_attr[('gripper_right', 'coordinate', 'gripper_left')] = ea_rl

    return {
        'node_feats': {
            'scene': scene_feat,
            'gripper_left': gripper_left_feat,
            'gripper_right': gripper_right_feat,
        },
        'node_pos': {
            'scene': scene_pos,
            'gripper_left': gripper_left_pos,
            'gripper_right': gripper_right_pos,
        },
        'edge_index': edge_index,
        'edge_attr': edge_attr,
    }


# ──────────────────────────────────────────────────────────────────────
# Bimanual context graph builder
# ──────────────────────────────────────────────────────────────────────

def build_bimanual_context_edges(
    demo_gripper_pos_left: List[List[torch.Tensor]],
    demo_gripper_pos_right: List[List[torch.Tensor]],
    current_gripper_pos_left: torch.Tensor,
    current_gripper_pos_right: torch.Tensor,
    freq_bands: int = 10,
    enable_bimanual_edges: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Build edges for the bimanual context graph phi.

    Node layout per arm (left and right are independent node type arrays):
      [demo_0_wp_0, ..., demo_0_wp_{L-1}, demo_1_wp_0, ..., current]
      Each block has K nodes.  Total per arm: (N*L + 1) * K.

    Edge types:
      ('gripper_left',  'temporal',  'gripper_left')   -- within-demo left
      ('gripper_right', 'temporal',  'gripper_right')  -- within-demo right
      ('gripper_left',  'context',   'gripper_left')   -- demo->current left
      ('gripper_right', 'context',   'gripper_right')  -- demo->current right
      ('gripper_left',  'bimanual',  'gripper_right')  -- cross-arm per timestep
      ('gripper_right', 'bimanual',  'gripper_left')   -- cross-arm per timestep

    Returns: (edge_index_dict, edge_attr_dict)
    """
    device = current_gripper_pos_left.device
    K = current_gripper_pos_left.shape[0]
    N = len(demo_gripper_pos_left)
    L = len(demo_gripper_pos_left[0]) if N > 0 else 0

    # Flatten per-arm positions
    def _flatten_positions(demo_pos_list, current_pos):
        all_pos = []
        for demo in demo_pos_list:
            for wp_pos in demo:
                all_pos.append(wp_pos)
        all_pos.append(current_pos)
        return torch.cat(all_pos, dim=0)  # ((N*L+1)*K, 3)

    all_pos_left = _flatten_positions(demo_gripper_pos_left,
                                      current_gripper_pos_left)
    all_pos_right = _flatten_positions(demo_gripper_pos_right,
                                       current_gripper_pos_right)

    num_blocks = N * L + 1  # total timestep blocks
    current_block = num_blocks - 1
    current_offset = current_block * K

    edge_index_dict = {}
    edge_attr_dict = {}

    # ── Helper: build within-arm temporal + context edges ────────
    def _build_single_arm_edges(all_pos, arm_name):
        temporal_src, temporal_dst = [], []
        context_src, context_dst = [], []

        for n in range(N):
            demo_base = n * L * K
            # Temporal: consecutive waypoints within each demo
            for l in range(L - 1):
                wp_off = demo_base + l * K
                nxt_off = demo_base + (l + 1) * K
                ei = _fully_connected_edges(K, K, wp_off, nxt_off, device)
                temporal_src.append(ei[0])
                temporal_dst.append(ei[1])

            # Context: every demo waypoint -> current
            for l in range(L):
                wp_off = demo_base + l * K
                ei = _fully_connected_edges(K, K, wp_off, current_offset, device)
                context_src.append(ei[0])
                context_dst.append(ei[1])

        if temporal_src:
            t_src = torch.cat(temporal_src)
            t_dst = torch.cat(temporal_dst)
            ei_t = torch.stack([t_src, t_dst])
            ea_t = _make_edge_attr(all_pos, all_pos, t_src, t_dst, freq_bands)
            triple_t = (arm_name, 'temporal', arm_name)
            edge_index_dict[triple_t] = ei_t
            edge_attr_dict[triple_t] = ea_t

        if context_src:
            c_src = torch.cat(context_src)
            c_dst = torch.cat(context_dst)
            ei_c = torch.stack([c_src, c_dst])
            ea_c = _make_edge_attr(all_pos, all_pos, c_src, c_dst, freq_bands)
            triple_c = (arm_name, 'context', arm_name)
            edge_index_dict[triple_c] = ei_c
            edge_attr_dict[triple_c] = ea_c

    _build_single_arm_edges(all_pos_left, 'gripper_left')
    _build_single_arm_edges(all_pos_right, 'gripper_right')

    # ── Cross-arm bimanual edges (at each timestep block) ────────
    if enable_bimanual_edges:
        bim_lr_src, bim_lr_dst = [], []
        bim_rl_src, bim_rl_dst = [], []

        for b in range(num_blocks):
            block_off = b * K
            # Left -> right  (indices are within each arm's own node array)
            ei = _fully_connected_edges(K, K, block_off, block_off, device)
            bim_lr_src.append(ei[0])
            bim_lr_dst.append(ei[1])
            # Right -> left
            bim_rl_src.append(ei[0])
            bim_rl_dst.append(ei[1])

        lr_src = torch.cat(bim_lr_src)
        lr_dst = torch.cat(bim_lr_dst)
        ei_lr = torch.stack([lr_src, lr_dst])
        # Cross-arm edge attrs use relative position between left and right
        ea_lr = _make_edge_attr(all_pos_left, all_pos_right,
                                lr_src, lr_dst, freq_bands)
        edge_index_dict[('gripper_left', 'bimanual', 'gripper_right')] = ei_lr
        edge_attr_dict[('gripper_left', 'bimanual', 'gripper_right')] = ea_lr

        rl_src = torch.cat(bim_rl_src)
        rl_dst = torch.cat(bim_rl_dst)
        ei_rl = torch.stack([rl_src, rl_dst])
        ea_rl = _make_edge_attr(all_pos_right, all_pos_left,
                                rl_src, rl_dst, freq_bands)
        edge_index_dict[('gripper_right', 'bimanual', 'gripper_left')] = ei_rl
        edge_attr_dict[('gripper_right', 'bimanual', 'gripper_left')] = ea_rl

    return edge_index_dict, edge_attr_dict


# ──────────────────────────────────────────────────────────────────────
# Bimanual action graph builder
# ──────────────────────────────────────────────────────────────────────

def build_bimanual_action_edges(
    current_gripper_pos_left: torch.Tensor,        # (K, 3)
    action_gripper_pos_left: List[torch.Tensor],   # T tensors of (K, 3)
    current_gripper_pos_right: torch.Tensor,       # (K, 3)
    action_gripper_pos_right: List[torch.Tensor],  # T tensors of (K, 3)
    freq_bands: int = 10,
    enable_sync_edges: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Build edges for the bimanual action graph psi.

    Node layout per arm:
      [current_K, action_0_K, ..., action_{T-1}_K]  =  (1+T)*K nodes

    Edge types:
      ('gripper_left',  'action', 'gripper_left')  -- within-arm temporal
      ('gripper_right', 'action', 'gripper_right') -- within-arm temporal
      ('gripper_left',  'sync',   'gripper_right') -- cross-arm per step
      ('gripper_right', 'sync',   'gripper_left')  -- cross-arm per step

    Returns: (edge_index_dict, edge_attr_dict)
    """
    K = current_gripper_pos_left.shape[0]
    T = len(action_gripper_pos_left)
    device = current_gripper_pos_left.device

    # Flatten per-arm positions: [current, action_0, ..., action_{T-1}]
    all_pos_left = torch.cat(
        [current_gripper_pos_left] + action_gripper_pos_left, dim=0
    )
    all_pos_right = torch.cat(
        [current_gripper_pos_right] + action_gripper_pos_right, dim=0
    )

    edge_index_dict = {}
    edge_attr_dict = {}

    # ── Within-arm action edges ──────────────────────────────────
    def _build_action_edges(all_pos, arm_name):
        src_list, dst_list = [], []
        # Current -> first action
        ei = _fully_connected_edges(K, K, 0, K, device)
        src_list.append(ei[0])
        dst_list.append(ei[1])
        # Action t -> action t+1
        for t in range(T - 1):
            off_t = (t + 1) * K
            off_t1 = (t + 2) * K
            ei = _fully_connected_edges(K, K, off_t, off_t1, device)
            src_list.append(ei[0])
            dst_list.append(ei[1])

        a_src = torch.cat(src_list)
        a_dst = torch.cat(dst_list)
        ei_a = torch.stack([a_src, a_dst])
        ea_a = _make_edge_attr(all_pos, all_pos, a_src, a_dst, freq_bands)
        triple = (arm_name, 'action', arm_name)
        edge_index_dict[triple] = ei_a
        edge_attr_dict[triple] = ea_a

    _build_action_edges(all_pos_left, 'gripper_left')
    _build_action_edges(all_pos_right, 'gripper_right')

    # ── Cross-arm sync edges (at each step: current + T actions) ─
    if enable_sync_edges:
        sync_lr_src, sync_lr_dst = [], []
        sync_rl_src, sync_rl_dst = [], []

        for step in range(1 + T):
            off = step * K
            ei = _fully_connected_edges(K, K, off, off, device)
            sync_lr_src.append(ei[0])
            sync_lr_dst.append(ei[1])
            sync_rl_src.append(ei[0])
            sync_rl_dst.append(ei[1])

        lr_src = torch.cat(sync_lr_src)
        lr_dst = torch.cat(sync_lr_dst)
        ei_lr = torch.stack([lr_src, lr_dst])
        ea_lr = _make_edge_attr(all_pos_left, all_pos_right,
                                lr_src, lr_dst, freq_bands)
        edge_index_dict[('gripper_left', 'sync', 'gripper_right')] = ei_lr
        edge_attr_dict[('gripper_left', 'sync', 'gripper_right')] = ea_lr

        rl_src = torch.cat(sync_rl_src)
        rl_dst = torch.cat(sync_rl_dst)
        ei_rl = torch.stack([rl_src, rl_dst])
        ea_rl = _make_edge_attr(all_pos_right, all_pos_left,
                                rl_src, rl_dst, freq_bands)
        edge_index_dict[('gripper_right', 'sync', 'gripper_left')] = ei_rl
        edge_attr_dict[('gripper_right', 'sync', 'gripper_left')] = ea_rl

    return edge_index_dict, edge_attr_dict

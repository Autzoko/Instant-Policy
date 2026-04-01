"""
Bimanual sub-networks sigma, phi, psi (extending Eq. 6).

Each network wraps a HeteroGraphTransformer instantiated with the
bimanual node types and edge triples.  Cross-arm edges enable
implicit coordination between the left and right arms via graph
message passing.

Weight sharing:
  - sigma:  scene_proj is shared; gripper_proj is shared between arms
            (same gripper geometry).  HGT edge-type weights are separate.
  - phi:    single HGT with 6 edge triples (temporal/context per arm + bimanual).
  - psi:    single HGT with 4 edge triples (action per arm + sync).
            Denoising head can be shared or separate (configurable).
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from ..config import IPConfig
from ..graph_transformer import HeteroGraphTransformer
from ..networks import DiffusionStepEmbedding  # noqa: re-export
from .config import BimanualIPConfig


class BimanualSigmaNetwork(nn.Module):
    """
    sigma: processes bimanual local subgraphs G_l.

    Node types: 'scene', 'gripper_left', 'gripper_right'
    Edge types: observe (scene->each gripper),
                coordinate (left<->right, optional)
    """

    def __init__(self, cfg: BimanualIPConfig):
        super().__init__()
        self.cfg = cfg

        # Input projections (shared between arms)
        self.scene_proj = nn.Linear(cfg.geo_feat_dim, cfg.hidden_dim)
        gripper_in_dim = cfg.gripper_feat_dim + cfg.gripper_state_embed_dim
        self.gripper_proj = nn.Linear(gripper_in_dim, cfg.hidden_dim)

        # Build edge triples list
        edge_triples = [
            ('scene', 'observe', 'gripper_left'),
            ('scene', 'observe', 'gripper_right'),
        ]
        if cfg.enable_coordinate_edges:
            edge_triples.extend([
                ('gripper_left', 'coordinate', 'gripper_right'),
                ('gripper_right', 'coordinate', 'gripper_left'),
            ])

        self.gnn = HeteroGraphTransformer(
            node_types=['scene', 'gripper_left', 'gripper_right'],
            edge_triples=edge_triples,
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )

    def forward(self, local_graph: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        local_graph: output of build_bimanual_local_graph().
        Returns: (gripper_left_features, gripper_right_features)
                 each (K, hidden_dim).
        """
        nf = local_graph['node_feats']
        scene_h = self.scene_proj(nf['scene'])
        left_h = self.gripper_proj(nf['gripper_left'])
        right_h = self.gripper_proj(nf['gripper_right'])

        node_feats = {
            'scene': scene_h,
            'gripper_left': left_h,
            'gripper_right': right_h,
        }
        out = self.gnn(node_feats,
                       local_graph['edge_index'],
                       local_graph['edge_attr'])
        return out['gripper_left'], out['gripper_right']


class BimanualPhiNetwork(nn.Module):
    """
    phi: processes the bimanual context graph G_c.

    Aggregates demo information from both arms into bottleneck
    representations for the current left and right grippers.

    Node types: 'gripper_left', 'gripper_right'
    Edge types: temporal (within-arm), context (demo->current),
                bimanual (cross-arm at same timestep)
    """

    def __init__(self, cfg: BimanualIPConfig):
        super().__init__()
        self.cfg = cfg

        edge_triples = [
            ('gripper_left', 'temporal', 'gripper_left'),
            ('gripper_right', 'temporal', 'gripper_right'),
            ('gripper_left', 'context', 'gripper_left'),
            ('gripper_right', 'context', 'gripper_right'),
        ]
        if cfg.enable_bimanual_edges:
            edge_triples.extend([
                ('gripper_left', 'bimanual', 'gripper_right'),
                ('gripper_right', 'bimanual', 'gripper_left'),
            ])

        self.gnn = HeteroGraphTransformer(
            node_types=['gripper_left', 'gripper_right'],
            edge_triples=edge_triples,
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )

    def forward(self,
                all_gripper_feats_left: torch.Tensor,
                all_gripper_feats_right: torch.Tensor,
                edge_index: dict, edge_attr: dict,
                current_slice_left: slice,
                current_slice_right: slice,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        all_gripper_feats_{left,right}: ((N*L+1)*K, hidden_dim)
        current_slice_{left,right}: slice selecting the K current nodes

        Returns: (bottleneck_left, bottleneck_right) each (K, hidden_dim).
        """
        node_feats = {
            'gripper_left': all_gripper_feats_left,
            'gripper_right': all_gripper_feats_right,
        }
        out = self.gnn(node_feats, edge_index, edge_attr)
        return (out['gripper_left'][current_slice_left],
                out['gripper_right'][current_slice_right])


class BimanualPsiNetwork(nn.Module):
    """
    psi: processes the bimanual action graph.

    Predicts per-node denoising flow for action gripper keypoints of
    both arms.  Cross-arm sync edges enable coordinated denoising.

    Node types: 'gripper_left', 'gripper_right'
    Edge types: action (within-arm temporal), sync (cross-arm per step)
    """

    def __init__(self, cfg: BimanualIPConfig):
        super().__init__()
        self.cfg = cfg

        edge_triples = [
            ('gripper_left', 'action', 'gripper_left'),
            ('gripper_right', 'action', 'gripper_right'),
        ]
        if cfg.enable_sync_edges:
            edge_triples.extend([
                ('gripper_left', 'sync', 'gripper_right'),
                ('gripper_right', 'sync', 'gripper_left'),
            ])

        self.gnn = HeteroGraphTransformer(
            node_types=['gripper_left', 'gripper_right'],
            edge_triples=edge_triples,
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )

        # Denoising head: 7 = 3 (trans flow) + 3 (rot flow) + 1 (grip)
        self.denoising_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 7),
        )
        if not cfg.share_denoising_head:
            self.denoising_head_right = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, 7),
            )

    def forward(self,
                all_feats_left: torch.Tensor,
                all_feats_right: torch.Tensor,
                edge_index: dict, edge_attr: dict,
                action_slice_left: slice,
                action_slice_right: slice,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        all_feats_{left,right}: (K + T*K, hidden_dim) current + action
        action_slice_{left,right}: slice selecting the T*K action nodes

        Returns: (flow_left, flow_right) each (T*K, 7).
        """
        node_feats = {
            'gripper_left': all_feats_left,
            'gripper_right': all_feats_right,
        }
        out = self.gnn(node_feats, edge_index, edge_attr)
        left_action = out['gripper_left'][action_slice_left]
        right_action = out['gripper_right'][action_slice_right]

        flow_left = self.denoising_head(left_action)
        if self.cfg.share_denoising_head:
            flow_right = self.denoising_head(right_action)
        else:
            flow_right = self.denoising_head_right(right_action)

        return flow_left, flow_right

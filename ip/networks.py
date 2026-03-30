"""
The three sub-networks σ, φ, ψ (Eq. 6, Appendix C).

  ε_θ(G^k) = ψ( G( σ(G_l^a), φ( G_c( σ(G_l^t), {σ(G_l^{1:L})}^N_1 ) ) ) )

- σ: operates on local subgraphs, propagates scene info → gripper nodes.
- φ: operates on context graph, aggregates demo info → current gripper (bottleneck).
- ψ: operates on action graph, propagates bottleneck → action nodes → denoising MLP.

Each is a 2-layer heterogeneous graph transformer with hidden_dim=1024.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .config import IPConfig
from .graph_transformer import HeteroGraphTransformer


class SigmaNetwork(nn.Module):
    """
    σ: processes local subgraphs G_l.

    Input node types: 'scene' (M=16), 'gripper' (K=6)
    Edge types: ('scene', 'observe', 'gripper')

    Input features are projected to hidden_dim, then processed by 2-layer HGT.
    Output: updated gripper node features (scene features discarded downstream).
    """

    def __init__(self, cfg: IPConfig):
        super().__init__()
        self.cfg = cfg

        # Input projections
        self.scene_proj = nn.Linear(cfg.geo_feat_dim, cfg.hidden_dim)
        gripper_in_dim = cfg.gripper_feat_dim + cfg.gripper_state_embed_dim
        self.gripper_proj = nn.Linear(gripper_in_dim, cfg.hidden_dim)

        # Graph transformer
        self.gnn = HeteroGraphTransformer(
            node_types=['scene', 'gripper'],
            edge_triples=[('scene', 'observe', 'gripper')],
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )

    def forward(self, local_graph: dict) -> torch.Tensor:
        """
        local_graph: output of build_local_graph().
        Returns: gripper features (K, hidden_dim).
        """
        nf = local_graph['node_feats']
        scene_h = self.scene_proj(nf['scene'])
        gripper_h = self.gripper_proj(nf['gripper'])

        node_feats = {'scene': scene_h, 'gripper': gripper_h}
        out = self.gnn(node_feats,
                       local_graph['edge_index'],
                       local_graph['edge_attr'])
        return out['gripper']  # (K, hidden_dim)


class PhiNetwork(nn.Module):
    """
    φ: processes the context graph G_c.

    After σ has been applied to all local subgraphs, we have gripper features
    from: N demos × L waypoints + 1 current observation.
    φ links them via temporal (red) and context (grey) edges and aggregates
    information into the current gripper nodes → bottleneck.

    Node type: 'gripper' (all gripper nodes from demos + current)
    Edge types: ('gripper', 'temporal', 'gripper'),
                ('gripper', 'context', 'gripper')
    """

    def __init__(self, cfg: IPConfig):
        super().__init__()
        self.cfg = cfg
        self.gnn = HeteroGraphTransformer(
            node_types=['gripper'],
            edge_triples=[
                ('gripper', 'temporal', 'gripper'),
                ('gripper', 'context', 'gripper'),
            ],
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )

    def forward(self, all_gripper_feats: torch.Tensor,
                edge_index: dict, edge_attr: dict,
                current_node_slice: slice) -> torch.Tensor:
        """
        all_gripper_feats: (N*L*K + K, hidden_dim)  all gripper node features
        edge_index, edge_attr: context edges (temporal + context)
        current_node_slice: slice selecting the current gripper nodes

        Returns: current gripper features (K, hidden_dim) — the bottleneck.
        """
        node_feats = {'gripper': all_gripper_feats}
        out = self.gnn(node_feats, edge_index, edge_attr)
        return out['gripper'][current_node_slice]  # (K, hidden_dim)


class PsiNetwork(nn.Module):
    """
    ψ: processes the action graph.

    Takes the bottleneck (current gripper features from φ) and action gripper
    nodes (from noisy actions processed by σ), linked by action temporal edges.
    Predicts per-node denoising directions via a 2-layer MLP with GeLU.

    Node type: 'gripper' (current + T action steps)
    Edge types: ('gripper', 'action', 'gripper')

    Output: per action-gripper-node flow predictions ∈ R^7
            (3 translation flow + 3 rotation flow + 1 gripper action gradient)
    """

    def __init__(self, cfg: IPConfig):
        super().__init__()
        self.cfg = cfg
        self.gnn = HeteroGraphTransformer(
            node_types=['gripper'],
            edge_triples=[('gripper', 'action', 'gripper')],
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )
        # Denoising head: 2-layer MLP with GeLU (Appendix C)
        # Output: 7 per node (3 translation flow + 3 rotation flow + 1 grip)
        self.denoising_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, 7),
        )

    def forward(self, all_gripper_feats: torch.Tensor,
                edge_index: dict, edge_attr: dict,
                action_node_slice: slice) -> torch.Tensor:
        """
        all_gripper_feats: (K + T*K, hidden_dim) current + action gripper feats
        edge_index, edge_attr: action edges
        action_node_slice: slice selecting the action gripper nodes

        Returns: (T*K, 7) per-node flow predictions for action nodes.
        """
        node_feats = {'gripper': all_gripper_feats}
        out = self.gnn(node_feats, edge_index, edge_attr)
        action_feats = out['gripper'][action_node_slice]  # (T*K, hidden_dim)
        return self.denoising_head(action_feats)           # (T*K, 7)


class DiffusionStepEmbedding(nn.Module):
    """
    Sinusoidal embedding for the diffusion timestep k,
    added to gripper node features before ψ processes the action graph.
    """

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, k: int, dim: int = 1024,
                device='cpu', dtype=torch.float32) -> torch.Tensor:
        """Returns embedding vector (hidden_dim,)."""
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device, dtype=dtype) *
            (torch.log(torch.tensor(10000.0)) / half)
        )
        t = torch.tensor([k], device=device, dtype=dtype)
        emb = torch.cat([torch.sin(t * freqs), torch.cos(t * freqs)])
        return self.mlp(emb)

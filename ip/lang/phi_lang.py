"""
Language-conditioned context network φ_lang (Appendix J).

Replaces the standard φ (which aggregates demo info into gripper nodes)
with a language-conditioned version:

  Input: current observation local graph G_l^t + language embedding f_lang
  Output: bottleneck representation (same format as φ output)

Architecture:
  - Same as σ (2-layer heterogeneous graph transformer), but with an
    additional 'lang' node type.
  - f_lang is incorporated as an additional node in the graph.
  - Edges: scene→gripper (from σ), lang→gripper, lang→scene.
  - Trained with contrastive + MSE loss to match the bottleneck from φ.

At inference, φ_lang replaces the entire "σ(demos) + φ(context)" pipeline.
Only the current observation and a language description are needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from ..config import IPConfig
from ..graph_transformer import HeteroGraphTransformer
from ..pos_encoding import nerf_positional_encoding
from .encoder import LanguageEncoder


class PhiLang(nn.Module):
    """
    Language-conditioned bottleneck predictor.

    Graph structure:
      Node types: 'scene' (M=16), 'gripper' (K=6), 'lang' (1)
      Edge types:
        ('scene', 'observe', 'gripper')  — same as in σ
        ('lang', 'guide', 'gripper')     — language informs gripper
        ('lang', 'attend', 'scene')      — language attends to scene

    The lang node has no spatial position, so lang→gripper and lang→scene
    edges use a zero-padded positional encoding (position-agnostic attention).
    """

    def __init__(self, cfg: IPConfig = None):
        super().__init__()
        cfg = cfg or IPConfig()
        self.cfg = cfg

        # Language encoder (Sentence-BERT + projection)
        self.lang_encoder = LanguageEncoder(
            sbert_model_name=cfg.sbert_model,
            sbert_dim=cfg.sbert_dim,
            hidden_dim=cfg.hidden_dim,
        )

        # Input projections (same dims as σ)
        self.scene_proj = nn.Linear(cfg.geo_feat_dim, cfg.hidden_dim)
        gripper_in_dim = cfg.gripper_feat_dim + cfg.gripper_state_embed_dim
        self.gripper_proj = nn.Linear(gripper_in_dim, cfg.hidden_dim)

        # Graph transformer with 3 node types and 3 edge types
        self.gnn = HeteroGraphTransformer(
            node_types=['scene', 'gripper', 'lang'],
            edge_triples=[
                ('scene', 'observe', 'gripper'),
                ('lang', 'guide', 'gripper'),
                ('lang', 'attend', 'scene'),
            ],
            hidden_dim=cfg.hidden_dim,
            edge_dim=cfg.edge_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            num_layers=cfg.num_layers,
        )

        # Learnable "position" for the language node (since it has no
        # spatial position, we learn a position-like embedding for edges)
        self.lang_pos_embed = nn.Parameter(
            torch.randn(1, cfg.edge_dim) * 0.02
        )

    def _build_lang_edges(self, num_scene: int, num_gripper: int,
                           scene_pos: torch.Tensor,
                           gripper_pos: torch.Tensor,
                           device: torch.device) -> Tuple[Dict, Dict]:
        """
        Build edges from the language node to scene and gripper nodes.

        Since the language node has no spatial position, edge attributes
        use the learned lang_pos_embed (same for all edges from lang node).
        """
        edge_index = {}
        edge_attr = {}

        # lang → gripper: lang node (index 0) connects to all gripper nodes
        lang_to_grip_src = torch.zeros(num_gripper, dtype=torch.long, device=device)
        lang_to_grip_dst = torch.arange(num_gripper, device=device)
        edge_index[('lang', 'guide', 'gripper')] = torch.stack(
            [lang_to_grip_src, lang_to_grip_dst]
        )
        edge_attr[('lang', 'guide', 'gripper')] = self.lang_pos_embed.expand(
            num_gripper, -1
        )

        # lang → scene: lang node connects to all scene nodes
        lang_to_scene_src = torch.zeros(num_scene, dtype=torch.long, device=device)
        lang_to_scene_dst = torch.arange(num_scene, device=device)
        edge_index[('lang', 'attend', 'scene')] = torch.stack(
            [lang_to_scene_src, lang_to_scene_dst]
        )
        edge_attr[('lang', 'attend', 'scene')] = self.lang_pos_embed.expand(
            num_scene, -1
        )

        return edge_index, edge_attr

    def forward(self, scene_feat: torch.Tensor,
                scene_pos: torch.Tensor,
                gripper_feat: torch.Tensor,
                gripper_pos: torch.Tensor,
                lang_feat: torch.Tensor,
                scene_gripper_edge_index: torch.Tensor,
                scene_gripper_edge_attr: torch.Tensor,
                ) -> torch.Tensor:
        """
        Forward pass: produce bottleneck from current observation + language.

        scene_feat:   (M, geo_feat_dim)   from geometry encoder
        scene_pos:    (M, 3)              centroid positions
        gripper_feat: (K, gripper_feat_dim) from GripperNodeEncoder
        gripper_pos:  (K, 3)              keypoint positions (EE frame)
        lang_feat:    (1, hidden_dim)     from LanguageEncoder
        scene_gripper_edge_index: (2, M*K) scene→gripper edges
        scene_gripper_edge_attr:  (M*K, edge_dim)

        Returns: (K, hidden_dim) bottleneck — same format as φ output.
        """
        device = scene_feat.device
        M = scene_feat.shape[0]
        K = gripper_feat.shape[0]

        # Project inputs
        scene_h = self.scene_proj(scene_feat)      # (M, hidden_dim)
        gripper_h = self.gripper_proj(gripper_feat) # (K, hidden_dim)
        lang_h = lang_feat                          # (1, hidden_dim)

        # Build full edge set
        edge_index = {
            ('scene', 'observe', 'gripper'): scene_gripper_edge_index,
        }
        edge_attr = {
            ('scene', 'observe', 'gripper'): scene_gripper_edge_attr,
        }

        # Add language edges
        lang_ei, lang_ea = self._build_lang_edges(M, K, scene_pos, gripper_pos, device)
        edge_index.update(lang_ei)
        edge_attr.update(lang_ea)

        # Run graph transformer
        node_feats = {
            'scene': scene_h,
            'gripper': gripper_h,
            'lang': lang_h,
        }
        out = self.gnn(node_feats, edge_index, edge_attr)

        return out['gripper']  # (K, hidden_dim) — the bottleneck


# ──────────────────────────────────────────────────────────────────────
# Contrastive + MSE loss for training φ_lang
# ──────────────────────────────────────────────────────────────────────

class BottleneckAlignmentLoss(nn.Module):
    """
    Combined contrastive (InfoNCE) + MSE loss (Appendix J).

    Contrastive: align (lang_bottleneck, demo_bottleneck) pairs.
    MSE: directly minimise distance between paired bottlenecks.
    """

    def __init__(self, temperature: float = 0.07, lambda_mse: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.lambda_mse = lambda_mse

    def forward(self, lang_bottleneck: torch.Tensor,
                demo_bottleneck: torch.Tensor) -> torch.Tensor:
        """
        lang_bottleneck: (B, K, hidden_dim) from φ_lang
        demo_bottleneck: (B, K, hidden_dim) from frozen φ (ground truth)

        Returns: scalar loss.
        """
        B, K, D = lang_bottleneck.shape

        # Pool over keypoints for contrastive: (B, hidden_dim)
        lang_pooled = lang_bottleneck.mean(dim=1)
        demo_pooled = demo_bottleneck.mean(dim=1)

        # L2 normalise for cosine similarity
        lang_norm = F.normalize(lang_pooled, dim=-1)
        demo_norm = F.normalize(demo_pooled, dim=-1)

        # InfoNCE loss (symmetric)
        logits = lang_norm @ demo_norm.T / self.temperature  # (B, B)
        labels = torch.arange(B, device=logits.device)
        loss_l2d = F.cross_entropy(logits, labels)
        loss_d2l = F.cross_entropy(logits.T, labels)
        contrastive_loss = (loss_l2d + loss_d2l) / 2

        # MSE loss (per-node)
        mse_loss = F.mse_loss(lang_bottleneck, demo_bottleneck)

        return contrastive_loss + self.lambda_mse * mse_loss

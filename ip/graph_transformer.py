"""
Heterogeneous Graph Transformer layer (Eq. 3, Appendix C).

Implements the attention mechanism from Shi et al. (2020) with edge features:

    F_i' = W_1 F_i + Σ_{j∈N(i)} att_{i,j} · (W_2 F_j + W_5 e_{ij})

    att_{i,j} = softmax( (W_3 F_i)^T (W_4 F_j + W_5 e_{ij}) / √d )

Each (src_type, edge_type, dst_type) triple has its own W matrices.
The network uses 16 heads with 64 dims each (hidden_dim=1024).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple

# Type aliases for readability
NodeType = str
EdgeTriple = Tuple[str, str, str]  # (src_type, edge_type, dst_type)


class HeteroAttentionLayer(nn.Module):
    """
    One layer of the heterogeneous graph transformer for a single edge type.
    Computes attention and message passing from src nodes to dst nodes.
    """

    def __init__(self, hidden_dim: int, edge_dim: int,
                 num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)

        # W_3: query projection (operates on dst / target node)
        self.W_q = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        # W_4: key projection (operates on src / neighbour node)
        self.W_k = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        # W_5: edge feature projection (used in both key and value)
        self.W_e = nn.Linear(edge_dim, num_heads * head_dim, bias=False)
        # W_2: value projection (operates on src / neighbour node)
        self.W_v = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor,
                edge_attr: torch.Tensor,
                src_idx: torch.Tensor, dst_idx: torch.Tensor,
                num_dst: int) -> torch.Tensor:
        """
        q_feat:    (num_dst_nodes, hidden_dim)    destination node features
        kv_feat:   (num_src_nodes, hidden_dim)    source node features
        edge_attr: (num_edges, edge_dim)          edge attributes
        src_idx:   (num_edges,)                   source node indices
        dst_idx:   (num_edges,)                   destination node indices
        num_dst:   number of destination nodes (for scatter)

        Returns:   (num_dst_nodes, num_heads * head_dim)  aggregated messages
        """
        H, D = self.num_heads, self.head_dim
        E = src_idx.shape[0]

        # Project features  → (N, H, D)
        Q = self.W_q(q_feat).view(-1, H, D)      # dst nodes
        K = self.W_k(kv_feat).view(-1, H, D)     # src nodes
        V = self.W_v(kv_feat).view(-1, H, D)     # src nodes
        E_proj = self.W_e(edge_attr).view(E, H, D)  # edges

        # Gather per-edge queries and keys
        q_e = Q[dst_idx]                           # (E, H, D)
        k_e = K[src_idx] + E_proj                  # (E, H, D)  key + edge
        v_e = V[src_idx] + E_proj                  # (E, H, D)  value + edge

        # Attention scores
        attn_logits = (q_e * k_e).sum(dim=-1) / self.scale  # (E, H)

        # Softmax per destination node (scatter softmax)
        attn = self._scatter_softmax(attn_logits, dst_idx, num_dst)  # (E, H)

        # Weighted aggregation
        messages = attn.unsqueeze(-1) * v_e        # (E, H, D)
        # Scatter-add to destination nodes
        out = torch.zeros(num_dst, H, D, device=q_feat.device, dtype=q_feat.dtype)
        dst_exp = dst_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, H, D)
        out.scatter_add_(0, dst_exp, messages)
        return out.reshape(num_dst, H * D)

    @staticmethod
    def _scatter_softmax(logits: torch.Tensor, idx: torch.Tensor,
                         num_nodes: int) -> torch.Tensor:
        """Softmax over groups defined by idx. logits: (E, H), idx: (E,)."""
        # Numerical stability: subtract max per group
        max_vals = torch.zeros(num_nodes, logits.shape[1],
                               device=logits.device, dtype=logits.dtype)
        max_vals.scatter_reduce_(
            0, idx.unsqueeze(-1).expand_as(logits), logits, reduce='amax',
            include_self=False)
        logits = logits - max_vals[idx]
        exp_logits = logits.exp()
        # Sum per group
        sum_exp = torch.zeros(num_nodes, logits.shape[1],
                              device=logits.device, dtype=logits.dtype)
        sum_exp.scatter_add_(0, idx.unsqueeze(-1).expand_as(exp_logits), exp_logits)
        return exp_logits / (sum_exp[idx] + 1e-8)


class HeteroGraphTransformerLayer(nn.Module):
    """
    Full heterogeneous graph transformer layer.
    Handles multiple node types and edge types.

    For each edge type, an independent attention sub-layer aggregates
    messages.  The messages from all edge types arriving at the same
    destination node type are summed (Appendix C: "aggregated via summation").

    Then:  F_i' = LayerNorm(W_1 F_i + aggregated_msg) + residual
    Followed by a feed-forward block with another residual + LayerNorm.
    """

    def __init__(self, node_types: List[NodeType],
                 edge_triples: List[EdgeTriple],
                 hidden_dim: int, edge_dim: int,
                 num_heads: int = 16, head_dim: int = 64,
                 ff_mult: int = 2):
        super().__init__()
        self.node_types = node_types
        self.edge_triples = edge_triples
        self.hidden_dim = hidden_dim

        # W_1 (skip / self projection) per destination node type
        self.skip_proj = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, hidden_dim, bias=False)
            for nt in node_types
        })

        # Attention sub-layer per edge triple
        self.attn_layers = nn.ModuleDict()
        for src_t, et, dst_t in edge_triples:
            key = f"{src_t}__{et}__{dst_t}"
            self.attn_layers[key] = HeteroAttentionLayer(
                hidden_dim, edge_dim, num_heads, head_dim
            )

        # Layer norms (post-attention, post-FFN) per node type
        self.norm1 = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
        self.norm2 = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})

        # Feed-forward per node type
        self.ffn = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * ff_mult),
                nn.GELU(),
                nn.Linear(hidden_dim * ff_mult, hidden_dim),
            )
            for nt in node_types
        })

    def forward(self, node_feats: Dict[NodeType, torch.Tensor],
                edge_index: Dict[EdgeTriple, torch.Tensor],
                edge_attr: Dict[EdgeTriple, torch.Tensor],
                ) -> Dict[NodeType, torch.Tensor]:
        """
        node_feats: {node_type: (N_type, hidden_dim)}
        edge_index: {(src, rel, dst): (2, E)}   row 0 = src, row 1 = dst
        edge_attr:  {(src, rel, dst): (E, edge_dim)}
        Returns: updated node_feats (same structure).
        """
        # Collect aggregated messages per destination node type
        agg_msg: Dict[NodeType, torch.Tensor] = {}

        for triple in self.edge_triples:
            if triple not in edge_index:
                continue
            src_t, et, dst_t = triple
            key = f"{src_t}__{et}__{dst_t}"
            ei = edge_index[triple]        # (2, E)
            ea = edge_attr[triple]         # (E, edge_dim)
            src_idx, dst_idx = ei[0], ei[1]
            num_dst = node_feats[dst_t].shape[0]

            msg = self.attn_layers[key](
                q_feat=node_feats[dst_t],
                kv_feat=node_feats[src_t],
                edge_attr=ea,
                src_idx=src_idx,
                dst_idx=dst_idx,
                num_dst=num_dst,
            )  # (num_dst, hidden_dim)

            if dst_t not in agg_msg:
                agg_msg[dst_t] = msg
            else:
                agg_msg[dst_t] = agg_msg[dst_t] + msg

        # Update each node type
        out = {}
        for nt in self.node_types:
            h = node_feats[nt]
            skip = self.skip_proj[nt](h)
            msg = agg_msg.get(nt, torch.zeros_like(h))
            # Attention residual + LayerNorm
            h = self.norm1[nt](h + skip + msg)
            # FFN residual + LayerNorm
            h = self.norm2[nt](h + self.ffn[nt](h))
            out[nt] = h
        return out


class HeteroGraphTransformer(nn.Module):
    """
    Stack of HeteroGraphTransformerLayer.
    Default: 2 layers as per Appendix C.
    """

    def __init__(self, node_types: List[NodeType],
                 edge_triples: List[EdgeTriple],
                 hidden_dim: int = 1024, edge_dim: int = 60,
                 num_heads: int = 16, head_dim: int = 64,
                 num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            HeteroGraphTransformerLayer(
                node_types, edge_triples,
                hidden_dim, edge_dim, num_heads, head_dim,
            )
            for _ in range(num_layers)
        ])

    def forward(self, node_feats, edge_index, edge_attr):
        for layer in self.layers:
            node_feats = layer(node_feats, edge_index, edge_attr)
        return node_feats

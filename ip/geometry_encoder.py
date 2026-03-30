"""
Geometry encoder φ_e (Appendix A).

Pre-trained as part of an occupancy network:
  - Encoder φ_e: 2 Set Abstraction layers (PointNet++) with NeRF-like
    positional encoding.  Input 2048 points → M=16 centroids with 512-dim features.
  - Decoder ψ_e: 8-layer MLP predicting occupancy given (centroid feature, query point).

During Instant Policy training, only φ_e is used (frozen).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .pos_encoding import nerf_positional_encoding


# ──────────────────────────────────────────────────────────────────────
# Farthest Point Sampling (pure PyTorch)
# ──────────────────────────────────────────────────────────────────────

def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (B, N, 3)
    Returns indices: (B, npoint)
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    # start from a random point
    farthest = torch.randint(0, N, (B,), device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)  # (B,1,3)
        dist = ((xyz - centroid_xyz) ** 2).sum(-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(dim=-1)  # (B,)
    return centroids


def knn_query(xyz: torch.Tensor, new_xyz: torch.Tensor,
              k: int) -> torch.Tensor:
    """
    For each point in new_xyz, find k nearest neighbours in xyz.
    xyz:     (B, N, 3)
    new_xyz: (B, S, 3)
    Returns: (B, S, k) indices into xyz.
    """
    # (B, S, N)
    dist = torch.cdist(new_xyz, xyz)
    _, idx = dist.topk(k, dim=-1, largest=False)
    return idx


# ──────────────────────────────────────────────────────────────────────
# Set Abstraction layer (PointNet++ with NeRF positional encoding)
# ──────────────────────────────────────────────────────────────────────

class SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction with NeRF-like position encoding
    for relative coordinates (Appendix A).
    """

    def __init__(self, npoint: int, nsample: int, in_channels: int,
                 mlp_channels: List[int], freq_bands: int = 10):
        super().__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.freq_bands = freq_bands
        # input to MLP: per-neighbour feature =
        #   in_channels (prev feature, 0 for first layer)
        #   + 3 * 2 * freq_bands (NeRF encoded relative pos)
        pos_enc_dim = 3 * 2 * freq_bands
        dims = [in_channels + pos_enc_dim] + mlp_channels
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor | None = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xyz:      (B, N, 3)
        features: (B, N, C) or None
        Returns:
            new_xyz:      (B, npoint, 3)
            new_features: (B, npoint, mlp[-1])
        """
        B, N, _ = xyz.shape

        # 1. Farthest Point Sampling → centroids
        fps_idx = farthest_point_sampling(xyz, self.npoint)   # (B, npoint)
        new_xyz = torch.gather(
            xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3)
        )  # (B, npoint, 3)

        # 2. KNN grouping
        knn_idx = knn_query(xyz, new_xyz, self.nsample)       # (B, npoint, nsample)

        # Gather neighbour positions
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, self.npoint, -1, -1),
            2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (B, npoint, nsample, 3)

        # Relative positions → NeRF encoding
        rel_pos = grouped_xyz - new_xyz.unsqueeze(2)          # (B, npoint, nsample, 3)
        rel_enc = nerf_positional_encoding(rel_pos, self.freq_bands)
        # (B, npoint, nsample, pos_enc_dim)

        # 3. Build per-point features
        if features is not None:
            grouped_feat = torch.gather(
                features.unsqueeze(1).expand(-1, self.npoint, -1, -1),
                2,
                knn_idx.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])
            )  # (B, npoint, nsample, C)
            point_feat = torch.cat([grouped_feat, rel_enc], dim=-1)
        else:
            point_feat = rel_enc

        # 4. PointNet: MLP → max pool
        point_feat = self.mlp(point_feat)                     # (B, npoint, nsample, out)
        new_features = point_feat.max(dim=2).values           # (B, npoint, out)
        return new_xyz, new_features


# ──────────────────────────────────────────────────────────────────────
# Geometry encoder φ_e
# ──────────────────────────────────────────────────────────────────────

class GeometryEncoder(nn.Module):
    """
    Two Set Abstraction layers:  2048 → 128 → 16 points.
    Output: M=16 centroids with 512-dim features.
    """

    def __init__(self, cfg=None):
        super().__init__()
        from .config import IPConfig
        cfg = cfg or IPConfig()

        self.sa1 = SetAbstraction(
            npoint=cfg.sa1_npoint,
            nsample=cfg.sa1_nsample,
            in_channels=0,                     # first layer: positions only
            mlp_channels=cfg.sa1_mlp,
            freq_bands=cfg.geo_freq_bands,
        )
        self.sa2 = SetAbstraction(
            npoint=cfg.sa2_npoint,             # == 16
            nsample=cfg.sa2_nsample,
            in_channels=cfg.sa1_mlp[-1],       # 256
            mlp_channels=cfg.sa2_mlp,
            freq_bands=cfg.geo_freq_bands,
        )

    def forward(self, pcd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        pcd: (B, 2048, 3) point cloud in EE frame.
        Returns:
            centroids: (B, M, 3)   centroid positions
            features:  (B, M, 512) local geometry features
        """
        xyz1, feat1 = self.sa1(pcd)            # (B, 128, 3), (B, 128, 256)
        xyz2, feat2 = self.sa2(xyz1, feat1)    # (B, 16, 3),  (B, 16, 512)
        return xyz2, feat2


# ──────────────────────────────────────────────────────────────────────
# Occupancy decoder ψ_e  (for pre-training only)
# ──────────────────────────────────────────────────────────────────────

class OccupancyDecoder(nn.Module):
    """
    8-layer MLP with residual connections (Appendix A).
    Given a centroid feature and a query point, predicts occupancy [0, 1].
    Query positions use the same NeRF-like encoding.
    """

    def __init__(self, feat_dim: int = 512, hidden_dim: int = 256,
                 num_layers: int = 8, freq_bands: int = 10):
        super().__init__()
        self.freq_bands = freq_bands
        query_enc_dim = 3 * 2 * freq_bands     # 60
        in_dim = feat_dim + query_enc_dim       # 572
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, centroid_feat: torch.Tensor,
                query_pos: torch.Tensor,
                centroid_pos: torch.Tensor) -> torch.Tensor:
        """
        centroid_feat: (B, C)  feature of the nearest centroid
        query_pos:     (B, 3)  query point in world coords
        centroid_pos:  (B, 3)  centroid position
        Returns: (B, 1) occupancy logit
        """
        rel = query_pos - centroid_pos
        q_enc = nerf_positional_encoding(rel, self.freq_bands)
        x = torch.cat([centroid_feat, q_enc], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


class OccupancyNetwork(nn.Module):
    """
    Full occupancy network = GeometryEncoder + OccupancyDecoder.
    Used for pre-training the geometry encoder on ShapeNet.
    """

    def __init__(self, cfg=None):
        super().__init__()
        from .config import IPConfig
        cfg = cfg or IPConfig()
        self.encoder = GeometryEncoder(cfg)
        self.decoder = OccupancyDecoder(
            feat_dim=cfg.geo_feat_dim,
            hidden_dim=cfg.occ_decoder_dim,
            num_layers=cfg.occ_decoder_layers,
            freq_bands=cfg.geo_freq_bands,
        )

    def forward(self, pcd: torch.Tensor,
                query_points: torch.Tensor) -> torch.Tensor:
        """
        pcd:          (B, 2048, 3) object surface point cloud
        query_points: (B, Q, 3)   query points
        Returns:      (B, Q, 1)   occupancy logits
        """
        centroids, features = self.encoder(pcd)  # (B,M,3), (B,M,512)
        # For each query, find nearest centroid
        dist = torch.cdist(query_points, centroids)  # (B, Q, M)
        nearest_idx = dist.argmin(dim=-1)             # (B, Q)
        # Gather nearest centroid features and positions
        B, Q = query_points.shape[:2]
        nearest_feat = torch.gather(
            features.unsqueeze(1).expand(-1, Q, -1, -1),
            2, nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, features.shape[-1])
        ).squeeze(2)  # (B, Q, 512)
        nearest_pos = torch.gather(
            centroids.unsqueeze(1).expand(-1, Q, -1, -1),
            2, nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
        ).squeeze(2)  # (B, Q, 3)
        # Flatten for decoder
        logits = self.decoder(
            nearest_feat.reshape(B * Q, -1),
            query_points.reshape(B * Q, 3),
            nearest_pos.reshape(B * Q, 3),
        )
        return logits.reshape(B, Q, 1)

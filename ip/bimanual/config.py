"""
Bimanual Instant Policy configuration.

Extends IPConfig with dual-arm specific parameters while keeping
all single-arm defaults unchanged.
"""
from dataclasses import dataclass, field
from typing import List

from ..config import IPConfig


@dataclass
class BimanualIPConfig(IPConfig):
    """
    Configuration for the bimanual (dual-arm) Instant Policy.

    Inherits all single-arm hyperparameters and adds cross-arm
    coordination flags and bimanual-specific settings.
    """

    # ── Cross-arm coordination edges ──────────────────────────────
    # These flags control whether cross-arm edges are added at each
    # level of the graph hierarchy.  Disabling them degrades the model
    # to two independent single-arm policies that share a scene encoder.
    enable_coordinate_edges: bool = True   # cross-arm edges in sigma (local)
    enable_bimanual_edges: bool = True     # cross-arm edges in phi   (context)
    enable_sync_edges: bool = True         # cross-arm edges in psi   (action)

    # ── Gripper sharing ───────────────────────────────────────────
    # When True, left and right arms share gripper encoder weights
    # and denoising head weights.  Appropriate when both arms have
    # identical gripper geometry.
    share_gripper_encoder: bool = True
    share_denoising_head: bool = True

    # ── Scene encoding frame ──────────────────────────────────────
    # 'midpoint' : PCD centred at the midpoint of both EE positions
    # 'world'    : PCD kept in world frame (no recentring)
    scene_frame: str = "midpoint"

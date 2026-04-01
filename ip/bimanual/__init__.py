"""
Bimanual Instant Policy: dual-arm extension.

Extends the single-arm Instant Policy with:
  - Cross-arm coordination edges in the heterogeneous graph (sigma, phi, psi)
  - Bimanual pseudo-demonstration generation
  - PerAct2 dataset loading for bimanual RLBench tasks
  - Independent SE(3) diffusion per arm with synchronized denoising

Usage:
  from ip.bimanual import BimanualIPConfig, BimanualGraphDiffusionPolicy
"""
from .config import BimanualIPConfig
from .model import BimanualGraphDiffusionPolicy

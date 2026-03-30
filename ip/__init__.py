"""
Instant Policy: In-Context Imitation Learning via Graph Diffusion.

Full reimplementation with language-conditioned modality transfer.

Modules:
  - config:            Centralised hyperparameters
  - se3_utils:         SE(3) math (logmap, expmap, SVD)
  - pos_encoding:      NeRF-like positional encoding
  - geometry_encoder:  PointNet++ Set Abstraction encoder (φ_e)
  - graph_builder:     Heterogeneous graph construction
  - graph_transformer: Heterogeneous graph transformer (Eq. 3)
  - networks:          σ, φ, ψ sub-networks (Eq. 6)
  - diffusion:         SE(3) diffusion process (DDPM/DDIM)
  - model:             Full GraphDiffusionPolicy
  - pseudo_demo:       Pseudo-demonstration generator
  - dataset:           Training datasets
  - train:             Training pipeline

  - lang/              Language transfer module (Appendix J)
    - encoder:         Sentence-BERT + projection
    - phi_lang:        Language-conditioned context network
    - lang_dataset:    Language-annotated dataset
    - train_lang:      Language transfer training
  - deploy_lang:       Language-guided deployment
"""
from .config import IPConfig
from .model import GraphDiffusionPolicy

"""
NeRF-like positional encoding used throughout Instant Policy.

Encodes x ∈ R^D as:
  [sin(2^0 π x), cos(2^0 π x), ..., sin(2^{L-1} π x), cos(2^{L-1} π x)]

Reference: Section 3.2 (edge encoding) and Appendix A (geometry encoder).
"""
import torch
import math


def nerf_positional_encoding(x: torch.Tensor, num_bands: int) -> torch.Tensor:
    """
    x: (..., D)  arbitrary last-dim vector (typically D=3 for 3D positions).
    num_bands: number of frequency bands (paper uses 10, i.e. 2^0 .. 2^9).
    Returns: (..., D * 2 * num_bands)
    """
    freqs = (2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype))  # (num_bands,)
    # x_expanded: (..., D, 1) * (1, num_bands) -> (..., D, num_bands)
    x_expanded = x.unsqueeze(-1) * freqs * math.pi
    sin_enc = torch.sin(x_expanded)
    cos_enc = torch.cos(x_expanded)
    # interleave sin, cos -> (..., D, 2*num_bands)
    enc = torch.stack([sin_enc, cos_enc], dim=-1).reshape(
        *x.shape[:-1], x.shape[-1] * 2 * num_bands
    )
    return enc

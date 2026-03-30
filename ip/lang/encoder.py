"""
Language encoder for modality transfer (Appendix J).

Uses Sentence-BERT (Reimers, 2019) to encode task descriptions into
a fixed-size embedding f_lang, then projects to the graph hidden dimension.

The Sentence-BERT model is frozen; only the projection MLP is trained.
"""
import torch
import torch.nn as nn
from typing import List


class LanguageEncoder(nn.Module):
    """
    Frozen Sentence-BERT + trainable projection MLP.

    Sentence-BERT maps text → f_lang ∈ R^{sbert_dim}.
    Projection MLP maps f_lang → R^{hidden_dim} for use as a graph node.
    """

    def __init__(self, sbert_model_name: str = 'all-MiniLM-L6-v2',
                 sbert_dim: int = 384,
                 hidden_dim: int = 1024):
        super().__init__()
        self.sbert_dim = sbert_dim
        self.hidden_dim = hidden_dim

        # Load Sentence-BERT (frozen)
        try:
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer(sbert_model_name)
            # Freeze all parameters
            for param in self.sbert.parameters():
                param.requires_grad = False
            self._sbert_available = True
        except ImportError:
            print("Warning: sentence-transformers not installed. "
                  "Install with: pip install sentence-transformers")
            self._sbert_available = False
            self.sbert = None

        # Trainable projection: sbert_dim → hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(sbert_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def encode_text(self, texts: List[str], device='cpu') -> torch.Tensor:
        """
        Encode a list of text strings using Sentence-BERT.
        Returns: (B, sbert_dim) tensor.
        """
        if not self._sbert_available:
            B = len(texts)
            return torch.randn(B, self.sbert_dim, device=device)

        with torch.no_grad():
            embeddings = self.sbert.encode(
                texts, convert_to_tensor=True,
                show_progress_bar=False,
            )
        return embeddings.to(device)

    def forward(self, texts: List[str] = None,
                text_embeddings: torch.Tensor = None,
                device='cpu') -> torch.Tensor:
        """
        Encode text and project to graph dimension.

        Either provide `texts` (list of strings) or pre-computed
        `text_embeddings` (B, sbert_dim).

        Returns: (B, hidden_dim) projected language features.
        """
        if text_embeddings is None:
            assert texts is not None, "Provide either texts or text_embeddings"
            text_embeddings = self.encode_text(texts, device)

        return self.projection(text_embeddings.to(device))

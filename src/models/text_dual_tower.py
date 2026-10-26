"""Frozen text dual tower with a trainable match head.

User and job vectors are sentence embeddings of the same dimension. The head
scores a pair from the concatenation ``[u, j, u*j, |u-j|]``.
"""

from __future__ import annotations

import torch
from torch import nn


class TextMatchHead(nn.Module):
    """MLP over ``[u, j, u*j, |u-j|]``."""

    def __init__(self, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, user_emb, job_emb):
        features = torch_cat(user_emb, job_emb)
        return self.mlp(features).squeeze(-1)


def torch_cat(user_emb, job_emb):
    return torch.cat([user_emb, job_emb, user_emb * job_emb, (user_emb - job_emb).abs()], dim=-1)

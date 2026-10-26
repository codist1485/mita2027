"""Non-learned MiniLM similarity references.

Cached embeddings in this project are L2-normalized
(``normalize_embeddings=True`` during encode). Under that condition,

    cosine(u, j) == dot(u, j)

for every pair. Cosine and Dot Product are therefore mathematically equivalent
reference scorers, not two independent algorithms.
"""

from __future__ import annotations

import torch
from torch import nn


class CosineSimilarityScorer(nn.Module):
    def forward(self, user_emb: torch.Tensor, job_emb: torch.Tensor) -> torch.Tensor:
        user = nn.functional.normalize(user_emb, dim=-1)
        job = nn.functional.normalize(job_emb, dim=-1)
        return (user * job).sum(dim=-1)


class DotProductScorer(nn.Module):
    def forward(self, user_emb: torch.Tensor, job_emb: torch.Tensor) -> torch.Tensor:
        return (user_emb * job_emb).sum(dim=-1)


EQUIVALENCE_NOTE = (
    "With L2-normalized MiniLM caches, CosineSimilarityScorer and DotProductScorer "
    "produce identical scores up to floating-point noise."
)

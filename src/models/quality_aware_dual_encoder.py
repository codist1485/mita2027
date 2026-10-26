"""Quality-aware dual encoder on the frozen text dual-tower backbone.

The match head is identical to ``TextMatchHead``: frozen MiniLM vectors go
through ``concat([u, j, u*j, |u-j|]) -> 128 -> 64 -> 1``. Two losses train
that head.

Quality-weighted BCE
    weight = QUALITY_WEIGHT_BIAS + quality_score
    default bias is 0.5, so weights lie in about [0.5, 1.5] when quality is in [0, 1].
    The batch loss is the weighted mean of per-pair BCE-with-logits, so a
    constant weight does not change the loss scale.

RJDB contrastive loss
    For each (job, resume_matched, resume_unmatched) triple the same head
    scores the positive and negative resume. The auxiliary term is a margin
    triplet on those logits:

        relu(margin + score(job, unmatched) - score(job, matched))

    The text encoder stays frozen, so the triplet gradient updates only the
    match head. ``CONTRASTIVE_MARGIN`` and the training-script ``LAMBDA`` are
    the ablation knobs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.text_dual_tower import TextMatchHead

QUALITY_WEIGHT_BIAS = 0.5
CONTRASTIVE_MARGIN = 1.0
WEIGHT_STRATEGIES = ("baseline", "mild", "reverse", "piecewise")
WEIGHT_FORMULAS = {
    "baseline": "0.5 + quality_score",
    "mild": "0.7 + 0.3 * quality_score",
    "reverse": "1.5 - quality_score",
    "piecewise": "1.0 if quality>=0.7; 0.8 if 0.5<=quality<0.7; 0.6 if quality<0.5",
}


class QualityAwareDualEncoder(nn.Module):
    """Trainable match head over frozen user and job embeddings."""

    def __init__(self, embedding_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head = TextMatchHead(embedding_dim, dropout=dropout)

    def forward(self, user_emb: torch.Tensor, job_emb: torch.Tensor) -> torch.Tensor:
        return self.head(user_emb, job_emb)


def reverse_weight_kappa(quality_score: torch.Tensor, kappa: float) -> torch.Tensor:
    """General reverse-weight family: clip(1 + kappa*(0.5 - q), 0.1, 3.0).

    kappa=1 recovers 1.5 - q on q in [0, 1] (clip inactive).
    """
    w = 1.0 + float(kappa) * (0.5 - quality_score)
    return w.clamp(0.1, 3.0)


def assert_kappa_one_matches_legacy_reverse() -> None:
    """DR self-check: κ=1 must equal 1.5 − q elementwise."""
    q = torch.linspace(0.0, 1.0, 1001)
    got = reverse_weight_kappa(q, kappa=1.0)
    if not torch.allclose(got, 1.5 - q, rtol=0.0, atol=1e-6):
        raise AssertionError("kappa=1 reverse weight must equal 1.5 - q elementwise")


def quality_sample_weight(
    quality_score: torch.Tensor,
    bias: float = QUALITY_WEIGHT_BIAS,
    strategy: str = "baseline",
    kappa: float | None = None,
) -> torch.Tensor:
    """Map a job quality score to a per-pair BCE weight.

    ``bias`` is kept for the original baseline call. Named strategies ignore it
    and use the formulas in ``WEIGHT_FORMULAS``. When ``strategy=="reverse"`` and
    ``kappa`` is set, use ``reverse_weight_kappa`` (kappa=1 ≡ 1.5 - q).
    """
    if strategy == "baseline":
        return bias + quality_score
    if strategy == "mild":
        return 0.7 + 0.3 * quality_score
    if strategy == "reverse":
        if kappa is None:
            return 1.5 - quality_score
        return reverse_weight_kappa(quality_score, kappa)
    if strategy == "piecewise":
        weight = torch.full_like(quality_score, 0.6)
        weight = torch.where(quality_score >= 0.5, torch.full_like(quality_score, 0.8), weight)
        weight = torch.where(quality_score >= 0.7, torch.full_like(quality_score, 1.0), weight)
        return weight
    raise ValueError(f"unknown weight strategy: {strategy}")


def quality_weighted_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    quality_score: torch.Tensor,
    pos_weight: torch.Tensor,
    bias: float = QUALITY_WEIGHT_BIAS,
    strategy: str = "baseline",
    kappa: float | None = None,
) -> torch.Tensor:
    """BCE-with-logits averaged with the selected quality weight."""
    weights = quality_sample_weight(quality_score, bias=bias, strategy=strategy, kappa=kappa)
    per_pair = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    return (per_pair * weights).sum() / weights.sum().clamp_min(1e-8)


def supervised_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    quality_score: torch.Tensor,
    pos_weight: torch.Tensor,
    use_quality_weight: bool,
    bias: float = QUALITY_WEIGHT_BIAS,
    strategy: str = "baseline",
) -> torch.Tensor:
    """Quality-weighted BCE, or plain mean BCE-with-logits when the switch is off."""
    if use_quality_weight:
        return quality_weighted_bce(logits, labels, quality_score, pos_weight, bias=bias, strategy=strategy)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def triplet_contrastive_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    margin: float = CONTRASTIVE_MARGIN,
) -> torch.Tensor:
    """Margin triplet: the matched resume must outscore the unmatched one."""
    return F.relu(margin + negative_logits - positive_logits).mean()

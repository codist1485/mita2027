"""Phase 1 losses.

Important: ``quality_weighted_bce`` already uses

    sum(weight * bce) / sum(weight)

so multiplying all weights by a constant does not change the loss. Phase 1
therefore does **not** add a second mean-normalization layer on top of that
form. Weight strategies only change relative sample importance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.quality_aware_dual_encoder import (
    CONTRASTIVE_MARGIN,
    quality_sample_weight,
    quality_weighted_bce,
    triplet_contrastive_loss,
)

FOCAL_GAMMA = 2.0


def assert_binary_labels(labels: torch.Tensor) -> None:
    unique = torch.unique(labels)
    bad = unique[(unique != 0) & (unique != 1)]
    if bad.numel():
        raise ValueError(f"BCE/Focal labels must be in {{0,1}}, found {bad.tolist()}")


def uniform_bce(logits: torch.Tensor, labels: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    assert_binary_labels(labels)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def weighted_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    quality_score: torch.Tensor,
    pos_weight: torch.Tensor,
    strategy: str,
    kappa: float | None = None,
) -> torch.Tensor:
    """Reuse historical weighted-mean BCE; no extra weight renormalization."""
    assert_binary_labels(labels)
    return quality_weighted_bce(
        logits, labels, quality_score, pos_weight, strategy=strategy, kappa=kappa
    )


def focal_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    """Binary focal loss on logits.

    Definition:
        p = sigmoid(logits)
        p_t = p if y=1 else 1-p
        bce = BCEWithLogits(logits, y; pos_weight)
        FL = (1 - p_t)^gamma * bce

    ``pos_weight`` is applied inside the base BCE term exactly as in A0.
    """
    assert_binary_labels(labels)
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1.0 - probs) * (1.0 - labels)
    return ((1.0 - p_t).clamp(min=0.0, max=1.0).pow(gamma) * bce).mean()


def supervised_loss(
    name: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    quality_score: torch.Tensor,
    pos_weight: torch.Tensor,
    reverse_kappa: float | None = None,
) -> torch.Tensor:
    if name == "uniform_bce":
        return uniform_bce(logits, labels, pos_weight)
    if name == "forward_quality_bce":
        return weighted_bce(logits, labels, quality_score, pos_weight, strategy="baseline")
    if name == "reverse_quality_bce":
        return weighted_bce(
            logits, labels, quality_score, pos_weight, strategy="reverse", kappa=reverse_kappa
        )
    if name == "focal_bce":
        return focal_bce(logits, labels, pos_weight)
    raise ValueError(f"unknown supervised loss {name}")


def contrastive_term(positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
    return triplet_contrastive_loss(positive_logits, negative_logits, margin=CONTRASTIVE_MARGIN)


def describe_weight_normalization() -> str:
    return (
        "quality_weighted_bce uses sum(w*bce)/sum(w); weights are scale-invariant. "
        "Phase 1 does not stack a second mean-normalization."
    )


__all__ = [
    "assert_binary_labels",
    "uniform_bce",
    "weighted_bce",
    "focal_bce",
    "supervised_loss",
    "contrastive_term",
    "quality_sample_weight",
    "describe_weight_normalization",
    "FOCAL_GAMMA",
    "CONTRASTIVE_MARGIN",
]

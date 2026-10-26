"""Phase 1 metrics with explicit label semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.eval.evaluate_ranking import ranking_metrics


def assert_binary_numpy(labels: np.ndarray) -> None:
    labels = np.asarray(labels)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("binary metrics require labels in {0,1}")


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    assert_binary_numpy(labels)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def binary_ap(labels: np.ndarray, scores: np.ndarray) -> float | None:
    assert_binary_numpy(labels)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.min() == labels.max():
        return None
    return float(average_precision_score(labels, scores))


def observed_ranking_diagnostics(
    query_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Optional diagnostic on observed candidates only.

    Does not impute unobserved pairs as negatives. Uses the shared ranking_metrics
    helper, which treats label==1 as relevant.
    """
    assert_binary_numpy(labels)
    metrics = ranking_metrics(query_ids, labels, scores)
    # candidate sizes
    order = np.argsort(query_ids, kind="mergesort")
    sorted_ids = query_ids[order]
    breaks = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    sizes = [int(end - start) for start, end in zip(breaks[:-1], breaks[1:])]
    metrics["candidate_pool"] = "observed interactions only; no imputed negatives"
    metrics["n_queries_total"] = int(len(sizes))
    metrics["mean_candidates"] = float(np.mean(sizes)) if sizes else 0.0
    metrics["median_candidates"] = float(np.median(sizes)) if sizes else 0.0
    return metrics

"""Query-grouped ranking metrics for Person-Job Fit baselines.

Relevance is binary: ``label == 1`` is relevant and every other value is not.
NDCG and MRR are macro-averaged over queries that have at least one relevant
candidate. Queries with no relevant candidate are excluded from those averages
and counted in ``n_queries_without_positive``.

AUC is computed on candidate pairs. ``auc`` treats label 1 as positive and
every other label as negative. ``auc_labeled`` keeps only labels in {0, 1}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _dcg(relevances: np.ndarray) -> float:
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances * discounts))


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    """Return NDCG@k, or None when the query has no relevant item."""
    relevant = (labels == 1).astype(np.float64)
    if relevant.sum() <= 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    dcg = _dcg(relevant[order][:k])
    idcg = _dcg(np.sort(relevant)[::-1][:k])
    if idcg <= 0:
        return None
    return dcg / idcg


def mrr_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float | None:
    """Return MRR@k, or None when the query has no relevant item."""
    relevant = labels == 1
    if not np.any(relevant):
        return None
    order = np.argsort(-scores, kind="mergesort")
    hits = np.flatnonzero(relevant[order][:k])
    if hits.size == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    binary = (labels == 1).astype(np.int32)
    if binary.min() == binary.max():
        return None
    return float(roc_auc_score(binary, scores))


def ranking_metrics(
    query_ids: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    ndcg_ks: tuple[int, ...] = (5, 10),
    mrr_k: int = 10,
) -> dict[str, Any]:
    """Compute NDCG, MRR, and AUC for one ranking task.

    Parameters
    ----------
    query_ids:
        Group key for each candidate. One query is one user on rank_job, or
        one job on rank_resume.
    labels:
        ``satisfied`` values. Only 1 counts as relevant.
    scores:
        Higher scores rank earlier. Ties keep the original row order.
    """
    query_ids = np.asarray(query_ids)
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if not (len(query_ids) == len(labels) == len(scores)):
        raise ValueError("query_ids, labels, and scores must have the same length")

    ndcg_sums = {k: 0.0 for k in ndcg_ks}
    mrr_sum = 0.0
    n_with_positive = 0
    n_queries = 0
    order = np.argsort(query_ids, kind="mergesort")
    sorted_ids = query_ids[order]
    breaks = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    for start, end in zip(breaks[:-1], breaks[1:]):
        n_queries += 1
        index = order[start:end]
        group_labels = labels[index]
        group_scores = scores[index]
        query_ndcg = {k: ndcg_at_k(group_labels, group_scores, k) for k in ndcg_ks}
        query_mrr = mrr_at_k(group_labels, group_scores, mrr_k)
        if query_mrr is None:
            continue
        n_with_positive += 1
        mrr_sum += query_mrr
        for k in ndcg_ks:
            value = query_ndcg[k]
            if value is not None:
                ndcg_sums[k] += value

    labeled = np.isin(labels, (0, 1))
    metrics: dict[str, Any] = {
        "n_pairs": int(len(labels)),
        "n_queries": int(n_queries),
        "n_queries_with_positive": int(n_with_positive),
        "n_queries_without_positive": int(n_queries - n_with_positive),
        "n_positives": int((labels == 1).sum()),
        "auc": _safe_auc(labels, scores),
        "auc_labeled": _safe_auc(labels[labeled], scores[labeled]) if np.any(labeled) else None,
        "relevance": "satisfied==1 is relevant; 0 and -1 are non-relevant",
        "auc_definition": "auc uses satisfied==1 against all other labels; auc_labeled keeps satisfied in {0, 1}",
        "average": "NDCG and MRR are macro-averaged over queries with at least one satisfied==1",
    }
    for k in ndcg_ks:
        metrics[f"ndcg@{k}"] = ndcg_sums[k] / n_with_positive if n_with_positive else None
    metrics[f"mrr@{mrr_k}"] = mrr_sum / n_with_positive if n_with_positive else None
    return metrics

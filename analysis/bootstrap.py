"""Two-stage percentile bootstrap for MiTA E4/E1/E2 effects.

Stage 1: resample training seeds with replacement (all cells of a seed travel together).
Stage 2: resample test users (or jobs) with replacement (all pairs of a drawn cluster).
Pooled AUC on resampled pairs (multiplicity via sample_weight); average over seeds; form effects.

RNG seed default: 20261026. B default: 10000.

Engineering: weighted AUC is a vectorized Mann–Whitney (bit-identical to sklearn on checks);
chunks run under ProcessPoolExecutor to bypass the GIL. B and RNG are unchanged.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


CELL_ORDER = ("base", "w", "t", "wt")

# Per-process state for ProcessPool workers (set by initializer).
_STATE: Dict[str, object] = {}


def weighted_rank_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """Weighted ROC-AUC via Mann–Whitney with multiplicity weights (O(n log n))."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m = w > 0
    if not np.any(m):
        return float("nan")
    y = y[m]
    s = s[m]
    w = w[m]
    order = np.argsort(s, kind="mergesort")
    y = y[order]
    s = s[order]
    w = w[order]
    distinct = np.empty(s.size, dtype=bool)
    distinct[0] = True
    distinct[1:] = s[1:] != s[:-1]
    gid = np.cumsum(distinct) - 1
    n_groups = int(gid[-1]) + 1
    g_pos = np.bincount(gid, weights=w * (y == 1), minlength=n_groups)
    g_neg = np.bincount(gid, weights=w * (y == 0), minlength=n_groups)
    w_pos = float(g_pos.sum())
    w_neg = float(g_neg.sum())
    if w_pos <= 0.0 or w_neg <= 0.0:
        return float("nan")
    cum_before = np.cumsum(g_pos) - g_pos
    pos_higher = w_pos - cum_before - g_pos
    num = float((g_neg * pos_higher + 0.5 * g_neg * g_pos).sum())
    return num / (w_pos * w_neg)


def rank_auc(y: np.ndarray, s: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    if sample_weight is None:
        if y.sum() == 0 or y.sum() == y.size:
            return float("nan")
        return float(roc_auc_score(y, s))
    return weighted_rank_auc(y, s, sample_weight)


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


@dataclass
class CellPreds:
    seed: int
    user_id: np.ndarray
    y: np.ndarray
    scores: Dict[str, np.ndarray]


def effects_from_means(means: Mapping[str, float]) -> Dict[str, float]:
    b, w, t, wt = means["base"], means["w"], means["t"], means["wt"]
    return {
        "W": 0.5 * ((w - b) + (wt - t)),
        "T": 0.5 * ((t - b) + (wt - w)),
        "I": (wt - t) - (w - b),
        "WT": wt - b,
    }


def _default_workers(n_jobs: Optional[int]) -> int:
    return n_jobs if n_jobs is not None else min(8, os.cpu_count() or 4)


def _init_effects_state(
    y: np.ndarray,
    inv: np.ndarray,
    n_clusters: int,
    score_stack: np.ndarray,
    seed_draws: np.ndarray,
    cl_draws: np.ndarray,
) -> None:
    _STATE.clear()
    _STATE.update(
        y=y,
        inv=inv,
        n_clusters=n_clusters,
        score_stack=score_stack,
        seed_draws=seed_draws,
        cl_draws=cl_draws,
    )


def _boot_chunk_effects(start_end: Tuple[int, int]) -> np.ndarray:
    start, end = start_end
    y = _STATE["y"]  # type: ignore[assignment]
    inv = _STATE["inv"]  # type: ignore[assignment]
    n_clusters = int(_STATE["n_clusters"])  # type: ignore[arg-type]
    score_stack = _STATE["score_stack"]  # type: ignore[assignment]
    seed_draws = _STATE["seed_draws"]  # type: ignore[assignment]
    cl_draws = _STATE["cl_draws"]  # type: ignore[assignment]
    out = np.empty((end - start, 4), dtype=np.float64)
    for j, b_i in enumerate(range(start, end)):
        seed_idx = seed_draws[b_i]
        mult = np.bincount(cl_draws[b_i], minlength=n_clusters).astype(np.float64)
        w = mult[inv]
        means = {}
        for ci, cname in enumerate(CELL_ORDER):
            aucs = [weighted_rank_auc(y, score_stack[si, ci], w) for si in seed_idx]
            means[cname] = float(np.mean(aucs))
        eff = effects_from_means(means)
        out[j, 0] = eff["W"]
        out[j, 1] = eff["T"]
        out[j, 2] = eff["I"]
        out[j, 3] = eff["WT"]
    return out


def two_stage_bootstrap(
    cell_preds: Sequence[CellPreds],
    *,
    B: int = 10_000,
    rng_seed: int = 20261026,
    cluster: str = "user",
    job_id: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = None,
) -> Dict[str, object]:
    if not cell_preds:
        raise ValueError("empty cell_preds")
    n_seeds = len(cell_preds)
    user_id = cell_preds[0].user_id
    y = np.asarray(cell_preds[0].y, dtype=np.int8)
    for cp in cell_preds[1:]:
        if not np.array_equal(cp.y, y):
            raise ValueError("misaligned CellPreds")

    point_means = {
        c: float(np.mean([rank_auc(cp.y, cp.scores[c]) for cp in cell_preds])) for c in CELL_ORDER
    }
    point_eff = effects_from_means(point_means)

    if cluster == "user":
        cluster_ids = user_id
    elif cluster == "job":
        if job_id is None:
            raise ValueError("job_id required for job cluster")
        cluster_ids = job_id
    else:
        raise ValueError(cluster)

    _, inv = np.unique(cluster_ids, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    n = int(y.size)
    score_stack = np.empty((n_seeds, 4, n), dtype=np.float64)
    for si, cp in enumerate(cell_preds):
        for ci, c in enumerate(CELL_ORDER):
            score_stack[si, ci] = cp.scores[c]

    rng = np.random.default_rng(rng_seed)
    seed_draws = rng.integers(0, n_seeds, size=(B, n_seeds))
    cl_draws = rng.integers(0, n_clusters, size=(B, n_clusters))

    workers = _default_workers(n_jobs)
    edges = np.linspace(0, B, workers + 1, dtype=int)
    chunks = [(int(edges[i]), int(edges[i + 1])) for i in range(workers) if edges[i + 1] > edges[i]]
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=_init_effects_state,
        initargs=(y, inv, n_clusters, score_stack, seed_draws, cl_draws),
    ) as ex:
        parts = list(ex.map(_boot_chunk_effects, chunks))
    mat = np.concatenate(parts, axis=0)
    boot = {"W": mat[:, 0], "T": mat[:, 1], "I": mat[:, 2], "WT": mat[:, 3]}

    out: Dict[str, object] = {
        "point_means": point_means,
        "point_effects": point_eff,
        "B": B,
        "cluster": cluster,
    }
    for k in ("W", "T", "I", "WT"):
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        out[k] = {"est": point_eff[k], "lo": float(lo), "hi": float(hi), "boot": boot[k]}
    return out


def _init_contrast_state(
    y: np.ndarray,
    inv: np.ndarray,
    n_clusters: int,
    scores_a: list,
    scores_b: list,
    ia_all: np.ndarray,
    ib_all: np.ndarray,
    cl_all: np.ndarray,
) -> None:
    _STATE.clear()
    _STATE.update(
        y=y,
        inv=inv,
        n_clusters=n_clusters,
        scores_a=scores_a,
        scores_b=scores_b,
        ia_all=ia_all,
        ib_all=ib_all,
        cl_all=cl_all,
    )


def _contrast_chunk(start_end: Tuple[int, int]) -> np.ndarray:
    start, end = start_end
    y = _STATE["y"]  # type: ignore[assignment]
    inv = _STATE["inv"]  # type: ignore[assignment]
    n_clusters = int(_STATE["n_clusters"])  # type: ignore[arg-type]
    scores_a = _STATE["scores_a"]  # type: ignore[assignment]
    scores_b = _STATE["scores_b"]  # type: ignore[assignment]
    ia_all = _STATE["ia_all"]  # type: ignore[assignment]
    ib_all = _STATE["ib_all"]  # type: ignore[assignment]
    cl_all = _STATE["cl_all"]  # type: ignore[assignment]
    out = np.empty(end - start, dtype=np.float64)
    for j, b_i in enumerate(range(start, end)):
        mult = np.bincount(cl_all[b_i], minlength=n_clusters).astype(np.float64)
        w = mult[inv]
        ma = float(np.mean([weighted_rank_auc(y, scores_a[i], w) for i in ia_all[b_i]]))
        mb = float(np.mean([weighted_rank_auc(y, scores_b[i], w) for i in ib_all[b_i]]))
        out[j] = ma - mb
    return out


def bootstrap_contrast_two_panels(
    panel_a: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    panel_b: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    B: int = 10_000,
    rng_seed: int = 20261026,
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    """mean(A)-mean(B); independent seed resample; shared user resample."""
    user_id, y, _ = panel_a[0]
    y = np.asarray(y, dtype=np.int8)
    _, inv = np.unique(user_id, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    n_a, n_b = len(panel_a), len(panel_b)
    scores_a = [np.asarray(s, dtype=np.float64) for _, _, s in panel_a]
    scores_b = [np.asarray(s, dtype=np.float64) for _, _, s in panel_b]
    point = float(np.mean([rank_auc(y, s) for s in scores_a])) - float(
        np.mean([rank_auc(y, s) for s in scores_b])
    )
    rng = np.random.default_rng(rng_seed)
    ia_all = rng.integers(0, n_a, size=(B, n_a))
    ib_all = rng.integers(0, n_b, size=(B, n_b))
    cl_all = rng.integers(0, n_clusters, size=(B, n_clusters))

    workers = _default_workers(n_jobs)
    edges = np.linspace(0, B, workers + 1, dtype=int)
    chunks = [(int(edges[i]), int(edges[i + 1])) for i in range(workers) if edges[i + 1] > edges[i]]
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=_init_contrast_state,
        initargs=(y, inv, n_clusters, scores_a, scores_b, ia_all, ib_all, cl_all),
    ) as ex:
        boot = np.concatenate(list(ex.map(_contrast_chunk, chunks)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"est": point, "lo": float(lo), "hi": float(hi)}


def _init_b0_state(
    y: np.ndarray,
    inv: np.ndarray,
    n_clusters: int,
    scores: list,
    b0_score: np.ndarray,
    ia_all: np.ndarray,
    cl_all: np.ndarray,
) -> None:
    _STATE.clear()
    _STATE.update(
        y=y,
        inv=inv,
        n_clusters=n_clusters,
        scores=scores,
        b0_score=b0_score,
        ia_all=ia_all,
        cl_all=cl_all,
    )


def _b0_chunk(start_end: Tuple[int, int]) -> np.ndarray:
    start, end = start_end
    y = _STATE["y"]  # type: ignore[assignment]
    inv = _STATE["inv"]  # type: ignore[assignment]
    n_clusters = int(_STATE["n_clusters"])  # type: ignore[arg-type]
    scores = _STATE["scores"]  # type: ignore[assignment]
    b0_score = _STATE["b0_score"]  # type: ignore[assignment]
    ia_all = _STATE["ia_all"]  # type: ignore[assignment]
    cl_all = _STATE["cl_all"]  # type: ignore[assignment]
    out = np.empty(end - start, dtype=np.float64)
    for j, b_i in enumerate(range(start, end)):
        mult = np.bincount(cl_all[b_i], minlength=n_clusters).astype(np.float64)
        w = mult[inv]
        mb = float(np.mean([weighted_rank_auc(y, scores[i], w) for i in ia_all[b_i]]))
        out[j] = weighted_rank_auc(y, b0_score, w) - mb
    return out


def bootstrap_b0_gap(
    base_preds: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    b0_score: np.ndarray,
    *,
    B: int = 10_000,
    rng_seed: int = 20261026,
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    user_id, y, _ = base_preds[0]
    y = np.asarray(y, dtype=np.int8)
    b0_score = np.asarray(b0_score, dtype=np.float64)
    _, inv = np.unique(user_id, return_inverse=True)
    n_clusters = int(inv.max()) + 1
    n_seeds = len(base_preds)
    scores = [np.asarray(s, dtype=np.float64) for _, _, s in base_preds]
    point_base = float(np.mean([rank_auc(y, s) for s in scores]))
    point_b0 = rank_auc(y, b0_score)
    point = point_b0 - point_base
    rng = np.random.default_rng(rng_seed)
    ia_all = rng.integers(0, n_seeds, size=(B, n_seeds))
    cl_all = rng.integers(0, n_clusters, size=(B, n_clusters))

    workers = _default_workers(n_jobs)
    edges = np.linspace(0, B, workers + 1, dtype=int)
    chunks = [(int(edges[i]), int(edges[i + 1])) for i in range(workers) if edges[i + 1] > edges[i]]
    with ProcessPoolExecutor(
        max_workers=len(chunks),
        initializer=_init_b0_state,
        initargs=(y, inv, n_clusters, scores, b0_score, ia_all, cl_all),
    ) as ex:
        boot = np.concatenate(list(ex.map(_b0_chunk, chunks)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"est": point, "lo": float(lo), "hi": float(hi), "auc_b0": point_b0, "auc_base": point_base}

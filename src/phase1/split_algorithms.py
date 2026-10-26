"""Deterministic U2 / J1 split algorithms (pure functions; no Tianchi materialize)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

MASS_TARGET = (0.8, 0.1, 0.1)
MASS_TOL = 0.005
U_SPLIT_SEED = 2026102317
J_SPLIT_SEED = 2026102318
MAX_SEED_TRIES = 100


@dataclass(frozen=True)
class SplitResult:
    train_ids: frozenset
    val_ids: frozenset
    test_ids: frozenset
    accepted_seed: int
    algorithm: str
    mass_frac: tuple[float, float, float]


def _both_classes(labels: Iterable[int]) -> bool:
    s = set(int(x) for x in labels)
    return 0 in s and 1 in s


def _mass_ok(n_train: int, n_val: int, n_test: int, total: int) -> bool:
    if total <= 0:
        return False
    fr = (n_train / total, n_val / total, n_test / total)
    return all(abs(a - b) <= MASS_TOL for a, b in zip(fr, MASS_TARGET))


def _greedy_pack(
    entity_ids: list,
    mass: dict,
    rng: np.random.Generator,
) -> tuple[set, set, set] | None:
    """Assign entities atomically to train/val/test by remaining mass deficit.

    Tie-break when deficits equal: prefer train, then val, then test (fixed order).
    Within a step, entities are consumed in the given ``entity_ids`` order
    (caller must make that order deterministic).
    """
    buckets = {"train": set(), "val": set(), "test": set()}
    rem = {k: MASS_TARGET[i] * sum(mass.values()) for i, k in enumerate(("train", "val", "test"))}
    order = list(entity_ids)
    # deterministic secondary shuffle of equal-priority stream already fixed by caller;
    # here we only consume in order.
    for eid in order:
        m = float(mass[eid])
        # choose bucket with largest remaining deficit; ties -> train, val, test
        best = max(("train", "val", "test"), key=lambda b: (rem[b], {"train": 2, "val": 1, "test": 0}[b]))
        buckets[best].add(eid)
        rem[best] -= m
    return buckets["train"], buckets["val"], buckets["test"]


def _interaction_bins(counts: pd.Series) -> dict:
    """Frozen quartile edges from unique sorted counts (duplicates: left-closed)."""
    vals = np.sort(counts.to_numpy(dtype=np.float64))
    q = np.quantile(vals, [0.25, 0.5, 0.75], method="linear")
    edges = (float(q[0]), float(q[1]), float(q[2]))
    return {"q25": edges[0], "q50": edges[1], "q75": edges[2]}


def _bin_label(c: float, edges: dict) -> int:
    if c <= edges["q25"]:
        return 0
    if c <= edges["q50"]:
        return 1
    if c <= edges["q75"]:
        return 2
    return 3


def u2_stratified_mass_pack(
    pairs: pd.DataFrame,
    split_seed: int = U_SPLIT_SEED,
    allow_u1_fallback: bool = False,
) -> SplitResult:
    """U2: strata = dual_label × interaction quartile; mass-pack within strata.

    Retry order (LOCKED):
      for k in 0..99: try U2(seed+k)
      if allow_u1_fallback: for k in 0..99: try U1(seed+k)
      else: raise

    Default ``allow_u1_fallback=False`` — U1 requires explicit researcher approval.
    """
    required = {"user_id", "jd_no", "satisfied"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"pairs missing {required - set(pairs.columns)}")

    user_counts = pairs.groupby("user_id").size()
    user_pos = pairs.groupby("user_id")["satisfied"].apply(lambda s: set(int(x) for x in s))
    dual = {u: (0 in labs and 1 in labs) for u, labs in user_pos.items()}
    edges = _interaction_bins(user_counts)

    def try_u2(seed: int) -> SplitResult | None:
        rng = np.random.default_rng(seed)
        # stable user list
        users = sorted(user_counts.index.astype(str).tolist())
        strata: dict[tuple[int, int], list[str]] = {}
        for u in users:
            key = (1 if dual[u] else 0, _bin_label(float(user_counts[u]), edges))
            strata.setdefault(key, []).append(u)
        # within stratum: sort then permute with stratum-local stream
        assigned_train, assigned_val, assigned_test = set(), set(), set()
        mass = {u: float(user_counts[u]) for u in users}
        # pack stratum by stratum in sorted key order; concatenate pack results
        # Global pack: permute all users with strata-aware order
        ordered: list[str] = []
        for key in sorted(strata.keys()):
            members = list(strata[key])
            members.sort()  # stable
            perm = rng.permutation(len(members))
            ordered.extend(members[i] for i in perm)
        packed = _greedy_pack(ordered, mass, rng)
        if packed is None:
            return None
        tr, va, te = packed
        if tr & va or tr & te or va & te:
            return None
        # assign pairs
        part = {}
        for name, ids in ("train", tr), ("val", va), ("test", te):
            part[name] = pairs[pairs["user_id"].astype(str).isin(ids)]
        n_tr, n_va, n_te = len(part["train"]), len(part["val"]), len(part["test"])
        total = n_tr + n_va + n_te
        if not _mass_ok(n_tr, n_va, n_te, total):
            return None
        if not _both_classes(part["val"]["satisfied"]) or not _both_classes(part["test"]["satisfied"]):
            return None
        return SplitResult(
            frozenset(tr),
            frozenset(va),
            frozenset(te),
            seed,
            "U2_STRATIFIED_MASS_PACK",
            (n_tr / total, n_va / total, n_te / total),
        )

    def try_u1(seed: int) -> SplitResult | None:
        rng = np.random.default_rng(seed)
        users = sorted(user_counts.index.astype(str).tolist())
        perm = rng.permutation(len(users))
        ordered = [users[i] for i in perm]
        mass = {u: float(user_counts[u]) for u in users}
        packed = _greedy_pack(ordered, mass, rng)
        if packed is None:
            return None
        tr, va, te = packed
        part = {
            "train": pairs[pairs["user_id"].astype(str).isin(tr)],
            "val": pairs[pairs["user_id"].astype(str).isin(va)],
            "test": pairs[pairs["user_id"].astype(str).isin(te)],
        }
        n_tr, n_va, n_te = len(part["train"]), len(part["val"]), len(part["test"])
        total = n_tr + n_va + n_te
        if not _mass_ok(n_tr, n_va, n_te, total):
            return None
        if not _both_classes(part["val"]["satisfied"]) or not _both_classes(part["test"]["satisfied"]):
            return None
        return SplitResult(
            frozenset(tr),
            frozenset(va),
            frozenset(te),
            seed,
            "U1_MASS_PACK",
            (n_tr / total, n_va / total, n_te / total),
        )

    for k in range(MAX_SEED_TRIES):
        got = try_u2(split_seed + k)
        if got is not None:
            return got
    if allow_u1_fallback:
        for k in range(MAX_SEED_TRIES):
            got = try_u1(split_seed + k)
            if got is not None:
                return got
    raise RuntimeError(
        f"U2 failed for seeds {split_seed}..{split_seed + MAX_SEED_TRIES - 1}; "
        f"U1 fallback={'on' if allow_u1_fallback else 'OFF — STOP for protocol revision'}"
    )


def j1_mass_pack_jobs(pairs: pd.DataFrame, split_seed: int = J_SPLIT_SEED) -> SplitResult:
    """J1 LOCKED algorithm:

    1. Group by jd_no; mass = interaction count.
    2. stable_sort by (-count, jd_no ascending).
    3. Apply rng.permutation to that list (seed).
    4. Greedy mass-pack whole jobs.
    5. Accept first seed in seed..seed+99 that meets mass+both-class constraints.
    """
    if not {"user_id", "jd_no", "satisfied"}.issubset(pairs.columns):
        raise ValueError("pairs missing required columns")
    job_counts = pairs.groupby("jd_no").size()
    jobs = list(job_counts.index.astype(str))
    # stable_sort (-count, jd_no)
    jobs.sort(key=lambda j: (-int(job_counts[j]), j))

    for k in range(MAX_SEED_TRIES):
        seed = split_seed + k
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(jobs))
        ordered = [jobs[i] for i in perm]
        mass = {j: float(job_counts[j]) for j in jobs}
        tr, va, te = _greedy_pack(ordered, mass, rng)
        if tr & va or tr & te or va & te:
            continue
        part = {
            "train": pairs[pairs["jd_no"].astype(str).isin(tr)],
            "val": pairs[pairs["jd_no"].astype(str).isin(va)],
            "test": pairs[pairs["jd_no"].astype(str).isin(te)],
        }
        n_tr, n_va, n_te = len(part["train"]), len(part["val"]), len(part["test"])
        total = n_tr + n_va + n_te
        if not _mass_ok(n_tr, n_va, n_te, total):
            continue
        if not _both_classes(part["val"]["satisfied"]) or not _both_classes(part["test"]["satisfied"]):
            continue
        return SplitResult(
            frozenset(tr),
            frozenset(va),
            frozenset(te),
            seed,
            "J1_MASS_PACK_JOBS",
            (n_tr / total, n_va / total, n_te / total),
        )
    raise RuntimeError(f"J1 failed for seeds {split_seed}..{split_seed + MAX_SEED_TRIES - 1}; STOP")

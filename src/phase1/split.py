"""Frozen 8:1:1 interaction split for Phase 1 (seed 42 for the split only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase1.paths import ACTION_PATH, RANK_JOB_PATH, RANK_RESUME_PATH, SPLIT_DIR, ensure_phase1_dirs

SPLIT_SEED = 42
EXPECTED = {"train": 123248, "val": 15406, "internal_test": 15407}


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    frame.columns = [
        str(column).replace("\ufeff", "").replace("\n", "").replace("\r", "").strip()
        for column in frame.columns
    ]
    return frame


def load_rank_pair_set() -> set[tuple[str, str]]:
    rank_job = _read_csv(RANK_JOB_PATH)
    rank_resume = _read_csv(RANK_RESUME_PATH)
    return set(zip(rank_job["user_id"], rank_job["jd_no"])) | set(zip(rank_resume["user_id"], rank_resume["jd_no"]))


def build_frozen_split(action_path: Path = ACTION_PATH) -> dict:
    actions = _read_csv(action_path)
    satisfied = actions["satisfied"].astype(int)
    unique = sorted(set(satisfied.tolist()))
    if unique != [0, 1]:
        raise RuntimeError(f"satisfied must be {{0,1}}, got {unique}")

    rank_pairs = load_rank_pair_set()
    held = np.array(
        [(user_id, job_id) in rank_pairs for user_id, job_id in zip(actions["user_id"], actions["jd_no"])],
        dtype=bool,
    )
    trainable = actions.loc[~held].reset_index(drop=True)
    expected_total = sum(EXPECTED.values())
    if len(trainable) != expected_total:
        raise RuntimeError(
            f"trainable size {len(trainable)} != expected {expected_total}; held_out={int(held.sum())}"
        )

    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(trainable))
    n_train = int(len(trainable) * 0.8)
    n_val = int(len(trainable) * 0.1)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    parts = {
        "train": trainable.iloc[train_idx].reset_index(drop=True),
        "val": trainable.iloc[val_idx].reset_index(drop=True),
        "internal_test": trainable.iloc[test_idx].reset_index(drop=True),
    }
    for name, expected in EXPECTED.items():
        if len(parts[name]) != expected:
            raise RuntimeError(f"{name} size {len(parts[name])} != expected {expected}")

    pair_sets = {name: set(zip(frame["user_id"], frame["jd_no"])) for name, frame in parts.items()}
    if pair_sets["train"] & pair_sets["val"]:
        raise RuntimeError("train/val pair overlap")
    if pair_sets["train"] & pair_sets["internal_test"]:
        raise RuntimeError("train/internal_test pair overlap")
    if pair_sets["val"] & pair_sets["internal_test"]:
        raise RuntimeError("val/internal_test pair overlap")
    for name, frame in parts.items():
        if frame.duplicated(["user_id", "jd_no"]).any():
            raise RuntimeError(f"{name} has duplicate pairs")

    ensure_phase1_dirs()
    meta: dict = {
        "split_seed": SPLIT_SEED,
        "held_out_rank_pairs": int(held.sum()),
        "rank_pair_set_size": len(rank_pairs),
        "action_path": str(action_path),
        "expected_sizes": EXPECTED,
        "note": "Training seeds 42/123/456 must not regenerate this split.",
        "parts": {},
    }
    np.save(SPLIT_DIR / "train_order.npy", train_idx)
    np.save(SPLIT_DIR / "val_order.npy", val_idx)
    np.save(SPLIT_DIR / "internal_test_order.npy", test_idx)
    for name, frame in parts.items():
        pairs_path = SPLIT_DIR / f"{name}_pairs.csv"
        out = frame[["user_id", "jd_no", "satisfied"]].copy()
        out.to_csv(pairs_path, index=False)
        sat = out["satisfied"].astype(int)
        meta["parts"][name] = {
            "path": str(pairs_path),
            "n_rows": int(len(out)),
            "n_users": int(out["user_id"].nunique()),
            "n_jobs": int(out["jd_no"].nunique()),
            "n_pos": int((sat == 1).sum()),
            "n_neg": int((sat == 0).sum()),
            "sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        }
    meta_path = SPLIT_DIR / "split_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_split_pairs(name: str) -> pd.DataFrame:
    """Load pairs from the Phase-1 default SPLIT_DIR (legacy Overlap)."""
    return load_split_pairs_from_dir(SPLIT_DIR, name)


def load_split_pairs_from_dir(split_dir: Path, name: str) -> pd.DataFrame:
    """Load ``{name}_pairs.csv`` from an explicit split directory (TCORE binding)."""
    path = Path(split_dir) / f"{name}_pairs.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["satisfied"] = frame["satisfied"].astype(int)
    if set(frame["satisfied"].tolist()) - {0, 1}:
        raise RuntimeError(f"{name} labels not in {{0,1}}")
    return frame

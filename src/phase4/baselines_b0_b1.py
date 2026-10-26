"""Phase 4A Gate B: B0 ID-prior and B1 logistic regression (validation only).

Does NOT read or score the held-out internal_test split.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.phase1.metrics import binary_ap, binary_auc
from src.phase1.paths import JOB_EMB_PATH, ROOT, USER_EMB_PATH
from src.phase1.split import load_split_pairs

OUT_DIR = ROOT / "results" / "phase4" / "phase4a" / "baselines"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BETAS = (0.5, 1.0, 2.0, 5.0)
N_MINS = (1, 3, 5)
CS = (0.01, 0.1, 1.0, 10.0)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _pair_features(user_emb: np.ndarray, job_emb: np.ndarray, u_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    hu = user_emb[u_idx]
    hj = job_emb[j_idx]
    return np.concatenate([hu, hj, hu * hj, np.abs(hu - hj)], axis=1).astype(np.float64)


def run_b0() -> dict:
    train = load_split_pairs("train")
    val = load_split_pairs("val")
    y_tr = train["satisfied"].to_numpy(dtype=np.float64)
    p_g = float(y_tr.mean())

    user_pos = train.groupby("user_id")["satisfied"].sum()
    user_cnt = train.groupby("user_id")["satisfied"].count()
    job_pos = train.groupby("jd_no")["satisfied"].sum()
    job_cnt = train.groupby("jd_no")["satisfied"].count()

    def rates_for(ids, pos_map, cnt_map, beta: float, n_min: int) -> tuple[np.ndarray, np.ndarray]:
        rates = np.full(len(ids), p_g, dtype=np.float64)
        seen = np.zeros(len(ids), dtype=bool)
        for i, entity_id in enumerate(ids):
            if entity_id not in cnt_map.index:
                continue
            c = float(cnt_map.loc[entity_id])
            if c < n_min:
                continue
            rates[i] = (float(pos_map.loc[entity_id]) + beta * p_g) / (c + beta)
            seen[i] = True
        return rates, seen

    y_val = val["satisfied"].to_numpy(dtype=np.float64)
    val_users = val["user_id"].tolist()
    val_jobs = val["jd_no"].tolist()
    rows = []
    best = None
    for beta in BETAS:
        for n_min in N_MINS:
            pu, seen_u = rates_for(val_users, user_pos, user_cnt, beta, n_min)
            pj, seen_j = rates_for(val_jobs, job_pos, job_cnt, beta, n_min)
            for alpha in ALPHAS:
                scores_arr = alpha * pu + (1.0 - alpha) * pj
                auc = binary_auc(y_val, scores_arr)
                ap = binary_ap(y_val, scores_arr)
                rec = {
                    "alpha": alpha,
                    "beta": beta,
                    "n_min": n_min,
                    "val_auc": auc,
                    "val_ap": ap,
                    "user_seen_frac": float(seen_u.mean()),
                    "job_seen_frac": float(seen_j.mean()),
                    "p_global": p_g,
                }
                rows.append(rec)
                key = (auc if auc is not None else -1.0, -alpha, -beta, n_min)
                if best is None or key > best[0]:
                    best = (key, rec)

    selected = best[1]
    payload = {
        "baseline": "B0_id_prior",
        "formula": "p_entity=(pos+beta*p_global)/(count+beta); score=alpha*p_u+(1-alpha)*p_j",
        "selected": selected,
        "grid_results": rows,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "split_sha256": {
            "train": _sha(ROOT / "data/phase1/splits/train_pairs.csv"),
            "val": _sha(ROOT / "data/phase1/splits/val_pairs.csv"),
        },
        "internal_test_touched": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "B0_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_b1() -> dict:
    train = load_split_pairs("train")
    val = load_split_pairs("val")
    user_payload = torch.load(USER_EMB_PATH, map_location="cpu", weights_only=False)
    job_payload = torch.load(JOB_EMB_PATH, map_location="cpu", weights_only=False)
    user_ids = [str(x) for x in user_payload["ids"]]
    job_ids = [str(x) for x in job_payload["ids"]]
    user_row = {u: i for i, u in enumerate(user_ids)}
    job_row = {j: i for i, j in enumerate(job_ids)}
    u_emb = user_payload["embeddings"].numpy()
    j_emb = job_payload["embeddings"].numpy()

    def indices(frame: pd.DataFrame):
        u = np.array([user_row[str(x)] for x in frame["user_id"]], dtype=np.int64)
        j = np.array([job_row[str(x)] for x in frame["jd_no"]], dtype=np.int64)
        y = frame["satisfied"].to_numpy(dtype=np.float64)
        return u, j, y

    tr_u, tr_j, y_tr = indices(train)
    va_u, va_j, y_va = indices(val)
    x_tr = _pair_features(u_emb, j_emb, tr_u, tr_j)
    x_va = _pair_features(u_emb, j_emb, va_u, va_j)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_va_s = scaler.transform(x_va)

    rows = []
    best = None
    t0 = time.time()
    for c in CS:
        clf = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            random_state=0,
        )
        clf.fit(x_tr_s, y_tr)
        scores = clf.decision_function(x_va_s)
        auc = binary_auc(y_va, scores)
        ap = binary_ap(y_va, scores)
        rec = {
            "C": c,
            "val_auc": auc,
            "val_ap": ap,
            "n_iter": int(clf.n_iter_[0]) if hasattr(clf, "n_iter_") else None,
            "converged": bool(clf.n_iter_[0] < 500) if hasattr(clf, "n_iter_") else None,
        }
        rows.append(rec)
        key = (auc if auc is not None else -1.0, -c)
        if best is None or key > best[0]:
            best = (key, rec, clf)

    selected = best[1]
    payload = {
        "baseline": "B1_logistic_regression",
        "feature_dim": int(x_tr.shape[1]),
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 500,
        "selected": selected,
        "grid_results": rows,
        "elapsed_sec": time.time() - t0,
        "user_emb_sha256": _sha(USER_EMB_PATH),
        "job_emb_sha256": _sha(JOB_EMB_PATH),
        "internal_test_touched": False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "B1_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    print("Running B0 (no test)...", flush=True)
    b0 = run_b0()
    print("B0 selected", b0["selected"], flush=True)
    print("Running B1 (no test)...", flush=True)
    b1 = run_b1()
    print("B1 selected", b1["selected"], flush=True)


if __name__ == "__main__":
    main()

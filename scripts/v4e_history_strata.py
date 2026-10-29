#!/usr/bin/env python3
"""Addendum e: B0 vs Base by training history (cutpoints FIRST, then AUCs). No training."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT
CUTPOINTS = ROOT / "results/v4e_cutpoints.json"
OUT_TEX = ROOT / "results" / "results_macros_v4e.tex"
OUT_REPORT = ROOT / "results/results_report_v4.md"
OUT_JSON = ROOT / "results/v4e_history_strata.json"

TRAIN = ROOT / "data/phase1/splits/train_pairs.csv"
TEST = ROOT / "data/phase1/splits/internal_test_pairs.csv"
B0_PRED = ROOT / "results/phase4/phase4a/gate_c/predictions/B0/predictions.csv"
E1_RUNS = ROOT / "results/e1/runs"
BASE_SEEDS = list(range(1001, 1011))
B = 10_000
RNG_SEED = 20261029
PREREG_COMMIT = "bc49ebbaaf5091d796f7744894cf195fa9334885"


def latex_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def latex_signed4(x: float) -> str:
    return f"{x:+.4f}"


def latex_auc4(x: float) -> str:
    return f"{x:.4f}"


def latex_range(lo: int, hi: int) -> str:
    if hi >= 1000:
        return f"{lo}--{latex_int(hi)}"
    return f"{lo}--{hi}"


def pool_auc_fast(y: np.ndarray, s: np.ndarray) -> float | None:
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pool_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    # Point estimates use sklearn; bootstrap uses pool_auc_fast.
    y = np.asarray(y)
    if y.size < 2 or y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


# Globals for process-pool workers
_G = {}


def _init_worker(payload: dict) -> None:
    _G.update(payload)


def _boot_chunk(spec: tuple[int, int, int]) -> tuple[dict, int]:
    start, nrep, seed = spec
    rng = np.random.default_rng(seed)
    y = _G["y"]
    b0_score = _G["b0_score"]
    base_scores = _G["base_scores"]
    jh = _G["jh"]
    uh = _G["uh"]
    user_ptr = _G["user_ptr"]
    jc1, jc2 = _G["jc1"], _G["jc2"]
    uc1, uc2 = _G["uc1"], _G["uc2"]
    thirds = _G["thirds"]
    n_seeds = base_scores.shape[0]
    n_users = len(user_ptr)
    boot = {k: [] for k in _G["boot_keys"]}
    skipped = 0
    for _ in range(nrep):
        seed_ix = rng.integers(0, n_seeds, size=n_seeds)
        drawn_ix = rng.integers(0, n_users, size=n_users)
        idxs = np.concatenate([user_ptr[i] for i in drawn_ix])
        y_b = y[idxs]
        b0_b = b0_score[idxs]
        base_b = base_scores[seed_ix][:, idxs]
        jh_b = jh[idxs]
        uh_b = uh[idxs]
        ok = True
        gaps = {}
        for g, hist_b, c1, c2 in (
            ("J", jh_b, jc1, jc2),
            ("U", uh_b, uc1, uc2),
        ):
            for t in thirds:
                m = stratum_mask(hist_b, c1, c2, t)
                ym = y_b[m]
                if ym.size < 2 or ym.min() == ym.max():
                    ok = False
                    break
                a0 = pool_auc_fast(ym, b0_b[m])
                if a0 is None:
                    ok = False
                    break
                seed_aucs = []
                for s in range(n_seeds):
                    a = pool_auc_fast(ym, base_b[s, m])
                    if a is None:
                        ok = False
                        break
                    seed_aucs.append(a)
                if not ok:
                    break
                gaps[f"{g}{t}"] = a0 - float(np.mean(seed_aucs))
            if not ok:
                break
        if not ok:
            skipped += 1
            continue
        for k, v in gaps.items():
            boot[k].append(v)
        boot["JTrend"].append(gaps["JHigh"] - gaps["JLow"])
        boot["UTrend"].append(gaps["UHigh"] - gaps["ULow"])
    return boot, skipped


def stratum_mask(hist: np.ndarray, c1: float, c2: float, third: str) -> np.ndarray:
    # Ties at a cut point go to the lower third.
    if third == "Low":
        return hist <= c1
    if third == "Mid":
        return (hist > c1) & (hist <= c2)
    if third == "High":
        return hist > c2
    raise ValueError(third)


def write_cutpoints(n_test: int, jh: np.ndarray, uh: np.ndarray) -> dict:
    jc1, jc2 = np.quantile(jh, [1 / 3, 2 / 3], method="linear")
    uc1, uc2 = np.quantile(uh, [1 / 3, 2 / 3], method="linear")
    payload = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_commit": PREREG_COMMIT,
        "n_test_pairs": int(n_test),
        "job": {
            "q33": float(jc1),
            "q66": float(jc2),
            "rule": "Low:h<=q33; Mid:q33<h<=q66; High:h>q66 (ties at cut -> lower)",
        },
        "user": {
            "q33": float(uc1),
            "q66": float(uc2),
            "rule": "Low:h<=q33; Mid:q33<h<=q66; High:h>q66 (ties at cut -> lower)",
        },
        "job_hist_on_test_min_max": [int(jh.min()), int(jh.max())],
        "user_hist_on_test_min_max": [int(uh.min()), int(uh.max())],
    }
    CUTPOINTS.parent.mkdir(parents=True, exist_ok=True)
    CUTPOINTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def align_scores(test: pd.DataFrame, pred: pd.DataFrame, score_col: str = "score") -> np.ndarray:
    """Return score vector aligned to test rows (user_id, jd_no)."""
    p = pred.copy()
    if "job_id" in p.columns and "jd_no" not in p.columns:
        p = p.rename(columns={"job_id": "jd_no"})
    p["user_id"] = p["user_id"].astype(str)
    p["jd_no"] = p["jd_no"].astype(str)
    t_user = test["user_id"].astype(str).to_numpy()
    t_job = test["jd_no"].astype(str).to_numpy()
    if (
        len(p) == len(test)
        and (p["user_id"].to_numpy() == t_user).all()
        and (p["jd_no"].to_numpy() == t_job).all()
    ):
        return p[score_col].to_numpy().astype(float)
    m = test.assign(user_id=test["user_id"].astype(str), jd_no=test["jd_no"].astype(str))[
        ["user_id", "jd_no"]
    ].merge(
        p[["user_id", "jd_no", score_col]],
        on=["user_id", "jd_no"],
        how="left",
        validate="one_to_one",
    )
    if m[score_col].isna().any():
        raise SystemExit("prediction alignment failed")
    return m[score_col].to_numpy().astype(float)


def main() -> None:
    train = pd.read_csv(TRAIN, dtype={"user_id": str, "jd_no": str})
    test = pd.read_csv(TEST, dtype={"user_id": str, "jd_no": str}).reset_index(drop=True)
    job_hist = train.groupby("jd_no").size().astype(int).to_dict()
    user_hist = train.groupby("user_id").size().astype(int).to_dict()

    b0 = pd.read_csv(B0_PRED, dtype={"user_id": str, "jd_no": str})
    b0_score = align_scores(test, b0, "score")
    p = b0.copy()
    if "job_id" in p.columns and "jd_no" not in p.columns:
        p = p.rename(columns={"job_id": "jd_no"})
    p["user_id"] = p["user_id"].astype(str)
    p["jd_no"] = p["jd_no"].astype(str)
    mlab = test.assign(user_id=test["user_id"].astype(str), jd_no=test["jd_no"].astype(str))[
        ["user_id", "jd_no"]
    ].merge(
        p[["user_id", "jd_no", "label"]],
        on=["user_id", "jd_no"],
        how="left",
        validate="one_to_one",
    )
    y = mlab["label"].to_numpy().astype(int)
    if "satisfied" in test.columns:
        y_test = test["satisfied"].to_numpy().astype(int)
        if not np.array_equal(y, y_test):
            raise SystemExit(f"label mismatch B0 vs test: {int((y != y_test).sum())}")

    jh = np.asarray([int(job_hist.get(str(j), 0)) for j in test["jd_no"]], dtype=np.int64)
    uh = np.asarray([int(user_hist.get(str(u), 0)) for u in test["user_id"]], dtype=np.int64)
    users = test["user_id"].astype(str).to_numpy()

    # ---- STEP 1: cutpoints BEFORE any AUC ----
    # Preserve an already-written cutpoints file (prereg-before-calc evidence).
    if CUTPOINTS.is_file():
        cuts = json.loads(CUTPOINTS.read_text(encoding="utf-8"))
        print("LOADED_EXISTING_CUTPOINTS", cuts["written_utc"])
    else:
        cuts = write_cutpoints(len(test), jh, uh)
        print("WROTE_CUTPOINTS", cuts["written_utc"])
    jc1, jc2 = cuts["job"]["q33"], cuts["job"]["q66"]
    uc1, uc2 = cuts["user"]["q33"], cuts["user"]["q66"]

    base_scores = []
    for seed in BASE_SEEDS:
        df = pd.read_csv(
            E1_RUNS / f"E1_BASE_S{seed}" / "predictions.csv",
            dtype={"user_id": str},
        )
        if "job_id" in df.columns:
            df["job_id"] = df["job_id"].astype(str)
        base_scores.append(align_scores(test, df, "score"))
    base_scores = np.stack(base_scores, axis=0)  # (10, n)

    thirds = ("Low", "Mid", "High")
    point: dict = {}
    for g, hist, c1, c2 in (
        ("J", jh, jc1, jc2),
        ("U", uh, uc1, uc2),
    ):
        for t in thirds:
            m = stratum_mask(hist, c1, c2, t)
            auc_b0 = pool_auc(y[m], b0_score[m])
            if auc_b0 is None:
                raise SystemExit(f"B0 AUC undefined for {g}{t}")
            auc_base_seeds = []
            for s in range(base_scores.shape[0]):
                a = pool_auc(y[m], base_scores[s, m])
                if a is None:
                    raise SystemExit(f"Base AUC undefined for {g}{t} seed {s}")
                auc_base_seeds.append(a)
            auc_base = float(np.mean(auc_base_seeds))
            vals = hist[m]
            point[f"{g}{t}"] = {
                "n": int(m.sum()),
                "range_lo": int(vals.min()),
                "range_hi": int(vals.max()),
                "auc_b0": float(auc_b0),
                "auc_base": auc_base,
                "gap": float(auc_b0 - auc_base),
                "auc_base_seeds": auc_base_seeds,
            }
        point[f"{g}Trend"] = {
            "trend": point[f"{g}High"]["gap"] - point[f"{g}Low"]["gap"],
        }

    # ---- bootstrap ----
    rng = np.random.default_rng(RNG_SEED)
    n_seeds = base_scores.shape[0]
    tmp: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(users):
        tmp[str(u)].append(i)
    u2ix = {u: np.asarray(ix, dtype=np.int64) for u, ix in tmp.items()}
    user_list = np.asarray(sorted(u2ix.keys()), dtype=object)

    boot_keys = [f"{g}{t}" for g in ("J", "U") for t in thirds] + ["JTrend", "UTrend"]
    boot = {k: [] for k in boot_keys}
    skipped = 0

    user_ptr = np.asarray([u2ix[str(u)] for u in user_list], dtype=object)
    n_workers = max(1, min(8, (os_cpu := __import__("os").cpu_count() or 4)))
    chunk = (B + n_workers - 1) // n_workers
    # Independent streams from SeedSequence for reproducibility across workers.
    ss = np.random.SeedSequence(RNG_SEED)
    child_seeds = ss.spawn(n_workers)
    specs = []
    done = 0
    for i, cs in enumerate(child_seeds):
        nrep = min(chunk, B - done)
        if nrep <= 0:
            break
        specs.append((done, nrep, int(cs.generate_state(1)[0])))
        done += nrep
    payload = {
        "y": y,
        "b0_score": b0_score,
        "base_scores": base_scores,
        "jh": jh,
        "uh": uh,
        "user_ptr": user_ptr,
        "jc1": jc1,
        "jc2": jc2,
        "uc1": uc1,
        "uc2": uc2,
        "thirds": thirds,
        "boot_keys": boot_keys,
    }
    print(f"BOOT_PARALLEL workers={n_workers} chunk~={chunk}", flush=True)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(payload,),
    ) as ex:
        for part, sk in ex.map(_boot_chunk, specs):
            skipped += sk
            for k, vals in part.items():
                boot[k].extend(vals)
            print(f"BOOT_CHUNK_DONE n={sum(len(v) for v in part.values())//len(boot_keys)} skipped_chunk={sk}", flush=True)
    # silence unused
    _ = rng
    _ = n_seeds

    def ci95(arr: list[float]) -> tuple[float, float]:
        a = np.asarray(arr, dtype=float)
        return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    lines = [
        "% results_macros_v4e.tex — Addendum e (B0 vs Base by training history)",
        f"% prereg commit: {PREREG_COMMIT}",
        f"% generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"% cutpoints: {CUTPOINTS.as_posix()}",
        f"% bootstrap B={B} seed={RNG_SEED} skipped={skipped}",
    ]
    for g in ("J", "U"):
        for t in thirds:
            p = point[f"{g}{t}"]
            lo, hi = ci95(boot[f"{g}{t}"])
            lines.append(
                rf"\newcommand{{\Hist{g}{t}Range}}{{{latex_range(p['range_lo'], p['range_hi'])}}}"
            )
            lines.append(rf"\newcommand{{\Hist{g}{t}N}}{{{latex_int(p['n'])}}}")
            lines.append(
                rf"\newcommand{{\Hist{g}{t}AUCb}}{{\ensuremath{{{latex_auc4(p['auc_b0'])}}}}}"
            )
            lines.append(
                rf"\newcommand{{\Hist{g}{t}AUCbase}}{{\ensuremath{{{latex_auc4(p['auc_base'])}}}}}"
            )
            lines.append(
                rf"\newcommand{{\Hist{g}{t}Gap}}{{\ensuremath{{{latex_signed4(p['gap'])}}}}}"
            )
            lines.append(
                rf"\newcommand{{\Hist{g}{t}Gaplo}}{{\ensuremath{{{latex_signed4(lo)}}}}}"
            )
            lines.append(
                rf"\newcommand{{\Hist{g}{t}Gaphi}}{{\ensuremath{{{latex_signed4(hi)}}}}}"
            )
        tr = point[f"{g}Trend"]["trend"]
        tlo, thi = ci95(boot[f"{g}Trend"])
        lines.append(rf"\newcommand{{\Hist{g}Trend}}{{\ensuremath{{{latex_signed4(tr)}}}}}")
        lines.append(rf"\newcommand{{\Hist{g}Trendlo}}{{\ensuremath{{{latex_signed4(tlo)}}}}}")
        lines.append(rf"\newcommand{{\Hist{g}Trendhi}}{{\ensuremath{{{latex_signed4(thi)}}}}}")

    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    report = [
        "# Results report v4",
        "",
        "## Addendum e — B0 vs Base by training history",
        "",
        f"- Prereg commit: `{PREREG_COMMIT}`",
        f"- Cutpoints file: `{CUTPOINTS.as_posix()}`",
        f"- Cutpoints written_utc: `{cuts['written_utc']}`",
        f"- Job cutpoints (test-pair quantiles 1/3, 2/3): q33={cuts['job']['q33']}, q66={cuts['job']['q66']}",
        f"- User cutpoints (test-pair quantiles 1/3, 2/3): q33={cuts['user']['q33']}, q66={cuts['user']['q66']}",
        f"- Bootstrap: B={B}, RNG seed={RNG_SEED}",
        f"- Skipped replicates (stratum missing a class): **{skipped}**",
        f"- Successful replicates: **{B - skipped}**",
        "",
        "### Point estimates (6 d.p.)",
        "",
    ]
    for g, name in (("J", "Job"), ("U", "User")):
        report.append(f"#### {name} history")
        for t in thirds:
            p = point[f"{g}{t}"]
            lo, hi = ci95(boot[f"{g}{t}"])
            report.append(
                f"- {t}: n={p['n']} range={p['range_lo']}--{p['range_hi']} "
                f"AUC_B0={p['auc_b0']:.6f} AUC_Base={p['auc_base']:.6f} "
                f"Gap={p['gap']:+.6f} 95%CI=[{lo:+.6f}, {hi:+.6f}]"
            )
        tr = point[f"{g}Trend"]["trend"]
        tlo, thi = ci95(boot[f"{g}Trend"])
        report.append(f"- Trend (High−Low Gap): {tr:+.6f} 95%CI=[{tlo:+.6f}, {thi:+.6f}]")
        report.append("")

    OUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    OUT_JSON.write_text(
        json.dumps(
            {
                "prereg_commit": PREREG_COMMIT,
                "cutpoints": cuts,
                "point": point,
                "skipped": skipped,
                "B": B,
                "ci95": {k: ci95(v) for k, v in boot.items()},
            },
            indent=2,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cutpoints_utc": cuts["written_utc"],
                "skipped": skipped,
                "tex": str(OUT_TEX),
                "JTrend": point["JTrend"]["trend"],
                "UTrend": point["UTrend"]["trend"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

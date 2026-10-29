#!/usr/bin/env python3
"""§6 only: fig_seeds.pdf + fig_curves.pdf (+ 300 dpi PNG). No training."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
E1_ANALYSIS = ROOT / "results/e1/e1_analysis.json"
E1_RUNS = ROOT / "results/e1/runs"

# Base 10-ep S0: original 3 + Gate C supplemental 7
S0_SEED_PATHS = {
    42: ROOT / "results/phase1/phase1c3_final/A0_seed42/metrics.json",
    123: ROOT / "results/phase1/phase1c3_final/A0_seed123/metrics.json",
    456: ROOT / "results/phase1/phase1c3_final/A0_seed456/metrics.json",
    11: ROOT / "results/phase4/phase4a/gate_c/predictions/S0_seed11/metrics.json",
    23: ROOT / "results/phase4/phase4a/gate_c/predictions/S0_seed23/metrics.json",
    37: ROOT / "results/phase4/phase4a/gate_c/predictions/S0_seed37/metrics.json",
    59: ROOT / "results/phase4/phase4a/gate_c/predictions/S0_seed59/metrics.json",
    71: ROOT / "results/phase4/phase4a/gate_c/predictions/S0_seed71/metrics.json",
    89: ROOT / "results/phase4/phase4a/gate_c/predictions/S0_seed89/metrics.json",
    101: ROOT / "results/phase4/phase4a/gate_c/predictions/S0_seed101/metrics.json",
}

CELL_ORDER = ["base", "w", "t", "wt"]
CELL_LABEL = {"base": "Base", "w": "+W", "t": "+T", "wt": "+W+T"}
CELL_COLOR = {
    "base": "#4C78A8",
    "w": "#1b9e77",
    "t": "#d95f02",
    "wt": "#7570b3",
}

B0_AUC = 0.6689
PROBE_AUC = 0.6457


def _style() -> None:
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.titlesize": 8,
        }
    )


def load_s0_ten() -> list[tuple[int, float, Path]]:
    rows = []
    for seed, path in S0_SEED_PATHS.items():
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append((seed, float(m["roc_auc"]), path))
    return rows


def load_e1_table() -> list[dict]:
    d = json.loads(E1_ANALYSIS.read_text(encoding="utf-8"))
    return list(d["run_table"])


def fig_seeds(table: list[dict], s0: list[tuple[int, float, Path]]) -> dict:
    groups = [
        ("Base 10 ep", [auc for _, auc, _ in s0]),
        ("Base", [r["test_auc"] for r in table if r["cell"] == "base"]),
        ("+W", [r["test_auc"] for r in table if r["cell"] == "w"]),
        ("+T", [r["test_auc"] for r in table if r["cell"] == "t"]),
        ("+W+T", [r["test_auc"] for r in table if r["cell"] == "wt"]),
    ]
    for name, vals in groups:
        assert len(vals) == 10, (name, len(vals))

    rng = np.random.default_rng(20261029)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    xs = np.arange(len(groups), dtype=float)
    for i, (name, vals) in enumerate(groups):
        y = np.asarray(vals, dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(y))
        ax.scatter(
            xs[i] + jitter,
            y,
            s=14,
            facecolors="none",
            edgecolors="0.2",
            linewidths=0.7,
            zorder=3,
        )
        mean = float(y.mean())
        ax.plot(
            [xs[i] - 0.18, xs[i] + 0.18],
            [mean, mean],
            color="0.1",
            lw=1.4,
            solid_capstyle="butt",
            zorder=4,
        )

    ax.axhline(B0_AUC, color="0.35", ls="--", lw=0.9, zorder=1)
    ax.axhline(PROBE_AUC, color="0.35", ls=":", lw=0.9, zorder=1)
    ax.set_ylim(0.640, 0.676)
    ax.text(
        0.02,
        B0_AUC + 0.0012,
        "B0 = 0.6689",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.25",
    )
    ax.text(
        0.02,
        PROBE_AUC - 0.0012,
        "linear probe = 0.6457",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        fontsize=7,
        color="0.25",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], rotation=15, ha="right")
    ax.set_ylabel("Test AUC")
    ax.set_xlim(-0.45, len(groups) - 0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.35)

    pdf = FIGDIR / "fig_seeds.pdf"
    png = FIGDIR / "fig_seeds.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {
        "pdf": str(pdf),
        "png": str(png),
        "n_points_per_group": 10,
        "n_groups": 5,
        "n_seeds_base10ep": 10,
        "n_seeds_e1": 10,
        "n_runs_e1_used": 40,
    }


def fig_curves(table: list[dict]) -> dict:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    used_hist = []
    for cell in CELL_ORDER:
        rows = [r for r in table if r["cell"] == cell]
        assert len(rows) == 10, (cell, len(rows))
        for r in rows:
            rid = r["run_id"]
            hp = E1_RUNS / rid / "history.json"
            hist = json.loads(hp.read_text(encoding="utf-8"))["history"]
            # truncate at early-stop (full history already ends at stop)
            epochs = [h["epoch"] for h in hist]
            vals = [h["val_auc"] for h in hist]
            ax.plot(
                epochs,
                vals,
                color=CELL_COLOR[cell],
                lw=0.6,
                alpha=0.6,
                label=CELL_LABEL[cell] if r is rows[0] else None,
            )
            used_hist.append(str(hp))

    ax.axvline(10, color="0.3", ls="--", lw=0.9)
    y0, y1 = ax.get_ylim()
    ax.text(
        10.6,
        y0 + 0.97 * (y1 - y0),
        "10-epoch cap",
        fontsize=7,
        color="0.25",
        rotation=90,
        va="top",
        ha="left",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.legend(
        loc="lower right",
        frameon=False,
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
        borderpad=0.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.35)

    pdf = FIGDIR / "fig_curves.pdf"
    png = FIGDIR / "fig_curves.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return {
        "pdf": str(pdf),
        "png": str(png),
        "n_runs": 40,
        "n_seeds": 10,
        "n_cells": 4,
        "history_files": used_hist,
    }


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    _style()
    s0 = load_s0_ten()
    table = load_e1_table()
    assert len(table) == 40

    meta_seeds = fig_seeds(table, s0)
    meta_curves = fig_curves(table)

    # re-fix curves label placement: redraw with cleaner text placement
    # (already saved; optional polish already in fig_curves)

    report = {
        "fig_seeds": {
            **{k: v for k, v in meta_seeds.items() if k != "history_files"},
            "data_files": {
                "e1_run_table": str(E1_ANALYSIS),
                "s0_10ep_metrics": [str(p) for _, _, p in s0],
            },
            "s0_seeds": [s for s, _, _ in s0],
            "e1_seeds": sorted({r["seed"] for r in table}),
        },
        "fig_curves": {
            "pdf": meta_curves["pdf"],
            "png": meta_curves["png"],
            "n_runs": meta_curves["n_runs"],
            "n_seeds": meta_curves["n_seeds"],
            "data_files": {
                "e1_run_table": str(E1_ANALYSIS),
                "history_json": meta_curves["history_files"],
            },
        },
    }
    out = FIGDIR / "fig_seeds_curves_data_manifest.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

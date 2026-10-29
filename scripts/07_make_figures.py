#!/usr/bin/env python3
"""Render fig_effects / fig_protocol / fig_dose from results/effects_summary.json (PDF vectors)."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).resolve().parents[1]
    effects = json.loads((root / "results/effects_summary.json").read_text(encoding="utf-8"))
    figdir = root / "figures"
    figdir.mkdir(exist_ok=True)
    mpl.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8})
    # dose panel
    fig, axes = plt.subplots(1, 2, figsize=(3.7, 1.9), sharey=True)
    ks = [1, 2, 4]
    off = 0.08
    leng = effects.get("v3", {}).get("length_only")
    series = [
        (
            "composite $q$",
            {
                1: effects["DR"]["kappa_1"],
                2: effects["DR"]["kappa_2"],
                4: effects["DR"]["kappa_4"],
            },
            "o",
            True,
            -off,
        )
    ]
    if leng:
        series.append(
            (
                "length-only",
                {1: leng["kappa_1"], 2: leng["kappa_2"], 4: leng["kappa_4"]},
                "s",
                False,
                +off,
            )
        )
    for label, dd, mk, fill, dx in series:
        x = [k + dx for k in ks]
        y = [dd[k]["est"] * 1e3 for k in ks]
        lo = [(dd[k]["est"] - dd[k]["lo"]) * 1e3 for k in ks]
        hi = [(dd[k]["hi"] - dd[k]["est"]) * 1e3 for k in ks]
        axes[0].errorbar(
            x,
            y,
            yerr=[lo, hi],
            fmt="-",
            color="black",
            lw=0.9,
            capsize=2.5,
            markersize=3.5,
            marker=mk,
            mfc=("black" if fill else "white"),
            mec="black",
            label=label,
        )
    axes[0].legend(loc="lower left", fontsize=6, frameon=False)
    axes[0].axhline(0, color="0.5", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\kappa$"); axes[0].set_ylabel(r"$\Delta$AUC ($\times 10^{-3}$)")
    lams = [0.05, 0.5, 2.0]
    keys = ["lambda_0.05", "lambda_0.5", "lambda_2.0"]
    ys = [effects["DR"][k]["est"] * 1e3 for k in keys]
    lo = [(effects["DR"][k]["est"] - effects["DR"][k]["lo"]) * 1e3 for k in keys]
    hi = [(effects["DR"][k]["hi"] - effects["DR"][k]["est"]) * 1e3 for k in keys]
    axes[1].errorbar(lams, ys, yerr=[lo, hi], fmt="o-", color="black", markersize=3.5, capsize=2.5)
    axes[1].set_xscale("log"); axes[1].set_xticks(lams)
    axes[1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axes[1].axhline(0, color="0.5", ls="--", lw=0.8)
    axes[1].set_xlabel(r"$\lambda$")
    fig.tight_layout(pad=0.3)
    fig.savefig(figdir / "fig_dose.pdf")
    plt.close(fig)
    print("wrote", figdir / "fig_dose.pdf")
    print("Note: fig_effects/fig_protocol require full factorial panels; see paper_v2 figures.")
    # Optional: fig_seeds / fig_curves (need E1 histories + S0 metrics under repo results/)
    try:
        from scripts import make_fig_seeds_curves as _sfc  # type: ignore
    except Exception:
        _sfc = None
    if _sfc is None:
        # packaged script is flat under scripts/
        import runpy
        sfc = root / 'scripts' / 'make_fig_seeds_curves.py'
        if sfc.is_file() and (root / 'results' / 'e1' / 'e1_analysis.json').is_file():
            runpy.run_path(str(sfc), run_name='__main__')
        else:
            print('skip fig_seeds/fig_curves (E1 histories not in this package; PDFs shipped under figures/)')


if __name__ == "__main__":
    main()

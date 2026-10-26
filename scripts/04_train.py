#!/usr/bin/env python3
"""Thin CLI wrapper around src.phase1.train_entry for paper cells."""
from __future__ import annotations
import argparse
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["base", "w", "t", "wt", "wk2", "wk4", "tl05", "tl2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-root", default="results/runs")
    args, extra = ap.parse_known_args()
    cmd = [sys.executable, "-m", "src.phase1.train_entry", "--seed", str(args.seed),
           "--max-epochs", str(args.max_epochs), "--early-stopping-patience", str(args.patience),
           "--device", args.device, "--output-root", args.output_root]
    # Map cell → loss flags (mirrors paper matrix)
    if args.cell in ("w", "wt", "wk2", "wk4"):
        cmd += ["--experiment", "A2" if args.cell == "w" else "A4"]
    if args.cell in ("t", "wt", "tl05", "tl2"):
        lam = {"t": 0.05, "wt": 0.05, "tl05": 0.5, "tl2": 2.0}[args.cell]
        cmd += ["--lambda-contrastive", str(lam)]
    if args.cell.startswith("wk"):
        cmd += ["--reverse-weight-kappa", args.cell[2:]]
    cmd += extra
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()

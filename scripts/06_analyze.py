#!/usr/bin/env python3
"""Aggregate run folders into effects (expects predictions.csv per run)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.bootstrap import rank_auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("results/runs/e1"))
    ap.add_argument("--out", type=Path, default=Path("results/effects_local.json"))
    args = ap.parse_args()
    rows = []
    for d in sorted(args.runs.glob("*")):
        pred = d / "predictions.csv"
        if not pred.is_file():
            continue
        df = pd.read_csv(pred)
        rows.append({"run": d.name, "auc": float(rank_auc(df["label"].to_numpy(), df["score"].to_numpy()))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote", args.out, "n=", len(rows))

if __name__ == "__main__":
    main()

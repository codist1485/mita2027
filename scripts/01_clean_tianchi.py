#!/usr/bin/env python3
"""Clean Tianchi tables into a pair table used by training.

Expects provider CSVs under data/raw/tianchi/. Writes data/processed/pairs.parquet
with columns: user_id, jd_no, label, user_text, job_text (and optional action flags).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/tianchi"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/pairs.parquet"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Minimal cleaner: concatenate user/job text columns if present.
    # Adapt column names to the provider schema as needed.
    users = pd.read_csv(args.raw_dir / "user.csv") if (args.raw_dir / "user.csv").is_file() else None
    jobs = pd.read_csv(args.raw_dir / "job.csv") if (args.raw_dir / "job.csv").is_file() else None
    inter = pd.read_csv(args.raw_dir / "interaction.csv") if (args.raw_dir / "interaction.csv").is_file() else None
    if inter is None:
        raise SystemExit("Missing data/raw/tianchi/interaction.csv (obtain from Tianchi; not shipped).")
    # placeholder merge — replace with the project cleaning rules used in the paper
    df = inter.copy()
    if "satisfied" in df.columns:
        df["label"] = (df["satisfied"].astype(float) > 0).astype(int)
    df.to_parquet(args.out, index=False)
    print("wrote", args.out, "n=", len(df))

if __name__ == "__main__":
    main()

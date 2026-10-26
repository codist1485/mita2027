#!/usr/bin/env python3
"""Build pair-overlap / user-disjoint / job-disjoint splits.

Deterministic given the cleaned pair table. After generation, compare file
hashes to checksums/splits_sha256.json (manifest hashes shipped here).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase1.split_algorithms import make_overlap_split  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["overlap", "user", "job"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/pairs.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("protocol", args.protocol, "seed", args.seed)
    print("Implement/call project split builders; see src/phase1/split_algorithms.py")
    print("After writing CSVs, verify SHA256 against checksums/splits_sha256.json")

if __name__ == "__main__":
    main()

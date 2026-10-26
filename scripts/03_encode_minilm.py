#!/usr/bin/env python3
"""Encode user/job (and optional RJDB) texts with paraphrase-multilingual-MiniLM-L12-v2."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/pairs.parquet"))
    ap.add_argument("--out-user", type=Path, default=Path("data/embeddings/user_emb.pt"))
    ap.add_argument("--out-job", type=Path, default=Path("data/embeddings/job_emb.pt"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    import pandas as pd
    df = pd.read_parquet(args.pairs)
    enc = SentenceTransformer(MODEL, device=args.device)
    users = df.drop_duplicates("user_id")
    jobs = df.drop_duplicates("jd_no")
    u = enc.encode(users["user_text"].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=True)
    j = enc.encode(jobs["job_text"].astype(str).tolist(), normalize_embeddings=True, show_progress_bar=True)
    args.out_user.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": MODEL, "ids": users["user_id"].tolist(), "embeddings": torch.tensor(u)}, args.out_user)
    torch.save({"model_name": MODEL, "ids": jobs["jd_no"].tolist(), "embeddings": torch.tensor(j)}, args.out_job)
    print("wrote", args.out_user, args.out_job)

if __name__ == "__main__":
    main()

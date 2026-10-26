#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data (no Tianchi/RJDB required)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

class TinyHead(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * 2, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, u, j):
        return self.net(torch.cat([u, j], dim=-1)).squeeze(-1)

def main():
    rng = np.random.default_rng(0)
    n_users, n_jobs, n_pairs = 50, 80, 1000
    users = np.arange(n_users)
    jobs = np.arange(n_jobs)
    u = rng.choice(users, size=n_pairs)
    j = rng.choice(jobs, size=n_pairs)
    # label depends on shared latent
    latent_u = rng.normal(size=(n_users, 8))
    latent_j = rng.normal(size=(n_jobs, 8))
    score = (latent_u[u] * latent_j[j]).sum(1)
    y = (score + rng.normal(scale=0.5, size=n_pairs) > 0).astype(np.int64)
    df = pd.DataFrame({"user_id": u, "jd_no": j, "label": y})
    # split
    idx = rng.permutation(n_pairs)
    train, val, test = idx[:700], idx[700:850], idx[850:]
    # random "embeddings"
    d = 32
    U = torch.tensor(rng.normal(size=(n_users, d)), dtype=torch.float32)
    J = torch.tensor(rng.normal(size=(n_jobs, d)), dtype=torch.float32)
    model = TinyHead(d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(1):
        for start in range(0, len(train), 64):
            batch = train[start:start+64]
            uu = U[df.user_id.to_numpy()[batch]]
            jj = J[df.jd_no.to_numpy()[batch]]
            yy = torch.tensor(df.label.to_numpy()[batch], dtype=torch.float32)
            loss = bce(model(uu, jj), yy)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(U[df.user_id.to_numpy()[test]], J[df.jd_no.to_numpy()[test]]).numpy()
    auc = roc_auc_score(df.label.to_numpy()[test], logits)
    out = {"n_pairs": n_pairs, "test_auc": float(auc), "status": "ok"}
    print(json.dumps(out, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "smoke_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("SMOKE_OK")

if __name__ == "__main__":
    main()

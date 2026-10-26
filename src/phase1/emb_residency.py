"""GPU-resident MiniLM embedding placement helpers (engineering candidate).

Frozen science protocol unchanged: same emb bytes, same indices, same losses.
When ``resident=False`` (default), behavior matches historical CPU-emb + per-batch H2D.
When ``resident=True``, full emb tables live on ``device`` and gathers run on-device.
"""

from __future__ import annotations

import os

import torch


def gpu_resident_enabled(cli_flag: bool | None = None) -> bool:
    """Resolve residency mode: explicit CLI wins; else env MITA_GPU_RESIDENT_EMB=1."""
    if cli_flag is not None:
        return bool(cli_flag)
    return os.environ.get("MITA_GPU_RESIDENT_EMB", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}


def place_frozen_embeddings(
    user_emb: torch.Tensor,
    job_emb: torch.Tensor,
    device: torch.device | str,
    *,
    resident: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place frozen emb tables. Never sets requires_grad."""
    if user_emb.requires_grad or job_emb.requires_grad:
        raise RuntimeError("embeddings must remain non-trainable")
    user_emb = user_emb.detach()
    job_emb = job_emb.detach()
    if resident:
        user_emb = user_emb.to(device, non_blocking=False)
        job_emb = job_emb.to(device, non_blocking=False)
    user_emb.requires_grad_(False)
    job_emb.requires_grad_(False)
    return user_emb, job_emb


def gather_pair_embeddings(
    user_emb: torch.Tensor,
    job_emb: torch.Tensor,
    user_index: torch.Tensor,
    job_index: torch.Tensor,
    device: torch.device | str,
    *,
    resident: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather user/job rows for a batch onto the compute device."""
    if resident:
        ui = user_index.to(device=user_emb.device, dtype=torch.long)
        ji = job_index.to(device=job_emb.device, dtype=torch.long)
        return user_emb[ui], job_emb[ji]
    return user_emb[user_index].to(device), job_emb[job_index].to(device)


def place_rjdb_embeddings(
    rjdb: dict,
    device: torch.device | str,
    *,
    resident: bool,
) -> dict:
    """Place frozen RJDB triplet emb tables. Keys: job, matched, unmatched.

    When ``resident=True``, tensors live on ``device`` and batch assembly is index-only.
    Does not change sampling RNG: callers still draw CPU/CUDA-generator integer indices.
    """
    out = dict(rjdb)
    for key in ("job", "matched", "unmatched"):
        if key not in rjdb:
            continue
        t = rjdb[key]
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        t = t.detach().float()
        if resident:
            t = t.to(device, non_blocking=False)
        t.requires_grad_(False)
        out[key] = t
    return out


def gather_rjdb_triplet(
    rjdb: dict,
    choice: torch.Tensor,
    device: torch.device | str,
    *,
    resident: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (matched, unmatched, job) rows for ``choice`` indices on compute device."""
    if resident:
        idx = choice.to(device=rjdb["job"].device, dtype=torch.long)
        return rjdb["matched"][idx], rjdb["unmatched"][idx], rjdb["job"][idx]
    idx = choice.long()
    return (
        rjdb["matched"][idx].to(device),
        rjdb["unmatched"][idx].to(device),
        rjdb["job"][idx].to(device),
    )

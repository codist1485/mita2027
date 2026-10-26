"""Cross-platform path resolution for Phase 1."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PHASE1 = ROOT / "data" / "phase1"
DATASET_DIR = ROOT / "dataset"
RESULTS_PHASE1 = ROOT / "results" / "phase1"
EMBED_DIR = ROOT / "data" / "embeddings"

ACTION_PATH = DATA_PROCESSED / "action_cleaned.csv"
JOB_V2_PATH = DATA_PROCESSED / "job_cleaned_v2.csv"
USER_PATH = DATA_PROCESSED / "user_cleaned.csv"
RANK_JOB_PATH = DATASET_DIR / "rank_job.csv"
RANK_RESUME_PATH = DATASET_DIR / "rank_resume.csv"

SPLIT_DIR = DATA_PHASE1 / "splits"
QUALITY_DIR = DATA_PHASE1 / "quality"
MANIFEST_DIR = RESULTS_PHASE1 / "manifests"
RUNS_DIR = RESULTS_PHASE1 / "runs"

USER_EMB_PATH = EMBED_DIR / "user_emb.pt"
JOB_EMB_PATH = EMBED_DIR / "job_emb.pt"
RJDB_EMB_PATH = EMBED_DIR / "rjdb_train_emb.pt"


def rjdb_train_jsonl() -> Path:
    """RJDB train.jsonl from env or default Windows research path."""
    override = os.environ.get("RJDB_TRAIN_JSONL") or os.environ.get("MITA_RJDB_TRAIN")
    if override:
        return Path(override)
    return Path(r"D:\work\phd\dataset\RJDB\train.jsonl")


def ensure_phase1_dirs() -> None:
    for path in (SPLIT_DIR, QUALITY_DIR, MANIFEST_DIR, RUNS_DIR, RESULTS_PHASE1 / "logs"):
        path.mkdir(parents=True, exist_ok=True)

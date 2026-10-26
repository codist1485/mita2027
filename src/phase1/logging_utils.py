"""Run logging for Phase 1."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.phase1.paths import RUNS_DIR, ensure_phase1_dirs


def git_commit(root: Path) -> str | None:
    git = root / ".git"
    if not git.exists():
        return None
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env_versions() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda"] = torch.version.cuda
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import sentence_transformers

        info["sentence_transformers"] = sentence_transformers.__version__
    except Exception as exc:
        info["sentence_transformers_error"] = str(exc)
    return info


class RunLogger:
    def __init__(
        self,
        config_name: str,
        seed: int,
        root: Path,
        *,
        run_id: str | None = None,
        output_dir: Path | None = None,
    ) -> None:
        ensure_phase1_dirs()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if output_dir is not None:
            # TCORE isolated run directory (already created + locked by tcore_binding).
            if run_id is None:
                raise ValueError("run_id is required when output_dir is set")
            self.run_id = run_id
            self.dir = Path(output_dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "logs").mkdir(exist_ok=True)
            (self.dir / "checkpoints").mkdir(exist_ok=True)
        else:
            self.run_id = f"{config_name}_seed{seed}_{stamp}_{uuid.uuid4().hex[:8]}"
            self.dir = RUNS_DIR / self.run_id
            self.dir.mkdir(parents=True, exist_ok=False)
        self.history: list[dict[str, Any]] = []
        self.started = time.time()
        self.meta: dict[str, Any] = {
            "run_id": self.run_id,
            "config_name": config_name,
            "seed": seed,
            "git_commit": git_commit(root),
            "created_utc": stamp,
            "env": env_versions(),
            "output_dir": str(self.dir),
        }
        self._write("meta.json", self.meta)

    def _write(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def log_config(self, config: dict[str, Any], config_path: Path | None = None) -> None:
        payload = dict(config)
        if config_path is not None:
            payload["config_path"] = str(config_path)
            payload["config_sha256"] = file_sha256(config_path)
        self._write("config.json", payload)

    def log_epoch(self, row: dict[str, Any]) -> None:
        self.history.append(row)
        self._write("history.json", {"history": self.history})

    def finalize(self, summary: dict[str, Any], error: str | None = None) -> Path:
        summary = {
            **self.meta,
            **summary,
            "elapsed_sec": time.time() - self.started,
            "error": error,
            "history": self.history,
        }
        return self._write("summary.json", summary)

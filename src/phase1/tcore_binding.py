"""TCORE B_FULL split binding, FAIL_CLOSED validation, and run output isolation.

No model imports. Safe to unit-test without initializing encoders or training.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_PROTOCOLS = {
    "SPLIT_PROTOCOL_OVERLAP_V1",
    "SPLIT_PROTOCOL_USER_DISJOINT_V2",
    "SPLIT_PROTOCOL_JOB_DISJOINT_V2",
    "SPLIT_PROTOCOL_USER_DISJOINT_E2",
    "SPLIT_PROTOCOL_JOB_DISJOINT_E2",
}


class TcoreBindingError(SystemExit):
    """Non-zero exit for FAIL_CLOSED binding failures (before model init)."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise TcoreBindingError(
            f"FAIL_CLOSED: invalid --run-id {run_id!r}; "
            "allowed pattern ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
        )
    if ".." in run_id or "/" in run_id or "\\" in run_id:
        raise TcoreBindingError(f"FAIL_CLOSED: path traversal rejected in run_id {run_id!r}")
    return run_id


def resolve_under_root(root: Path, path: Path) -> Path:
    """Resolve path and ensure it stays under root (no escape)."""
    root_r = root.resolve()
    cand = path if path.is_absolute() else (root / path)
    resolved = cand.resolve()
    try:
        resolved.relative_to(root_r)
    except ValueError as exc:
        raise TcoreBindingError(
            f"FAIL_CLOSED: path escapes project root: {path} -> {resolved}"
        ) from exc
    return resolved


@dataclass(frozen=True)
class ResolvedSplitBinding:
    mode: str  # TCORE | LEGACY_OVERLAP_ONLY
    run_id: str | None
    split_protocol: str | None
    split_dir: Path | None
    train_path: Path
    val_path: Path
    train_sha256: str
    val_sha256: str
    split_manifest_path: Path | None
    split_manifest_sha256: str | None
    output_dir: Path | None
    max_epochs: int
    early_stopping_patience: int

    def to_manifest(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("split_dir", "train_path", "val_path", "split_manifest_path", "output_dir"):
            if d[key] is not None:
                d[key] = str(d[key])
        return d


def load_split_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise TcoreBindingError(f"FAIL_CLOSED: missing split manifest {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TcoreBindingError(f"FAIL_CLOSED: invalid JSON split manifest {path}") from exc
    return data


def _require_pair_columns(path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(path, dtype=str, nrows=5, keep_default_na=False)
    needed = {"user_id", "jd_no", "satisfied"}
    missing = needed - set(frame.columns)
    if missing:
        raise TcoreBindingError(f"FAIL_CLOSED: {path} missing columns {sorted(missing)}")


def verify_split_dir(
    *,
    project_root: Path,
    split_dir: Path,
    split_protocol: str,
    expected_train_sha: str,
    expected_val_sha: str,
    require_manifest: bool = True,
    split_manifest: Path | None = None,
) -> tuple[Path, Path, Path | None, str | None]:
    if split_protocol not in SAFE_PROTOCOLS:
        raise TcoreBindingError(f"FAIL_CLOSED: unknown split_protocol {split_protocol!r}")

    split_dir = resolve_under_root(project_root, split_dir)
    if not split_dir.is_dir():
        raise TcoreBindingError(f"FAIL_CLOSED: split-dir does not exist: {split_dir}")

    train_path = split_dir / "train_pairs.csv"
    val_path = split_dir / "val_pairs.csv"
    if not train_path.is_file():
        raise TcoreBindingError(f"FAIL_CLOSED: missing train_pairs.csv under {split_dir}")
    if not val_path.is_file():
        raise TcoreBindingError(f"FAIL_CLOSED: missing val_pairs.csv under {split_dir}")

    # Symlink / unexpected escape check on leaf files
    for p in (train_path, val_path):
        resolve_under_root(project_root, p)

    train_sha = file_sha256(train_path)
    val_sha = file_sha256(val_path)
    if train_sha.lower() != expected_train_sha.lower():
        raise TcoreBindingError(
            f"FAIL_CLOSED: train SHA mismatch\n  expected {expected_train_sha}\n  actual   {train_sha}"
        )
    if val_sha.lower() != expected_val_sha.lower():
        raise TcoreBindingError(
            f"FAIL_CLOSED: val SHA mismatch\n  expected {expected_val_sha}\n  actual   {val_sha}"
        )

    _require_pair_columns(train_path)
    _require_pair_columns(val_path)

    manifest_path: Path | None
    if split_manifest is not None:
        manifest_path = resolve_under_root(project_root, Path(split_manifest))
    else:
        manifest_path = split_dir / "split_manifest.json"
    manifest_sha = None
    if require_manifest:
        if not manifest_path.is_file():
            raise TcoreBindingError(f"FAIL_CLOSED: missing split manifest {manifest_path}")
        meta = load_split_manifest(manifest_path)
        proto = meta.get("split_protocol")
        if proto != split_protocol:
            raise TcoreBindingError(
                f"FAIL_CLOSED: split_protocol mismatch: CLI={split_protocol!r} manifest={proto!r}"
            )
        m_train = meta.get("train_sha256")
        m_val = meta.get("val_sha256")
        if m_train and m_train.lower() != train_sha.lower():
            raise TcoreBindingError("FAIL_CLOSED: manifest train_sha256 != file bytes")
        if m_val and m_val.lower() != val_sha.lower():
            raise TcoreBindingError("FAIL_CLOSED: manifest val_sha256 != file bytes")
        if m_train and m_train.lower() != expected_train_sha.lower():
            raise TcoreBindingError("FAIL_CLOSED: expected train sha != manifest train_sha256")
        if m_val and m_val.lower() != expected_val_sha.lower():
            raise TcoreBindingError("FAIL_CLOSED: expected val sha != manifest val_sha256")
        manifest_sha = file_sha256(manifest_path)
    else:
        manifest_path = None

    return train_path, val_path, manifest_path, manifest_sha


def acquire_run_dir(output_root: Path, run_id: str, config_fingerprint: str) -> Path:
    """Create or reopen run directory with exclusive lock + config consistency."""
    validate_run_id(run_id)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = (output_root / run_id).resolve()
    try:
        run_dir.relative_to(output_root.resolve())
    except ValueError as exc:
        raise TcoreBindingError(f"FAIL_CLOSED: run_dir escapes output_root: {run_dir}") from exc

    lock_path = run_dir / ".run.lock"
    fingerprint_path = run_dir / "config_fingerprint.txt"

    if run_dir.exists():
        if fingerprint_path.is_file():
            old = fingerprint_path.read_text(encoding="utf-8").strip()
            if old != config_fingerprint:
                raise TcoreBindingError(
                    f"FAIL_CLOSED: run_id {run_id} exists with different config fingerprint\n"
                    f"  existing {old}\n  new      {config_fingerprint}"
                )
        if lock_path.is_file():
            raise TcoreBindingError(
                f"FAIL_CLOSED: run_id {run_id} appears locked ({lock_path}); "
                "another process may be writing"
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    # Exclusive create lock
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
    except FileExistsError as exc:
        raise TcoreBindingError(
            f"FAIL_CLOSED: could not acquire lock for run_id {run_id}"
        ) from exc

    fingerprint_path.write_text(config_fingerprint + "\n", encoding="utf-8")
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    return run_dir


def release_run_lock(run_dir: Path) -> None:
    lock_path = run_dir / ".run.lock"
    if lock_path.is_file():
        lock_path.unlink()


def config_fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_binding(
    *,
    project_root: Path,
    tcore_mode: bool,
    run_id: str | None,
    split_protocol: str | None,
    split_dir: str | None,
    expected_train_sha: str | None,
    expected_val_sha: str | None,
    output_root: str | None,
    max_epochs: int,
    early_stopping_patience: int,
    default_split_dir: Path,
    split_manifest: str | None = None,
) -> ResolvedSplitBinding:
    """Resolve split + output binding. Raises TcoreBindingError before any model work."""
    if tcore_mode:
        if not run_id or not split_protocol or not split_dir:
            raise TcoreBindingError(
                "FAIL_CLOSED: TCORE mode requires --run-id, --split-protocol, and --split-dir"
            )
        if not expected_train_sha or not expected_val_sha:
            raise TcoreBindingError(
                "FAIL_CLOSED: TCORE mode requires --expected-train-sha and --expected-val-sha"
            )
        train_path, val_path, manifest_path, manifest_sha = verify_split_dir(
            project_root=project_root,
            split_dir=Path(split_dir),
            split_protocol=split_protocol,
            expected_train_sha=expected_train_sha,
            expected_val_sha=expected_val_sha,
            require_manifest=True,
            split_manifest=Path(split_manifest) if split_manifest else None,
        )
        out_root = Path(output_root) if output_root else (project_root / "results/phase6/bfull_tcore_runs")
        out_root = resolve_under_root(project_root, out_root)
        fp = config_fingerprint(
            {
                "run_id": run_id,
                "split_protocol": split_protocol,
                "split_dir": str(train_path.parent),
                "train_sha256": file_sha256(train_path),
                "val_sha256": file_sha256(val_path),
                "max_epochs": max_epochs,
                "patience": early_stopping_patience,
            }
        )
        run_dir = acquire_run_dir(out_root, run_id, fp)
        return ResolvedSplitBinding(
            mode="TCORE",
            run_id=run_id,
            split_protocol=split_protocol,
            split_dir=train_path.parent,
            train_path=train_path,
            val_path=val_path,
            train_sha256=file_sha256(train_path),
            val_sha256=file_sha256(val_path),
            split_manifest_path=manifest_path,
            split_manifest_sha256=manifest_sha,
            output_dir=run_dir,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
        )

    # Legacy: any new TCORE-only flags without --tcore-mode are rejected to avoid silent half-config
    if run_id or split_protocol or split_dir or expected_train_sha or expected_val_sha:
        raise TcoreBindingError(
            "FAIL_CLOSED: split/run binding flags require --tcore-mode "
            "(legacy entry is LEGACY_OVERLAP_ONLY and must not be used by B_FULL launcher)"
        )

    train_path = default_split_dir / "train_pairs.csv"
    val_path = default_split_dir / "val_pairs.csv"
    if not train_path.is_file() or not val_path.is_file():
        raise TcoreBindingError("FAIL_CLOSED: legacy Overlap split files missing")
    return ResolvedSplitBinding(
        mode="LEGACY_OVERLAP_ONLY",
        run_id=None,
        split_protocol="SPLIT_PROTOCOL_OVERLAP_V1",
        split_dir=default_split_dir,
        train_path=train_path,
        val_path=val_path,
        train_sha256=file_sha256(train_path),
        val_sha256=file_sha256(val_path),
        split_manifest_path=None,
        split_manifest_sha256=None,
        output_dir=None,
        max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience,
    )

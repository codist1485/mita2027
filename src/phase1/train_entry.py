"""Unified Phase 1 training / evaluation entry.

Official rank evaluation is disabled unless ``--allow-official-rank-eval`` is set.
That flag never participates in checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.quality_aware_dual_encoder import (
    QualityAwareDualEncoder,
    assert_kappa_one_matches_legacy_reverse,
)
from src.phase1.emb_residency import (
    gather_pair_embeddings,
    gather_rjdb_triplet,
    gpu_resident_enabled,
    place_frozen_embeddings,
    place_rjdb_embeddings,
)
from src.phase1.logging_utils import RunLogger, file_sha256
from src.phase1.losses import contrastive_term, describe_weight_normalization, supervised_loss
from src.phase1.metrics import binary_ap, binary_auc, observed_ranking_diagnostics
from src.phase1.paths import (
    JOB_EMB_PATH,
    RJDB_EMB_PATH,
    SPLIT_DIR,
    USER_EMB_PATH,
    ensure_phase1_dirs,
    rjdb_train_jsonl,
)
from src.phase1.quality_train_only import load_trainfit_quality_map
from src.phase1.similarity_baselines import EQUIVALENCE_NOTE, CosineSimilarityScorer, DotProductScorer
from src.phase1.split import load_split_pairs, load_split_pairs_from_dir
from src.phase1.tcore_binding import release_run_lock, resolve_binding

MATRIX_JSON = ROOT / "configs" / "phase1" / "matrix.json"


class PairDataset(Dataset):
    def __init__(self, user_index, job_index, labels, quality):
        self.user_index = torch.as_tensor(user_index, dtype=torch.long)
        self.job_index = torch.as_tensor(job_index, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.quality = torch.as_tensor(quality, dtype=torch.float32)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, index):
        return self.user_index[index], self.job_index[index], self.labels[index], self.quality[index]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch. CUDA seeding is best-effort."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Prefer deterministic algorithms when available; some CUDA kernels remain nondeterministic.
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_matrix() -> dict:
    return json.loads(MATRIX_JSON.read_text(encoding="utf-8"))


def build_index_tables(users, jobs):
    user_row = {user_id: index for index, user_id in enumerate(users)}
    job_row = {job_id: index for index, job_id in enumerate(jobs)}
    return user_row, job_row


def pairs_to_tensors(frame, user_row, job_row, quality_map):
    users = np.array([user_row[user_id] for user_id in frame["user_id"]], dtype=np.int64)
    jobs = np.array([job_row[job_id] for job_id in frame["jd_no"]], dtype=np.int64)
    labels = frame["satisfied"].to_numpy(dtype=np.float32)
    if set(np.unique(labels).tolist()) - {0.0, 1.0}:
        raise RuntimeError("non-binary labels in split")
    quality = np.array([quality_map.get(job_id, 0.5) for job_id in frame["jd_no"]], dtype=np.float32)
    return users, jobs, labels, quality


@torch.no_grad()
def predict(
    model,
    user_emb,
    job_emb,
    user_index,
    job_index,
    device,
    batch_size: int,
    *,
    resident: bool = False,
    oneshot: bool | None = None,
) -> np.ndarray:
    """Score pairs. When ``resident`` and ``oneshot`` (default True if resident), one GPU gather+forward."""
    model.eval()
    user_tensor = torch.as_tensor(user_index, dtype=torch.long)
    job_tensor = torch.as_tensor(job_index, dtype=torch.long)
    use_oneshot = bool(resident) if oneshot is None else bool(oneshot)
    if use_oneshot and len(user_index) > 0:
        u_batch, j_batch = gather_pair_embeddings(
            user_emb, job_emb, user_tensor, job_tensor, device, resident=resident
        )
        logits = model(u_batch, j_batch)
        return logits.detach().cpu().numpy()
    outputs = []
    for start in range(0, len(user_index), batch_size):
        stop = start + batch_size
        u_batch, j_batch = gather_pair_embeddings(
            user_emb,
            job_emb,
            user_tensor[start:stop],
            job_tensor[start:stop],
            device,
            resident=resident,
        )
        logits = model(u_batch, j_batch)
        outputs.append(logits.detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def resolve_early_stopping_patience(matrix_patience: int, cli_override: int | None) -> int:
    """Effective patience: matrix default unless a positive CLI override is supplied."""
    if cli_override is None:
        return int(matrix_patience)
    if not isinstance(cli_override, int) or isinstance(cli_override, bool):
        raise ValueError("early_stopping_patience override must be a positive int")
    if cli_override < 1:
        raise ValueError(f"early_stopping_patience must be >= 1, got {cli_override}")
    return int(cli_override)


def early_stop_update(
    val_auc: float | None,
    best_val_auc: float,
    bad_rounds: int,
    patience: int,
) -> tuple[float, int, bool, bool]:
    """Pure early-stop / best-AUC step (checkpoint criterion = val_auc only).

    Returns (best_val_auc, bad_rounds, improved, should_stop).
    """
    if val_auc is not None and val_auc > best_val_auc:
        return float(val_auc), 0, True, False
    bad = bad_rounds + 1
    return best_val_auc, bad, False, bad >= patience


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=sorted(load_matrix()["experiments"]))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="optional positive int override; default keeps matrix early_stopping_patience",
    )
    parser.add_argument(
        "--lambda-contrastive",
        type=float,
        default=None,
        help="override matrix lambda for A4/A5 only; leaves configs/phase1/matrix.json unchanged",
    )
    parser.add_argument(
        "--reverse-weight-kappa",
        type=float,
        default=None,
        help="DR dose: for reverse_quality_bce use clip(1+κ(0.5-q),0.1,3); κ=1 ≡ 1.5-q",
    )
    parser.add_argument("--dry-run", action="store_true", help="one forward/backward batch then exit")
    parser.add_argument(
        "--allow-official-rank-eval",
        action="store_true",
        help="optional post-hoc official rank scoring; never used for selection",
    )
    parser.add_argument("--eval-internal-test", action="store_true")
    parser.add_argument("--ranking-diagnostics", action="store_true")
    # --- TCORE B_FULL binding (FAIL_CLOSED; no silent Overlap fallback) ---
    parser.add_argument(
        "--tcore-mode",
        action="store_true",
        help="require explicit split binding; B_FULL launcher must set this",
    )
    parser.add_argument("--run-id", type=str, default=None, help="TCORE unique run id / output folder name")
    parser.add_argument(
        "--split-protocol",
        type=str,
        default=None,
        help="e.g. SPLIT_PROTOCOL_OVERLAP_V1 | USER_DISJOINT_V2 | JOB_DISJOINT_V2",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="directory containing train_pairs.csv, val_pairs.csv, split_manifest.json",
    )
    parser.add_argument("--expected-train-sha", type=str, default=None)
    parser.add_argument("--expected-val-sha", type=str, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="default results/phase6/bfull_tcore_runs under project root",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="optional path to split_manifest.json (default: <split-dir>/split_manifest.json)",
    )
    parser.add_argument(
        "--gpu-resident-emb",
        action="store_true",
        help="candidate: keep frozen MiniLM emb tables on device (engineering; not frozen V1 default)",
    )
    parser.add_argument(
        "--quality-csv",
        type=str,
        default=None,
        help="optional path to job_quality_trainfit CSV (default: data/phase1/quality/job_quality_trainfit_v1.csv)",
    )
    args = parser.parse_args(argv)
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise SystemExit(
            f"--early-stopping-patience must be a positive integer (>=1), got {args.early_stopping_patience}"
        )
    if args.max_epochs is not None and args.max_epochs < 1:
        raise SystemExit(f"--max-epochs must be a positive integer (>=1), got {args.max_epochs}")
    return args


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cuda requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_internal_test_gate(eval_internal_test: bool) -> None:
    """Block internal_test reads until configs are frozen after validation-only selection."""
    if not eval_internal_test:
        return
    frozen = ROOT / "results" / "phase1" / "frozen_config.json"
    if not frozen.is_file():
        raise SystemExit(
            "BLOCKED: --eval-internal-test requires results/phase1/frozen_config.json. "
            "Freeze A0-A5 (including chosen lambdas) before any internal_test evaluation."
        )


def main() -> None:
    args = parse_args()
    matrix = load_matrix()
    exp = dict(matrix["experiments"][args.experiment])
    matrix_lambda = float(exp.get("lambda_contrastive", 0.0))
    if args.lambda_contrastive is not None:
        if args.experiment not in {"A4", "A5"}:
            raise SystemExit("--lambda-contrastive is only allowed for experiments A4 and A5")
        lambda_c_effective = float(args.lambda_contrastive)
        exp["lambda_contrastive"] = lambda_c_effective
        lambda_override = True
    else:
        lambda_c_effective = matrix_lambda
        lambda_override = False

    max_epochs = int(args.max_epochs or int(matrix["max_epochs"]))
    patience = resolve_early_stopping_patience(
        int(matrix["early_stopping_patience"]),
        args.early_stopping_patience,
    )

    # FAIL_CLOSED split/output binding BEFORE device/seed/model/embedding I/O.
    binding = resolve_binding(
        project_root=ROOT,
        tcore_mode=bool(args.tcore_mode),
        run_id=args.run_id,
        split_protocol=args.split_protocol,
        split_dir=args.split_dir,
        expected_train_sha=args.expected_train_sha,
        expected_val_sha=args.expected_val_sha,
        output_root=args.output_root,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        default_split_dir=SPLIT_DIR,
        split_manifest=args.split_manifest,
    )

    if binding.mode == "TCORE" and args.eval_internal_test:
        raise SystemExit(
            "FAIL_CLOSED: --eval-internal-test is forbidden in --tcore-mode; "
            "test scoring requires a separate TEST_ACCESS approval path"
        )

    _assert_internal_test_gate(bool(args.eval_internal_test))
    ensure_phase1_dirs()
    device = resolve_device(args.device)
    resident = gpu_resident_enabled(True if args.gpu_resident_emb else None)
    set_seed(args.seed)
    run_label = args.experiment
    if args.experiment in {"A4", "A5"}:
        run_label = f"{args.experiment}_lam{lambda_c_effective:g}"
    logger = RunLogger(
        run_label,
        args.seed,
        ROOT,
        run_id=binding.run_id,
        output_dir=binding.output_dir,
    )

    config = {
        "protocol": matrix["protocol"],
        "experiment_id": args.experiment,
        "experiment": exp,
        "seed": args.seed,
        "device": str(device),
        "max_epochs": max_epochs,
        "batch_size": int(matrix["batch_size"]),
        "learning_rate": float(matrix["learning_rate"]),
        "selection_metric": matrix["selection_metric"],
        "early_stopping_patience_matrix": int(matrix["early_stopping_patience"]),
        "early_stopping_patience": patience,
        "early_stopping_patience_cli_override": args.early_stopping_patience,
        "lambda_contrastive_effective": lambda_c_effective,
        "lambda_contrastive_matrix_default": matrix_lambda,
        "lambda_override": lambda_override,
        "rjdb_train_jsonl": str(rjdb_train_jsonl()),
        "weight_normalization": describe_weight_normalization(),
        "similarity_note": EQUIVALENCE_NOTE,
        "allow_official_rank_eval": bool(args.allow_official_rank_eval),
        "eval_internal_test": bool(args.eval_internal_test),
        "selection_uses_official_rank": False,
        "user_emb_sha256": file_sha256(USER_EMB_PATH),
        "job_emb_sha256": file_sha256(JOB_EMB_PATH),
        "split_meta_sha256": file_sha256(ROOT / "data" / "phase1" / "splits" / "split_meta.json"),
        "quality_stats_sha256": file_sha256(
            ROOT / "data" / "phase1" / "quality" / "job_quality_trainfit_v1_stats.json"
        ),
        "tcore_binding": binding.to_manifest(),
        "legacy_mode": binding.mode == "LEGACY_OVERLAP_ONLY",
        "gpu_resident_emb": bool(resident),
        "engineering_note": (
            "gpu_resident_emb is an engineering candidate path; frozen V1 formal default is False"
        ),
    }
    logger.log_config(config, MATRIX_JSON)
    if binding.output_dir is not None:
        (binding.output_dir / "input_manifest.json").write_text(
            json.dumps(binding.to_manifest(), indent=2), encoding="utf-8"
        )
        (binding.output_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        (binding.output_dir / "run_status.json").write_text(
            json.dumps({"status": "BOUND_READY", "run_id": binding.run_id}, indent=2),
            encoding="utf-8",
        )

    user_payload = torch.load(USER_EMB_PATH, map_location="cpu", weights_only=False)
    job_payload = torch.load(JOB_EMB_PATH, map_location="cpu", weights_only=False)
    user_ids = [str(item) for item in user_payload["ids"]]
    job_ids = [str(item) for item in job_payload["ids"]]
    user_emb = user_payload["embeddings"].float()
    job_emb = job_payload["embeddings"].float()
    user_emb, job_emb = place_frozen_embeddings(user_emb, job_emb, device, resident=resident)
    quality_map = load_trainfit_quality_map(args.quality_csv)
    user_row, job_row = build_index_tables(user_ids, job_ids)

    split_dir = binding.train_path.parent
    train_df = load_split_pairs_from_dir(split_dir, "train")
    val_df = load_split_pairs_from_dir(split_dir, "val")
    train_u, train_j, train_y, train_q = pairs_to_tensors(train_df, user_row, job_row, quality_map)
    val_u, val_j, val_y, val_q = pairs_to_tensors(val_df, user_row, job_row, quality_map)

    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    model_kind = exp.get("model", "text_match_head")
    if model_kind == "cosine":
        model = CosineSimilarityScorer().to(device)
        trainable = False
    elif model_kind == "dot":
        model = DotProductScorer().to(device)
        trainable = False
    else:
        model = QualityAwareDualEncoder(int(user_emb.shape[1])).to(device)
        trainable = True

    max_epochs = int(config["max_epochs"])
    batch_size = int(config["batch_size"])
    lambda_c = float(lambda_c_effective)
    supervised_name = exp.get("supervised_loss", "uniform_bce")
    if args.reverse_weight_kappa is not None:
        assert_kappa_one_matches_legacy_reverse()
        if supervised_name != "reverse_quality_bce":
            raise SystemExit(
                f"--reverse-weight-kappa requires reverse_quality_bce, got {supervised_name}"
            )

    rjdb = None
    if lambda_c != 0.0:
        if not RJDB_EMB_PATH.is_file():
            raise FileNotFoundError(
                f"RJDB embeddings required for contrastive runs: {RJDB_EMB_PATH}. "
                f"Source jsonl expected at {rjdb_train_jsonl()}"
            )
        rjdb_raw = torch.load(RJDB_EMB_PATH, map_location="cpu", weights_only=False)
        rjdb = place_rjdb_embeddings(rjdb_raw, device, resident=resident)

    if not trainable:
        scores = predict(model, user_emb, job_emb, val_u, val_j, device, batch_size, resident=resident)
        summary = {
            "best_epoch": 0,
            "best_val_auc": binary_auc(val_y, scores),
            "val_ap": binary_ap(val_y, scores),
            "trainable": False,
            "official_rank_touched": False,
        }
        if args.eval_internal_test:
            test_df = load_split_pairs("internal_test")
            test_u, test_j, test_y, _ = pairs_to_tensors(test_df, user_row, job_row, quality_map)
            test_scores = predict(
                model, user_emb, job_emb, test_u, test_j, device, batch_size, resident=resident
            )
            summary["internal_test_auc"] = binary_auc(test_y, test_scores)
            summary["internal_test_ap"] = binary_ap(test_y, test_scores)
        path = logger.finalize(summary)
        if binding.output_dir is not None:
            release_run_lock(binding.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"wrote {path}")
        return

    train_loader = DataLoader(
        PairDataset(train_u, train_j, train_y, train_q),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(matrix["learning_rate"]))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_auc = -1.0
    best_epoch = 0
    patience = int(config["early_stopping_patience"])
    bad_rounds = 0
    stop_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_sup = 0.0
        total_con = 0.0
        rows = 0
        gen = torch.Generator().manual_seed(args.seed + epoch)
        for user_index, job_index, labels, quality in train_loader:
            labels = labels.to(device)
            u_batch, j_batch = gather_pair_embeddings(
                user_emb, job_emb, user_index, job_index, device, resident=resident
            )
            logits = model(u_batch, j_batch)
            sup = supervised_loss(
                supervised_name,
                logits,
                labels,
                quality.to(device),
                pos_weight,
                reverse_kappa=args.reverse_weight_kappa,
            )
            if lambda_c != 0.0 and rjdb is not None:
                n_triples = int(rjdb["job"].shape[0])
                take = min(int(matrix["contrastive_per_batch"]), n_triples)
                # RNG unchanged: CPU generator → integer indices; emb gather may be on-device.
                choice = torch.randint(n_triples, (take,), generator=gen)
                matched, unmatched, job_t = gather_rjdb_triplet(
                    rjdb, choice, device, resident=resident
                )
                positive = model(matched, job_t)
                negative = model(unmatched, job_t)
                con = contrastive_term(positive, negative)
                loss = sup + lambda_c * con
                con_value = float(con.item())
            else:
                loss = sup
                con_value = 0.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_sup += float(sup.item()) * len(labels)
            total_con += con_value * len(labels)
            rows += len(labels)
            if args.dry_run:
                break
        # Single validation forward; AUC and AP share the same logits (no extra forward).
        val_scores = predict(
            model, user_emb, job_emb, val_u, val_j, device, batch_size, resident=resident
        )
        val_auc = binary_auc(val_y, val_scores)
        val_ap = binary_ap(val_y, val_scores)
        row = {
            "epoch": epoch,
            "train_supervised": total_sup / max(rows, 1),
            "train_contrastive": total_con / max(rows, 1),
            "val_auc": val_auc,
            "val_ap": val_ap,
        }
        logger.log_epoch(row)
        print(row, flush=True)
        best_val_auc, bad_rounds, improved, should_stop = early_stop_update(
            val_auc, best_val_auc, bad_rounds, patience
        )
        if improved:
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stop_epoch = epoch
        if should_stop:
            print(f"early stop at epoch {epoch}", flush=True)
            break
        if args.dry_run:
            break

    model.load_state_dict(best_state)
    ckpt_dir = (binding.output_dir / "checkpoints") if binding.output_dir is not None else logger.dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save({"model_state": best_state, "config": config, "best_epoch": best_epoch}, ckpt_path)
    summary = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "stop_epoch": stop_epoch,
        "early_stopping_patience": patience,
        "early_stopping_patience_matrix": int(config["early_stopping_patience_matrix"]),
        "early_stopping_patience_cli_override": config["early_stopping_patience_cli_override"],
        "checkpoint": str(ckpt_path),
        "official_rank_touched": False,
        "pos_weight": neg / max(pos, 1.0),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
        "lambda_contrastive_effective": lambda_c,
        "lambda_override": lambda_override,
        "reverse_weight_kappa": args.reverse_weight_kappa,
        "max_epochs": max_epochs,
        "selection_metric": "val_auc",
    }
    if args.eval_internal_test:
        test_df = load_split_pairs("internal_test")
        test_u, test_j, test_y, _ = pairs_to_tensors(test_df, user_row, job_row, quality_map)
        test_scores = predict(
            model, user_emb, job_emb, test_u, test_j, device, batch_size, resident=resident
        )
        summary["internal_test_auc"] = binary_auc(test_y, test_scores)
        summary["internal_test_ap"] = binary_ap(test_y, test_scores)
        if args.ranking_diagnostics:
            summary["internal_test_rank_job_diag"] = observed_ranking_diagnostics(
                test_df["user_id"].to_numpy(), test_y, test_scores
            )
            summary["internal_test_rank_resume_diag"] = observed_ranking_diagnostics(
                test_df["jd_no"].to_numpy(), test_y, test_scores
            )
    if args.allow_official_rank_eval:
        from load_pjf_data import load_rank_job, load_rank_resume
        from src.eval.evaluate_ranking import ranking_metrics

        rank_job = load_rank_job()
        rank_resume = load_rank_resume()
        job_scores = predict(
            model,
            user_emb,
            job_emb,
            np.array([user_row[u] for u in rank_job["user_id"]], dtype=np.int64),
            np.array([job_row[j] for j in rank_job["jd_no"]], dtype=np.int64),
            device,
            batch_size,
            resident=resident,
        )
        resume_scores = predict(
            model,
            user_emb,
            job_emb,
            np.array([user_row[u] for u in rank_resume["user_id"]], dtype=np.int64),
            np.array([job_row[j] for j in rank_resume["jd_no"]], dtype=np.int64),
            device,
            batch_size,
            resident=resident,
        )
        summary["official_rank_touched"] = True
        summary["official_rank_job"] = ranking_metrics(
            rank_job["user_id"].to_numpy(), rank_job["satisfied"].to_numpy(), job_scores
        )
        summary["official_rank_resume"] = ranking_metrics(
            rank_resume["jd_no"].to_numpy(), rank_resume["satisfied"].to_numpy(), resume_scores
        )
        summary["official_rank_warning"] = (
            "Official rank metrics are contaminated for selection history and must not "
            "drive checkpoint choice or hyperparameter search."
        )
        (logger.dir / "official_rank_access.json").write_text(
            json.dumps(
                {
                    "accessed": True,
                    "used_for_selection": False,
                    "warning": summary["official_rank_warning"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    path = logger.finalize(summary)
    if binding.output_dir is not None:
        (binding.output_dir / "run_status.json").write_text(
            json.dumps({"status": "COMPLETED", "run_id": binding.run_id, "summary": path.name}, indent=2),
            encoding="utf-8",
        )
        release_run_lock(binding.output_dir)
    printable = {key: value for key, value in summary.items() if not key.startswith("official_rank_j")}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

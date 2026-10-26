"""Train-only quality score fitting for Phase 1.

Reuses deterministic structure and keyword rules from the v2 pipeline.
Only text-length p05/p95 are fitted on jobs that appear in the frozen train split.
Original ``job_cleaned_v2.csv`` is never modified.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase1.paths import JOB_V2_PATH, QUALITY_DIR, ensure_phase1_dirs
from src.phase1.split import load_split_pairs

JOB_STRUCTURE_FIELDS = [
    "city",
    "jd_sub_type",
    "min_salary",
    "max_salary",
    "min_years",
    "min_edu_level",
    "job_text",
]
SALARY_FIELDS = {"min_salary", "max_salary"}
SENTINELS = {"", "-", "\\n", "nan", "none", "null", "na", "n/a", "nat"}
KEY_FULL_CHARS = 8
TITLE_FULL_CHARS = 8
SUBTYPE_FULL_CHARS = 6
HEAD_CHARS = 200
HEAD_FULL_CHARS = 80
STOP_ZH_CHARS = set("的了和与或在是为及等对将并从以按把被让向于也就都而但其该这那些中上下")
STOP_PHRASES = (
    "以及",
    "一个",
    "我们",
    "公司",
    "岗位",
    "职位",
    "工作",
    "职责",
    "要求",
    "任职",
    "资格",
    "以上",
    "以下",
    "相关",
    "能力",
    "具有",
    "熟悉",
    "负责",
    "进行",
    "能够",
    "可以",
)
STOP_EN = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "as",
    "by",
    "at",
    "from",
    "is",
    "are",
    "be",
    "this",
    "that",
    "our",
    "we",
    "you",
    "your",
    "job",
    "role",
    "work",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#./_-]*|[\u4e00-\u9fff]")


def is_missing(value: object, field: str | None = None) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text.lower() in SENTINELS or text == "\\N":
        return True
    if field in SALARY_FIELDS and text in {"0", "0.0"}:
        return True
    tokens = [part.strip() for part in text.replace("|", ",").split(",")]
    if tokens and all(part == "" or part == "-" or part == "\\N" or part.lower() in SENTINELS for part in tokens):
        return True
    return False


def present_text(value: object, field: str | None = None) -> str:
    if is_missing(value, field):
        return ""
    return str(value).strip()


def length_signal(value: object, full_chars: int, field: str | None = None) -> float:
    text = present_text(value, field)
    if not text:
        return 0.0
    return min(1.0, len(text) / full_chars)


def non_stopword_chars(text: str) -> int:
    head = text[:HEAD_CHARS]
    for phrase in STOP_PHRASES:
        head = head.replace(phrase, " ")
    total = 0
    for token in TOKEN_RE.findall(head):
        if len(token) == 1 and "\u4e00" <= token <= "\u9fff":
            if token in STOP_ZH_CHARS:
                continue
            total += 1
        elif token.lower() in STOP_EN:
            continue
        else:
            total += len(token)
    return total


def head_signal(value: object) -> float:
    text = present_text(value, "job_text")
    if not text:
        return 0.0
    return min(1.0, non_stopword_chars(text) / HEAD_FULL_CHARS)


def structure_score(frame: pd.DataFrame) -> pd.Series:
    flags = pd.DataFrame(
        {field: ~frame[field].map(lambda value, field=field: is_missing(value, field)) for field in JOB_STRUCTURE_FIELDS}
    )
    return flags.mean(axis=1).astype(float)


def fit_length_percentiles(train_job_texts: pd.Series) -> tuple[float, float]:
    lengths = train_job_texts.map(lambda value: len(present_text(value, "job_text")))
    low = float(lengths.quantile(0.05, interpolation="linear"))
    high = float(lengths.quantile(0.95, interpolation="linear"))
    return low, high


def length_score_from_stats(texts: pd.Series, low: float, high: float) -> pd.Series:
    lengths = texts.map(lambda value: float(len(present_text(value, "job_text"))))
    if high <= low:
        return (lengths > 0).astype(float)
    clipped = lengths.clip(lower=low, upper=high)
    return ((clipped - low) / (high - low)).astype(float)


def build_train_only_quality(job_path: Path = JOB_V2_PATH) -> dict:
    ensure_phase1_dirs()
    jobs = pd.read_csv(job_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    train = load_split_pairs("train")
    train_job_ids = set(train["jd_no"].tolist())
    train_jobs = jobs[jobs["jd_no"].isin(train_job_ids)]
    if train_jobs.empty:
        raise RuntimeError("no train jobs found in job table")

    low, high = fit_length_percentiles(train_jobs["job_text"])
    text_score = length_score_from_stats(jobs["job_text"], low, high)
    structure = structure_score(jobs)
    key_signal = jobs["key"].map(lambda value: length_signal(value, KEY_FULL_CHARS, "key"))
    title_signal = jobs["jd_title"].map(lambda value: length_signal(value, TITLE_FULL_CHARS, "jd_title"))
    subtype_signal = jobs["jd_sub_type"].map(lambda value: length_signal(value, SUBTYPE_FULL_CHARS, "jd_sub_type"))
    head = jobs["job_text"].map(head_signal)
    keyword = (0.25 * key_signal + 0.25 * title_signal + 0.25 * subtype_signal + 0.25 * head).clip(0.0, 1.0)
    quality = (0.4 * text_score + 0.4 * structure + 0.2 * keyword).clip(0.0, 1.0)

    out = pd.DataFrame(
        {
            "jd_no": jobs["jd_no"],
            "text_length_score_trainfit": text_score.round(6),
            "structure_score": structure.round(6),
            "keyword_score": keyword.round(6),
            "quality_score_trainfit": quality.round(6),
            "in_train_jobs": jobs["jd_no"].isin(train_job_ids).astype(int),
        }
    )
    out_path = QUALITY_DIR / "job_quality_trainfit_v1.csv"
    out.to_csv(out_path, index=False)
    stats = {
        "formula": "0.4 * text_length_score + 0.4 * structure_score + 0.2 * keyword_score",
        "length_fit_scope": "jobs appearing in frozen train split only",
        "n_train_jobs": int(len(train_job_ids)),
        "length_p05": low,
        "length_p95": high,
        "structure_fields": JOB_STRUCTURE_FIELDS,
        "structure_rule": "deterministic missingness; no corpus-fitted parameters",
        "keyword_rule": "deterministic length/head signals; no corpus-fitted parameters beyond fixed stop lists",
        "source_job_table": str(job_path),
        "output": str(out_path),
        "quality_mean_all": float(quality.mean()),
        "quality_mean_train_jobs": float(quality[jobs["jd_no"].isin(train_job_ids)].mean()),
        "original_job_cleaned_v2_untouched": True,
    }
    stats_path = QUALITY_DIR / "job_quality_trainfit_v1_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def load_trainfit_quality_map(path: Path | str | None = None) -> dict[str, float]:
    path = Path(path) if path is not None else (QUALITY_DIR / "job_quality_trainfit_v1.csv")
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    scores = pd.to_numeric(frame["quality_score_trainfit"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    return {job_id: float(score) for job_id, score in zip(frame["jd_no"], scores)}


def build_train_only_quality_for_split(
    train_pairs: pd.DataFrame,
    out_dir: Path,
    job_path: Path = JOB_V2_PATH,
) -> dict:
    """Fit length p05/p95 on jobs in ``train_pairs`` only; write CSV+stats under ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = pd.read_csv(job_path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    train_job_ids = set(train_pairs["jd_no"].astype(str).tolist())
    train_jobs = jobs[jobs["jd_no"].astype(str).isin(train_job_ids)]
    if train_jobs.empty:
        raise RuntimeError("no train jobs found in job table")

    low, high = fit_length_percentiles(train_jobs["job_text"])
    text_score = length_score_from_stats(jobs["job_text"], low, high)
    structure = structure_score(jobs)
    key_signal = jobs["key"].map(lambda value: length_signal(value, KEY_FULL_CHARS, "key"))
    title_signal = jobs["jd_title"].map(lambda value: length_signal(value, TITLE_FULL_CHARS, "jd_title"))
    subtype_signal = jobs["jd_sub_type"].map(lambda value: length_signal(value, SUBTYPE_FULL_CHARS, "jd_sub_type"))
    head = jobs["job_text"].map(head_signal)
    keyword = (0.25 * key_signal + 0.25 * title_signal + 0.25 * subtype_signal + 0.25 * head).clip(0.0, 1.0)
    quality = (0.4 * text_score + 0.4 * structure + 0.2 * keyword).clip(0.0, 1.0)

    out = pd.DataFrame(
        {
            "jd_no": jobs["jd_no"],
            "text_length_score_trainfit": text_score.round(6),
            "structure_score": structure.round(6),
            "keyword_score": keyword.round(6),
            "quality_score_trainfit": quality.round(6),
            "in_train_jobs": jobs["jd_no"].astype(str).isin(train_job_ids).astype(int),
        }
    )
    out_path = out_dir / "job_quality_trainfit_v1.csv"
    out.to_csv(out_path, index=False)
    stats = {
        "formula": "0.4 * text_length_score + 0.4 * structure_score + 0.2 * keyword_score",
        "length_fit_scope": "jobs appearing in this split's train pairs only",
        "n_train_jobs": int(len(train_job_ids)),
        "length_p05": low,
        "length_p95": high,
        "structure_fields": JOB_STRUCTURE_FIELDS,
        "source_job_table": str(job_path),
        "output": str(out_path),
        "quality_mean_all": float(quality.mean()),
        "quality_mean_train_jobs": float(quality[jobs["jd_no"].astype(str).isin(train_job_ids)].mean()),
        "original_job_cleaned_v2_untouched": True,
    }
    (out_dir / "job_quality_trainfit_v1_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats

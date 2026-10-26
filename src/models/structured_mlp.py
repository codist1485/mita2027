"""Structured dual-tower MLP for Person-Job Fit.

Categorical fields are label-encoded. Index 0 is reserved for missing values
and categories unseen at fit time, and it is the embedding padding index.
Numeric fields are median-imputed and z-scored with statistics from the
training entities. The two towers emit 64-d vectors. The match logit is a
linear layer over the concatenation of the two vectors and their product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

SEED = 42
HIDDEN_SIZE = 64
EMBEDDING_DIM = 16
REF_YEAR = 2019

USER_CATEGORICAL = (
    "live_city_id",
    "desire_jd_city_id",
    "desire_jd_industry_id",
    "desire_jd_type_id",
    "desire_jd_salary_id",
    "cur_industry_id",
    "cur_jd_type",
    "cur_salary_id",
    "cur_degree_id",
)
JOB_CATEGORICAL = (
    "city",
    "jd_sub_type",
    "is_travel",
    "min_years",
    "min_edu_level",
)
USER_NUMERIC = ("age", "start_work_year", "work_years")
JOB_NUMERIC = ("require_nums", "min_salary", "max_salary")
SENTINELS = {"", "-", "\\n", "nan", "none", "null", "na", "n/a", "nat"}


def is_missing_token(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text == "" or text == "\\N" or text.lower() in SENTINELS


def clean_category(value: object) -> str:
    if is_missing_token(value):
        return ""
    return str(value).strip()


def _to_float(value: object) -> float:
    if is_missing_token(value):
        return np.nan
    try:
        return float(str(value).strip())
    except ValueError:
        return np.nan


@dataclass
class NumericNorm:
    median: float
    mean: float
    std: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(values), self.median, values)
        return ((filled - self.mean) / self.std).astype(np.float32)


class CategoricalEncoder:
    """Sorted label encoder. 0 means missing or unknown."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def fit(self, values: pd.Series) -> None:
        tokens = sorted({clean_category(value) for value in values})
        tokens = [token for token in tokens if token]
        self.vocab = {token: index for index, token in enumerate(tokens, start=1)}

    def transform(self, values: pd.Series) -> np.ndarray:
        encoded = np.zeros(len(values), dtype=np.int64)
        for index, value in enumerate(values.tolist()):
            token = clean_category(value)
            encoded[index] = self.vocab.get(token, 0) if token else 0
        return encoded

    @property
    def cardinality(self) -> int:
        return len(self.vocab) + 1


class StructuredFeatureStore:
    """User and job matrices aligned with the row order of the source frames."""

    def __init__(self) -> None:
        self.user_encoders = {name: CategoricalEncoder() for name in USER_CATEGORICAL}
        self.job_encoders = {name: CategoricalEncoder() for name in JOB_CATEGORICAL}
        self.user_norms: dict[str, NumericNorm] = {}
        self.job_norms: dict[str, NumericNorm] = {}
        self.user_cat = np.empty((0, len(USER_CATEGORICAL)), dtype=np.int64)
        self.user_num = np.empty((0, len(USER_NUMERIC)), dtype=np.float32)
        self.job_cat = np.empty((0, len(JOB_CATEGORICAL)), dtype=np.int64)
        self.job_num = np.empty((0, len(JOB_NUMERIC)), dtype=np.float32)

    def fit(self, users: pd.DataFrame, jobs: pd.DataFrame, train_user_ids: set[str], train_job_ids: set[str]) -> None:
        user_fit = users.loc[users["user_id"].isin(train_user_ids)]
        job_fit = jobs.loc[jobs["jd_no"].isin(train_job_ids)]
        for name, encoder in self.user_encoders.items():
            encoder.fit(user_fit[name])
        for name, encoder in self.job_encoders.items():
            encoder.fit(job_fit[name])
        user_raw = _user_numeric_frame(users)
        job_raw = _job_numeric_frame(jobs)
        self.user_norms = {
            name: _fit_norm(user_raw.loc[users["user_id"].isin(train_user_ids), name].to_numpy(dtype=np.float64))
            for name in USER_NUMERIC
        }
        self.job_norms = {
            name: _fit_norm(job_raw.loc[jobs["jd_no"].isin(train_job_ids), name].to_numpy(dtype=np.float64))
            for name in JOB_NUMERIC
        }
        self.user_cat = np.column_stack([self.user_encoders[name].transform(users[name]) for name in USER_CATEGORICAL])
        self.job_cat = np.column_stack([self.job_encoders[name].transform(jobs[name]) for name in JOB_CATEGORICAL])
        self.user_num = np.column_stack([self.user_norms[name].apply(user_raw[name].to_numpy(dtype=np.float64)) for name in USER_NUMERIC])
        self.job_num = np.column_stack([self.job_norms[name].apply(job_raw[name].to_numpy(dtype=np.float64)) for name in JOB_NUMERIC])

    def user_cards(self) -> list[int]:
        return [self.user_encoders[name].cardinality for name in USER_CATEGORICAL]

    def job_cards(self) -> list[int]:
        return [self.job_encoders[name].cardinality for name in JOB_CATEGORICAL]


def _fit_norm(values: np.ndarray) -> NumericNorm:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return NumericNorm(0.0, 0.0, 1.0)
    median = float(np.median(valid))
    filled = np.where(np.isnan(values), median, values)
    mean = float(filled.mean())
    std = float(filled.std())
    if std < 1e-6:
        std = 1.0
    return NumericNorm(median, mean, std)


def _user_numeric_frame(users: pd.DataFrame) -> pd.DataFrame:
    age = users["birthday"].map(_to_float).to_numpy(dtype=np.float64)
    age = np.where(age <= 0, np.nan, age)
    start_year = users["start_work_date"].map(_to_float).to_numpy(dtype=np.float64)
    start_year = np.where((start_year < 1950) | (start_year > REF_YEAR), np.nan, start_year)
    work_years = REF_YEAR - start_year
    work_years = np.where((work_years < 0) | (work_years > 60), np.nan, work_years)
    return pd.DataFrame({"age": age, "start_work_year": start_year, "work_years": work_years})


def _job_numeric_frame(jobs: pd.DataFrame) -> pd.DataFrame:
    require_nums = jobs["require_nums"].map(_to_float).to_numpy(dtype=np.float64)
    min_salary = jobs["min_salary"].map(_to_float).to_numpy(dtype=np.float64)
    max_salary = jobs["max_salary"].map(_to_float).to_numpy(dtype=np.float64)
    min_salary = np.where(min_salary <= 0, np.nan, min_salary)
    max_salary = np.where(max_salary <= 0, np.nan, max_salary)
    require_nums = np.where(require_nums < 0, np.nan, require_nums)
    return pd.DataFrame(
        {
            "require_nums": np.log1p(require_nums),
            "min_salary": np.log1p(min_salary),
            "max_salary": np.log1p(max_salary),
        }
    )


class Tower(nn.Module):
    def __init__(self, cardinalities: list[int], n_numeric: int, hidden_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embedding_dim, padding_idx=0) for cardinality in cardinalities]
        )
        input_size = embedding_dim * len(cardinalities) + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, categories: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(categories[:, index]) for index, embedding in enumerate(self.embeddings)]
        hidden = torch.cat([*embedded, numeric], dim=-1)
        return self.mlp(hidden)


class StructuredMLP(nn.Module):
    """Dual tower with a linear head on ``[user, job, user * job]``."""

    def __init__(
        self,
        user_cards: list[int],
        job_cards: list[int],
        n_user_numeric: int,
        n_job_numeric: int,
        hidden_size: int = HIDDEN_SIZE,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.user_tower = Tower(user_cards, n_user_numeric, hidden_size, embedding_dim)
        self.job_tower = Tower(job_cards, n_job_numeric, hidden_size, embedding_dim)
        self.head = nn.Linear(hidden_size * 3, 1)

    def forward(
        self,
        user_cat: torch.Tensor,
        user_num: torch.Tensor,
        job_cat: torch.Tensor,
        job_num: torch.Tensor,
    ) -> torch.Tensor:
        user_vector = self.user_tower(user_cat, user_num)
        job_vector = self.job_tower(job_cat, job_num)
        combined = torch.cat([user_vector, job_vector, user_vector * job_vector], dim=-1)
        return self.head(combined).squeeze(-1)

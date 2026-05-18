"""XGBoost trainer family — config schema, factory, and ingest-path helpers.

Public API (re-exported from ``rux_ml.training``):

- :class:`XGBoostTraining` — the discriminated-union variant for XGBoost
- :func:`make_xgboost_trainer` — factory registered in ``TRAINER_FAMILIES``
- :func:`select_ingest`, :func:`estimate_x_bytes`, :data:`DEFAULT_BYTES_PER_GB`
  — XGBoost-specific ingest-path helpers (per D3)
"""

from rux_ml.training.xgboost.config import XGBoostTraining
from rux_ml.training.xgboost.factory import make_xgboost_trainer
from rux_ml.training.xgboost.ingest import (
    DEFAULT_BYTES_PER_GB,
    estimate_x_bytes,
    select_ingest,
)

__all__ = [
    "DEFAULT_BYTES_PER_GB",
    "XGBoostTraining",
    "estimate_x_bytes",
    "make_xgboost_trainer",
    "select_ingest",
]

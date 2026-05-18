"""LightGBM trainer family — config schema, factory, and Dataset helper.

Public API (re-exported from ``rux_ml.training``):

- :class:`LightGBMTraining` — the discriminated-union variant for LightGBM
- :func:`make_lightgbm_trainer` — factory registered in ``TRAINER_FAMILIES``
- :func:`build_dataset` — LightGBM-specific ingest helper (per Q-Ingest research:
  no tier-switch needed; histogram uint8-binned at construction).

GPU support is intentionally deferred per Q-GPU research findings — LightGBM-GPU
is benchmarked 8-28x slower than XGBoost-GPU on workbench-scale data, the
CUDA install path via ``uv`` is non-trivial (no prebuilt CUDA wheel on PyPI),
and multiple open GitHub issues document brittle pip-source-CUDA installs.
``make_lightgbm_trainer`` raises ``NotImplementedError`` when ``cfg.device ==
"cuda"``; future PR widens the path when prereqs are met.
"""

from rux_ml.training.lightgbm.config import LightGBMTraining
from rux_ml.training.lightgbm.factory import make_lightgbm_trainer
from rux_ml.training.lightgbm.ingest import build_dataset

__all__ = [
    "LightGBMTraining",
    "build_dataset",
    "make_lightgbm_trainer",
]

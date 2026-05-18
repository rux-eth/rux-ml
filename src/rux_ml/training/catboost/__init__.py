"""CatBoost trainer family — config schema, factory, and Pool placeholder.

Public API (re-exported from ``rux_ml.training``):

- :class:`CatBoostTraining` — the discriminated-union variant for CatBoost
- :func:`make_catboost_trainer` — factory registered in ``TRAINER_FAMILIES``
- :func:`build_pool` — placeholder helper (CatBoost ``fit(DataFrame, y)`` works
  natively without ``Pool`` construction; the helper exists for symmetry with
  the xgboost/ + lightgbm/ subpackages' ``build_*`` helpers and as a future
  hook point if a CatBoost-specific ingest decision rule emerges).

Per PR-019 Q-Cat: CatBoost does NOT auto-detect pandas Categorical dtype
(opposite of LightGBM). The ``_CatBoostTrainerShim`` extracts categorical
column names from the DataFrame at fit() time and passes them via the
``cat_features=`` kwarg. The workbench's ``_ColumnRouter`` (PR-005) is
unchanged — its low-card-passthrough output (pandas Categorical dtype on
low-card columns) is exactly what the shim looks for.

Per PR-019 Q-GPU: CatBoost ships with prebuilt CUDA-enabled PyPI wheels
out-of-box (no extras, no source build, no container delta). Default
``device == "cuda"`` activates ``task_type="GPU"``; ``"cpu"`` activates
``task_type="CPU"``. RTX 4090 (CC 8.9) field-confirmed via issue #2649.

Per PR-019 Q-Parallel: CatBoost uses Intel TBB, NOT OpenMP. The workbench's
``OMP_NUM_THREADS`` env-var pinning (from PR-011) is invisible to CatBoost.
The factory passes ``cfg.memory.omp_threads`` explicitly as
``thread_count=`` to match the workbench's intent.
"""

from rux_ml.training.catboost.config import CatBoostTraining
from rux_ml.training.catboost.factory import make_catboost_trainer
from rux_ml.training.catboost.ingest import build_pool

__all__ = [
    "CatBoostTraining",
    "build_pool",
    "make_catboost_trainer",
]

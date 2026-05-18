"""``CatBoostTraining`` — discriminated-union variant for CatBoost (per PR-019).

Inherits family-agnostic fields (``device``, ``metric``, ``early_stopping_rounds``)
from :class:`rux_ml.training.base.TrainingBase`. CatBoost's outlier surface
shapes this variant:

- **No `model_kwargs` escape hatch** (Q-MK / Q-Wrap, PROVEN at
  ``catboost/python-package/catboost/core.py @ v1.2.10`` L5305-5425):
  ``CatBoostClassifier.__init__`` has 117 explicit params with no ``**kwargs``,
  so every workbench-exposed knob must be a typed field. The 14-field set
  below is the Q-Wrap "minimal" recommendation (search-space dims + family-
  agnostic plumbing); CTR / text / class-imbalance knobs are deferred to
  follow-up PRs.

- **`loss_function` is NOT exposed** (Q-Wrap §4): CatBoost's ``AUC`` is a
  valid ``eval_metric`` but NOT a valid ``loss_function``. The factory
  derives ``loss_function`` from the workbench task (classifier →
  ``"Logloss"``; regressor → ``"RMSE"``); the user-facing ``cfg.metric``
  maps to CatBoost's ``eval_metric`` via ``_METRIC_TRANSLATE`` in the
  factory.

- **`cat_features` is NOT a config field** (Q-Cat): the factory extracts
  it from the DataFrame's dtypes at fit time via the ``_CatBoostTrainerShim``.
  Users don't list categorical columns here; the workbench's features layer
  (``_ColumnRouter``) is the single source of truth.

CatBoost-canonical field names (not the sklearn aliases):
- ``iterations`` (not ``n_estimators``)
- ``depth`` (not ``max_depth``)
- ``l2_leaf_reg`` (CatBoost-specific)
- ``random_seed`` (renamed from ``random_state``; factory passes it
  from the ``seed`` arg, not from a config field)
"""

from __future__ import annotations

from typing import Literal

from rux_ml.training.base import TrainingBase


class CatBoostTraining(TrainingBase):
    """CatBoost variant of the discriminated ``TrainingConfig``."""

    kind: Literal["catboost"] = "catboost"

    # Boosting + capacity. CatBoost-canonical names.
    iterations: int = 1000
    learning_rate: float = 0.03  # CatBoost default
    depth: int = 6  # CatBoost default
    border_count: int = 254  # CPU default (Q-HPO)

    # Regularization (CatBoost-specific).
    l2_leaf_reg: float = 3.0  # CatBoost default
    random_strength: float = 1.0  # CatBoost default
    bagging_temperature: float = 1.0  # only active when bootstrap_type="Bayesian"
    subsample: float = 0.8  # only active when bootstrap_type != "Bayesian"

    # Boosting strategy + bootstrap.
    bootstrap_type: Literal["Bayesian", "Bernoulli", "MVS", "Poisson", "No"] = "Bayesian"
    grow_policy: Literal["SymmetricTree", "Depthwise", "Lossguide"] = "SymmetricTree"

    # Leaf-level regularization.
    min_data_in_leaf: int = 1  # CatBoost default
    one_hot_max_size: int = 2  # CatBoost default; bigger one-hot than this triggers CTR

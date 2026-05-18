"""``LightGBMTraining`` — discriminated-union variant for LightGBM (per PR-018).

Inherits family-agnostic fields (``device``, ``metric``, ``early_stopping_rounds``)
from :class:`rux_ml.training.base.TrainingBase`. The field set follows Q-Wrap
research findings (LightGBM v4.5+ sklearn-wrapper) and the Q-HPO recommended
search-space defaults; field defaults match LightGBM's own defaults so users
who don't override anything get the upstream-library-canonical behavior.

Q-MK locked: ``model_kwargs: dict[str, Any]`` is retained because
``LGBMModel.__init__`` accepts ``**kwargs: Any`` (verified at v4.5.0
``python-package/lightgbm/sklearn.py:506``). Convention: ``model_kwargs``
wins on key collisions with the typed fields.

Q-Wrap surprises captured at field level:
- ``early_stopping_rounds`` (inherited from ``TrainingBase``) is NOT a
  constructor kwarg on LightGBM — the factory translates it to a
  ``lightgbm.early_stopping(N)`` callback at fit time.
- ``bagging_freq`` defaults to ``1`` so ``bagging_fraction < 1.0``
  actually fires (LightGBM gotcha: ``bagging_freq=0`` disables bagging
  regardless of ``bagging_fraction``).
- ``deterministic`` defaults to ``False`` (matches LightGBM default).
  Required ``True`` for PR-013 CPU bit-exact contract; users opt in per
  determinism test, not on the default path.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rux_ml.training.base import TrainingBase


class LightGBMTraining(TrainingBase):
    """LightGBM variant of the discriminated ``TrainingConfig``."""

    kind: Literal["lightgbm"] = "lightgbm"

    # Boosting strategy. "gbdt" (default; conventional), "dart", "rf".
    # Q-HPO: not a search dimension by default; users override deliberately.
    boosting_type: Literal["gbdt", "dart", "rf"] = "gbdt"

    # Core capacity knobs. ``num_leaves`` is the primary regularizer in
    # LightGBM (leaf-wise growth); ``max_depth=-1`` means "no limit".
    learning_rate: float = 0.1
    num_leaves: int = 31
    max_depth: int = -1
    n_estimators: int = 100

    # Regularization. Sklearn-wrapper aliases (``min_child_samples`` for
    # ``min_data_in_leaf``; ``colsample_bytree`` for ``feature_fraction``;
    # ``subsample`` for ``bagging_fraction``). The factory translates
    # rux-ml names to LightGBM names.
    min_data_in_leaf: int = 20
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0
    # ``bagging_freq`` MUST be > 0 for bagging to actually fire (LightGBM
    # docs: "0 means disable bagging"). Default 1 means "bag every iteration"
    # — when ``bagging_fraction = 1.0`` this is still a no-op.
    bagging_freq: int = 1

    # L1/L2 regularization (rux-ml names match XGBoost convention; factory
    # translates to ``reg_alpha``/``reg_lambda`` which LightGBM accepts as
    # sklearn-wrapper aliases of ``lambda_l1``/``lambda_l2``).
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0

    # PR-013 CPU bit-exact contract requires ``deterministic=True`` alongside
    # ``random_state`` (LightGBM's own ``random_state`` has lower priority
    # than per-feature seeds — see Q-Wrap §6). Default False to avoid the
    # ~10 % wall-clock overhead on training; opt in for determinism tests.
    deterministic: bool = False

    # LightGBM-specific kwargs escape hatch. Overrides typed fields on key
    # collision (matches PR-006 / XGBoost convention; verified by Q-MK that
    # LGBMModel.__init__ accepts **kwargs at v4.5.0 sklearn.py:506).
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

"""``XGBoostTraining`` — discriminated-union variant for XGBoost (per PR-017).

Inherits family-agnostic fields (``device``, ``metric``, ``early_stopping_rounds``)
from :class:`rux_ml.training.base.TrainingBase`. The XGBoost-specific defaults
match the v0 ``TrainingConfig`` (per PR-006) so ``training_cfg_hash`` is
preserved for every existing v0 XGBoost trial across the PR-017 refactor —
``canonical_json`` sorts keys (per ``_internal.hashing``), so field declaration
order does not affect the hash.

``model_kwargs`` is XGBoost-supported (``XGBClassifier.__init__`` takes
``**kwargs``) and follows the PR-006 convention: ``model_kwargs`` wins on key
collisions with the top-level typed fields. Per-family variants for other
families that do NOT accept ``**kwargs`` (notably CatBoost in PR-019) omit
``model_kwargs`` entirely — see ``docs/0.1/DESIGN-log.md`` Q-MK.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from rux_ml.training.base import TrainingBase


class XGBoostTraining(TrainingBase):
    """XGBoost variant of the discriminated ``TrainingConfig``."""

    kind: Literal["xgboost"] = "xgboost"

    # XGBoost-specific defaults that match D3 / D4 decisions (per PR-006).
    enable_categorical: bool = True
    tree_method: Literal["hist", "approx", "exact"] = "hist"

    # Native xgb.train() escape hatch per D5 (~5% of cases).
    use_native: bool = False

    # Common tunable hyperparameters surfaced at the top level so dot-path
    # CLI/env overrides stay clean (e.g. RUXML_TRAINING__LEARNING_RATE=0.01).
    learning_rate: float = 0.1
    max_depth: int = 6
    n_estimators: int = 100
    subsample: float = 1.0
    colsample_bytree: float = 1.0

    # XGBoost-specific kwargs escape hatch. Overrides typed fields on key
    # collision (per PR-006 convention; verified in test_factory.py).
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

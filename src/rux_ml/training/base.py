"""Family-agnostic ``TrainingBase`` (per PR-017 / docs/0.1/DESIGN-log.md Q1).

Lives in a neutral module (``training.base``, not ``training.xgboost`` and not
``config.training``) so per-family variants can inherit from it without
creating a circular import — the discriminated union assembled in
``config.training`` then imports each variant from its family subpackage.

Fields here are the ones every estimator family carries identically:
``device``, ``metric``, ``early_stopping_rounds``. Family-specific fields
(``learning_rate``, ``num_leaves``, ``depth``, ``iterations``, ``model_kwargs``,
etc.) live in the per-family variant under ``training/<family>/config.py``.

The strict ``model_config`` is inlined here rather than inherited from
``rux_ml.config._strict_model.StrictModel`` because importing the latter
triggers ``rux_ml.config.__init__`` to run, which transitively re-imports this
module (via ``rux_ml.config.training``'s discriminated-union assembly) and
deadlocks. Same semantic contract, no shared base class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class TrainingBase(BaseModel):
    """Family-agnostic training-config fields shared across all Trainer families.

    The discriminator field ``kind`` is declared on each concrete variant
    (``XGBoostTraining``, future ``LightGBMTraining``, ``CatBoostTraining``) as
    a ``Literal[<family>]``, not on this base — declaring it as a plain ``str``
    here would force variants to violate basedpyright's
    ``reportIncompatibleVariableOverride`` rule (mutable invariant type
    narrowed by a Literal override).

    ``TrainingBase`` itself should not be instantiated directly; the factory
    dispatcher takes the union :data:`rux_ml.config.training.TrainingConfig`
    so basedpyright sees ``cfg.kind`` on each variant.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )

    # GPU-first per D3 (user override of CPU-first lean).
    device: Literal["cuda", "cpu"] = "cuda"

    # Eval metric name; concrete metric registry lives in ``rux_ml.training.metrics``.
    metric: str = "auc"

    # Family-agnostic concept; per-family factories translate to the upstream
    # library's keyword (XGBoost: ``early_stopping_rounds``; LightGBM:
    # ``early_stopping_round``; CatBoost: ``od_wait``).
    early_stopping_rounds: int | None = 50

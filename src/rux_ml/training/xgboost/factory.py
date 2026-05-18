"""``make_xgboost_trainer`` — XGBoost-specific factory (per PR-017).

Moved verbatim from the pre-PR-017 ``src/rux_ml/training/factory.py``, narrowed
to the XGBoost variant of the discriminated ``TrainingConfig``. The top-level
``rux_ml.training.factory.make_trainer`` dispatches into this function via
``TRAINER_FAMILIES["xgboost"]``.

The task (classifier vs regressor) is derived from ``cfg.metric`` via the
metric registry — keeping task off the config itself prevents metric+task
from drifting apart (they're the same decision).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from xgboost import XGBClassifier, XGBRegressor

from rux_ml.training.metrics import task_for_metric

if TYPE_CHECKING:
    from rux_ml.training.protocol import Trainer
    from rux_ml.training.xgboost.config import XGBoostTraining


def _xgb_kwargs(cfg: XGBoostTraining, *, seed: int | None) -> dict[str, object]:
    """Assemble the XGBoost kwargs from ``XGBoostTraining`` fields + model_kwargs.

    ``model_kwargs`` always wins on key collisions so callers can override
    top-level defaults from TOML without having to add new ``XGBoostTraining``
    fields.

    ``seed`` (PR-013): when not None, sets XGBoost's ``random_state``. The
    sklearn wrapper threads this into the booster's RNG for bootstrap
    sampling, column subsampling, and any other internal randomness so two
    fits with the same seed (and ``tree_method="hist"`` + single thread on
    CPU) produce bit-exact predictions. GPU `hist` is near-deterministic
    only (per D9 + CONSTRAINTS.md tolerance-based golden-tests rule).
    """
    base: dict[str, object] = {
        "device": cfg.device,
        "tree_method": cfg.tree_method,
        "enable_categorical": cfg.enable_categorical,
        "learning_rate": cfg.learning_rate,
        "max_depth": cfg.max_depth,
        "n_estimators": cfg.n_estimators,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "early_stopping_rounds": cfg.early_stopping_rounds,
        "eval_metric": cfg.metric,
    }
    if seed is not None:
        base["random_state"] = seed
    base.update(cfg.model_kwargs)
    return base


def make_xgboost_trainer(cfg: XGBoostTraining, *, seed: int | None = None) -> Trainer:
    """Return the concrete XGBoost trainer for ``cfg``.

    The task (classifier vs regressor) is derived from ``cfg.metric`` via the
    metric registry so the user-visible config has a single source of truth.

    ``seed`` (PR-013): plumbs ``SeedBag.xgb_seed`` into XGBoost's
    ``random_state``. When ``None``, XGBoost defaults to its internal
    nondeterministic RNG. ``model_kwargs`` in the config takes precedence
    over the ``seed`` argument on key collision (matches PR-006's existing
    "model_kwargs wins" convention for direct user overrides).
    """
    kwargs = _xgb_kwargs(cfg, seed=seed)
    task = task_for_metric(cfg.metric)
    # XGBoost stubs don't expose ``best_iteration_`` (set at runtime) so the
    # structural match against the minimal Trainer Protocol needs a cast here.
    if task == "classification":
        return cast("Trainer", XGBClassifier(**kwargs))
    return cast("Trainer", XGBRegressor(**kwargs))

"""``make_trainer`` factory — pick the concrete XGBoost estimator per ``TrainingConfig``.

Per D5: the contract across families is the sklearn estimator API; the factory
returns a concrete class whose surface matches :class:`Trainer` structurally.

``cfg.kind`` selects the family ("xgboost" today); the **task** (classification
vs regression) is derived from ``cfg.metric`` via the metric registry — AUC /
logloss → ``XGBClassifier``; RMSE / MAE → ``XGBRegressor``. Keeping task off
``TrainingConfig`` itself prevents the metric and task from drifting apart
(which they shouldn't — they're the same decision).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from xgboost import XGBClassifier, XGBRegressor

from rux_ml.training.metrics import task_for_metric

if TYPE_CHECKING:
    from rux_ml.config import TrainingConfig
    from rux_ml.training.protocol import Trainer


def _xgb_kwargs(cfg: TrainingConfig, *, seed: int | None) -> dict[str, object]:
    """Assemble the XGBoost kwargs from TrainingConfig top-level fields + model_kwargs.

    ``model_kwargs`` always wins on key collisions so callers can override
    top-level defaults from TOML without having to add new TrainingConfig fields.

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


def make_trainer(cfg: TrainingConfig, *, seed: int | None = None) -> Trainer:
    """Return the concrete trainer for ``cfg``.

    ``cfg.kind`` selects the family (only ``"xgboost"`` implemented in v0); the
    task (classifier vs regressor) is derived from ``cfg.metric`` so the
    user-visible config has a single source of truth.

    ``seed`` (PR-013): plumbs ``SeedBag.xgb_seed`` into XGBoost's
    ``random_state``. When ``None``, XGBoost defaults to its internal
    nondeterministic RNG. ``model_kwargs`` in the config takes precedence
    over the ``seed`` argument on key collision (matches PR-006's existing
    "model_kwargs wins" convention for direct user overrides).

    Raises:
        NotImplementedError: when ``cfg.kind`` is set to a family beyond
            ``"xgboost"`` — those land in their own follow-up PRs (per D5).
    """
    if cfg.kind != "xgboost":
        msg = (
            f"training.kind={cfg.kind!r} is declared in TrainingConfig but only "
            f"'xgboost' is implemented in v0; LightGBM/CatBoost/sklearn families "
            f"land in their own follow-up PRs."
        )
        raise NotImplementedError(msg)

    kwargs = _xgb_kwargs(cfg, seed=seed)
    task = task_for_metric(cfg.metric)
    # XGBoost stubs don't expose ``best_iteration_`` (set at runtime) so the
    # structural match against the minimal Trainer Protocol needs a cast here.
    if task == "classification":
        return cast("Trainer", XGBClassifier(**kwargs))
    return cast("Trainer", XGBRegressor(**kwargs))

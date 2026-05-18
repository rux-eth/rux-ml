"""``make_catboost_trainer`` — CatBoost-specific factory (per PR-019).

Dispatched by ``rux_ml.training.factory.make_trainer`` via the
``TRAINER_FAMILIES["catboost"]`` registry entry. Task (classifier vs
regressor) derived from ``cfg.metric`` via the metric registry.

Four Q-finding-driven implementation details:

1. **Metric translation** (Q-Wrap). CatBoost's metric names are case-
   sensitive and differ from the workbench's: ``auc → AUC``, ``logloss →
   Logloss``, ``rmse → RMSE``, ``mae → MAE``. ``_METRIC_TRANSLATE`` covers
   the rename. Note: ``AUC`` is a valid ``eval_metric`` but NOT a valid
   ``loss_function`` — the factory derives ``loss_function`` from task
   (``Logloss`` for classifier; ``RMSE`` for regressor).

2. **GPU is the default** (Q-GPU). CatBoost ships prebuilt CUDA wheels via
   ``uv add catboost`` (no extras, no source build, no container delta).
   ``cfg.device == "cuda"`` → ``task_type="GPU"`` + ``devices="0"`` for
   single-GPU. ``cfg.device == "cpu"`` → ``task_type="CPU"``.

3. **Explicit ``thread_count``** (Q-Parallel). CatBoost uses Intel TBB,
   NOT OpenMP. The workbench's ``OMP_NUM_THREADS`` pinning is invisible to
   CatBoost. The factory passes ``cfg.memory.omp_threads`` explicitly as
   ``thread_count=`` to match the workbench's intent. (Per Phase 4 sub-
   decision 2: reuse existing field rather than add a parallel knob.)

4. **`cat_features` extraction at fit time** (Q-Cat). CatBoost does NOT
   auto-detect pandas Categorical dtype (opposite of LightGBM); raises
   error on category-dtype columns not in ``cat_features=``. The
   ``_CatBoostTrainerShim`` Pattern-A wrapper extracts categorical column
   names from the input DataFrame at ``fit()`` time and threads them
   through to the underlying estimator. ``_ColumnRouter`` is unchanged.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

from rux_ml.training.metrics import task_for_metric

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from rux_ml.training.catboost.config import CatBoostTraining
    from rux_ml.training.protocol import Trainer


# Workbench metric names → CatBoost metric names (case-sensitive).
_METRIC_TRANSLATE = {
    "auc": "AUC",
    "logloss": "Logloss",
    "rmse": "RMSE",
    "mae": "MAE",
}


def _extract_categorical_columns(x: pd.DataFrame) -> list[str]:
    """Return the names of pandas Categorical-dtype columns in ``x``.

    Wrapped as a named helper so the basedpyright ignore on pandas type
    stubs (``.select_dtypes`` returns ``Unknown`` columns in their stubs)
    is scoped tightly to one function. Used only by the
    :class:`_CatBoostTrainerShim` to populate ``cat_features=`` at fit time
    (Q-Cat: CatBoost does NOT auto-detect Categorical dtype, opposite of
    LightGBM).
    """
    cat_df: pd.DataFrame = x.select_dtypes(include="category")  # pyright: ignore[reportUnknownMemberType]
    return [str(c) for c in cat_df.columns]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]


class _CatBoostTrainerShim:
    """Pattern-A shim: wraps a CatBoost estimator + injects ``cat_features`` at fit time.

    CatBoost rejects pandas Categorical-dtype columns that aren't listed in
    ``cat_features=`` (Q-Cat finding: issue #757, FR #1386 still open). The
    shim's ``fit(X, y, **kwargs)`` extracts column names with Categorical
    dtype from ``X`` and threads them through as ``cat_features=`` to the
    underlying estimator's ``fit()``.

    Non-DataFrame inputs (e.g., numpy arrays) skip the extraction — the
    workbench's features layer outputs pandas DataFrames per
    ``_polars_select_then_pandas``, so this is the happy path; the guard
    just keeps the shim safe for direct numpy callers (e.g., the
    conformance test using ``make_classification``).

    Forwards everything else (``predict_proba``, ``best_iteration_``,
    ``get_params``, ``get_feature_importance``, ...) to the underlying
    estimator via ``__getattr__``. Mirrors PR-018's ``_LightGBMTrainerShim``.
    """

    def __init__(self, base: Any) -> None:
        self._base = base

    def fit(self, x: Any, y: ArrayLike, **kwargs: Any) -> _CatBoostTrainerShim:
        if isinstance(x, pd.DataFrame) and "cat_features" not in kwargs:
            cat_cols = _extract_categorical_columns(x)
            if cat_cols:
                kwargs["cat_features"] = cat_cols
        self._base.fit(x, y, **kwargs)
        return self

    def predict(self, x: Any) -> ArrayLike:
        return cast("ArrayLike", self._base.predict(x))

    def __getattr__(self, name: str) -> Any:
        # Forwards predict_proba, best_iteration_, get_params, etc to the
        # underlying CatBoost estimator. __getattr__ is only called when
        # normal attribute lookup fails — methods/attrs defined directly on
        # the shim take precedence.
        return getattr(self._base, name)


def _catboost_kwargs(
    cfg: CatBoostTraining, *, seed: int | None, thread_count: int
) -> dict[str, Any]:
    """Assemble the CatBoost kwargs from ``CatBoostTraining`` fields + factory args.

    Per Q-Wrap research at v1.2.10 source (core.py L5305-5425). CatBoost has no
    ``**kwargs`` escape hatch, so every kwarg here must be a known CatBoost
    parameter — basedpyright/ruff catch typos because there's no permissive
    fall-through.

    Derived fields:
    - ``task_type``: ``"GPU"`` when ``cfg.device == "cuda"``, else ``"CPU"``
      (Q-GPU). ``devices="0"`` when GPU for single-GPU workbench.
    - ``loss_function``: task-derived (``Logloss`` classifier; ``RMSE``
      regressor). NOT exposed as a config field because ``AUC`` is
      eval-only (Q-Wrap §4).
    - ``eval_metric``: ``_METRIC_TRANSLATE[cfg.metric]``.
    - ``thread_count``: factory arg (sourced from ``cfg.memory.omp_threads``
      by the dispatcher in higher layers; Q-Parallel sub-decision).
    """
    task = task_for_metric(cfg.metric)
    eval_metric = _METRIC_TRANSLATE[cfg.metric]
    loss_function = "Logloss" if task == "classification" else "RMSE"

    base: dict[str, Any] = {
        # Boosting + capacity
        "iterations": cfg.iterations,
        "learning_rate": cfg.learning_rate,
        "depth": cfg.depth,
        "border_count": cfg.border_count,
        # Regularization
        "l2_leaf_reg": cfg.l2_leaf_reg,
        "random_strength": cfg.random_strength,
        # Boosting strategy
        "bootstrap_type": cfg.bootstrap_type,
        "grow_policy": cfg.grow_policy,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "one_hot_max_size": cfg.one_hot_max_size,
        # Loss + metric (derived)
        "loss_function": loss_function,
        "eval_metric": eval_metric,
        # Family-agnostic plumbing
        "task_type": "GPU" if cfg.device == "cuda" else "CPU",
        "thread_count": thread_count,
        # Factory-injected defaults
        "verbose": False,
        "allow_writing_files": False,  # workbench owns artifacts (PR-010 registry)
    }

    # bootstrap_type ↔ randomness-kwarg interaction (CatBoost rejects mismatches):
    # - Bayesian: accepts ``bagging_temperature``, rejects ``subsample``.
    # - Bernoulli / MVS / Poisson: accept ``subsample``, reject ``bagging_temperature``.
    # - No: rejects both.
    if cfg.bootstrap_type == "Bayesian":
        base["bagging_temperature"] = cfg.bagging_temperature
    elif cfg.bootstrap_type in ("Bernoulli", "MVS", "Poisson"):
        base["subsample"] = cfg.subsample

    if cfg.device == "cuda":
        # Single-GPU workbench; pin device index per Q-GPU recommendation
        # to avoid the #2649 multi-GPU memory-leak footgun if a second GPU
        # is ever added to the host.
        base["devices"] = "0"

    if cfg.early_stopping_rounds is not None:
        # CatBoost accepts ``early_stopping_rounds`` as a top-level
        # constructor kwarg (Q-Wrap §2; core.py L5396 + docstring L2866-7).
        # Activates Iter overfitting detector with od_wait set to N. Simpler
        # than LightGBM (which needs a fit-time callback) or the
        # od_type+od_wait pair.
        base["early_stopping_rounds"] = cfg.early_stopping_rounds

    if seed is not None:
        base["random_seed"] = seed

    return base


def _resolve_thread_count() -> int:
    """Sample the workbench's thread-pin intent from ``OMP_NUM_THREADS`` env var.

    CatBoost uses Intel TBB and does NOT honor ``OMP_NUM_THREADS`` directly
    (Q-Parallel PROVEN). PR-011's ``pin_threads()`` exports the env var from
    ``cfg.memory.omp_threads`` into the trial subprocess; the CatBoost
    factory reads it back as an integer and passes through to
    ``thread_count=`` explicitly. Same intent, different transport.

    Falls back to ``-1`` (CatBoost auto = all logical cores) when the env
    var is unset or unparseable — matches PR-011's default behavior when
    pinning isn't active.
    """
    raw = os.environ.get("OMP_NUM_THREADS")
    if raw is None:
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def make_catboost_trainer(
    cfg: CatBoostTraining,
    *,
    seed: int | None = None,
) -> Trainer:
    """Return the concrete CatBoost trainer for ``cfg`` wrapped in the cat_features shim.

    Args:
        cfg: ``CatBoostTraining`` variant of the discriminated ``TrainingConfig``.
        seed: PR-013 seed (factory plumbs to ``random_seed``).

    Thread count is read from the ``OMP_NUM_THREADS`` env var (set by PR-011's
    ``pin_threads()`` from ``cfg.memory.omp_threads``) and passed explicitly
    to CatBoost since it doesn't honor ``OMP_NUM_THREADS`` itself.
    """
    from catboost import CatBoostClassifier, CatBoostRegressor  # noqa: PLC0415

    thread_count = _resolve_thread_count()
    kwargs = _catboost_kwargs(cfg, seed=seed, thread_count=thread_count)
    task = task_for_metric(cfg.metric)
    base = CatBoostClassifier(**kwargs) if task == "classification" else CatBoostRegressor(**kwargs)
    return cast("Trainer", _CatBoostTrainerShim(base))

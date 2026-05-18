"""``make_lightgbm_trainer`` — LightGBM-specific factory (per PR-018).

Dispatched by ``rux_ml.training.factory.make_trainer`` via the
``TRAINER_FAMILIES["lightgbm"]`` registry entry. The task (classifier vs
regressor) is derived from ``cfg.metric`` via the metric registry — keeping
task off the config itself so it can't drift away from the metric.

Three Q-Wrap-driven implementation details:

1. **Metric translation** — LightGBM rejects the string ``"logloss"`` as a
   ``metric=`` value; the canonical LightGBM name is ``"binary_logloss"``.
   ``_METRIC_TRANSLATE`` covers the rename. Other workbench metrics
   (``auc``, ``rmse``, ``mae``) pass through unchanged.
2. **GPU deferred** — when ``cfg.device == "cuda"`` the factory raises
   ``NotImplementedError`` (Q-GPU: LightGBM-CUDA is 8-28x slower than
   XGBoost-CUDA on workbench-scale data; ship CPU-only and revisit when
   prereqs land). CPU is the only supported path in PR-018.
3. **Early-stopping injection at fit time** — LightGBM v4.5+ takes
   ``early_stopping_rounds`` as a fit-time callback (``callbacks=[lightgbm
   .early_stopping(N)]``), NOT a constructor kwarg. Pattern A (per Phase 4
   sub-decision): the trainer is a thin shim whose ``fit()`` injects the
   callback automatically when ``cfg.early_stopping_rounds`` is set and the
   caller passes ``eval_set=...``. Caller-side code (cli/train.py,
   tuning/objective.py) is unchanged from the XGBoost path.

Pattern-A shim preserves the ``Trainer`` Protocol surface (``fit``, ``predict``)
and forwards everything else (``predict_proba``, ``get_params``,
``best_iteration_``, ...) via ``__getattr__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from rux_ml.training.metrics import task_for_metric

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from rux_ml.training.lightgbm.config import LightGBMTraining
    from rux_ml.training.protocol import Trainer

# LightGBM rejects "logloss" as a metric name; canonical is "binary_logloss".
# Other workbench metrics (auc, rmse, mae) pass through unchanged.
_METRIC_TRANSLATE = {"logloss": "binary_logloss"}


class _LightGBMTrainerShim:
    """Pattern-A shim: wraps an LGBM estimator + injects early-stopping callback.

    The shim exposes the :class:`Trainer` Protocol surface (``fit`` + ``predict``)
    and forwards everything else (``predict_proba``, ``best_iteration_``,
    ``get_params``, ...) to the wrapped LightGBM estimator via ``__getattr__``.

    When the caller invokes ``fit(X, y, eval_set=[...])`` and the trainer was
    constructed with ``cfg.early_stopping_rounds`` set, the shim appends a
    ``lightgbm.early_stopping(stopping_rounds)`` callback to whatever
    ``callbacks=`` the caller passes (or starts a fresh list). LightGBM's
    early-stopping callback observes the eval_set and stops training when the
    primary metric stops improving.
    """

    def __init__(self, base: Any, *, early_stopping_rounds: int | None) -> None:
        self._base = base
        self._early_stopping_rounds = early_stopping_rounds

    def fit(self, x: Any, y: ArrayLike, **kwargs: Any) -> _LightGBMTrainerShim:
        if self._early_stopping_rounds is not None and "eval_set" in kwargs:
            import lightgbm  # noqa: PLC0415 — lazy import (extra-gated)

            callbacks = list(kwargs.pop("callbacks", []))
            callbacks.append(lightgbm.early_stopping(self._early_stopping_rounds))
            kwargs["callbacks"] = callbacks
        self._base.fit(x, y, **kwargs)
        return self

    def predict(self, x: Any) -> ArrayLike:
        return cast("ArrayLike", self._base.predict(x))

    def __getattr__(self, name: str) -> Any:
        # Forwards everything else (predict_proba, get_params,
        # best_iteration_, get_booster, ...) to the underlying LightGBM
        # estimator. ``__getattr__`` is only called when normal attribute
        # lookup fails — methods/attrs defined directly on the shim take
        # precedence.
        return getattr(self._base, name)


def _lgbm_kwargs(cfg: LightGBMTraining, *, seed: int | None) -> dict[str, Any]:
    """Assemble the LightGBM kwargs from ``LightGBMTraining`` fields + model_kwargs.

    Field translation table (rux-ml field → LightGBM kwarg) — see Q-Wrap research.
    ``model_kwargs`` always wins on key collisions so users can override typed
    fields from TOML without adding new ``LightGBMTraining`` fields.

    ``seed`` (PR-013): when not None, sets LightGBM's ``random_state``. PR-013
    CPU bit-exact contract additionally requires ``cfg.deterministic = True``
    (LightGBM's ``random_state`` has lower priority than per-feature seeds
    when ``deterministic`` is False — see Q-Wrap §6).
    """
    metric = _METRIC_TRANSLATE.get(cfg.metric, cfg.metric)
    base: dict[str, Any] = {
        # Boosting + capacity
        "boosting_type": cfg.boosting_type,
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        # Regularization (LightGBM sklearn-wrapper alias names where present)
        "min_child_samples": cfg.min_data_in_leaf,  # sklearn alias
        "subsample": cfg.bagging_fraction,  # sklearn alias of bagging_fraction
        "subsample_freq": cfg.bagging_freq,  # alias of bagging_freq
        "colsample_bytree": cfg.feature_fraction,  # alias of feature_fraction
        "reg_alpha": cfg.lambda_l1,
        "reg_lambda": cfg.lambda_l2,
        # Native kwargs that flow through LGBMModel **kwargs
        "metric": metric,
        "deterministic": cfg.deterministic,
        # CPU-only in PR-018; cuda raises in make_lightgbm_trainer
        "device_type": "cpu",
    }
    if seed is not None:
        base["random_state"] = seed
    base.update(cfg.model_kwargs)
    return base


def make_lightgbm_trainer(cfg: LightGBMTraining, *, seed: int | None = None) -> Trainer:
    """Return the concrete LightGBM trainer for ``cfg`` wrapped in the early-stopping shim.

    The task (classifier vs regressor) is derived from ``cfg.metric``:
    ``auc`` / ``logloss`` → ``LGBMClassifier``; ``rmse`` / ``mae`` →
    ``LGBMRegressor``.

    Raises:
        NotImplementedError: when ``cfg.device == "cuda"``. LightGBM-GPU
            support is deferred per Q-GPU research (Microsoft + szilard
            benchmarks: LightGBM-GPU is 8-28x slower than XGBoost-GPU on
            workbench-scale tabular workloads; install path via ``uv`` is
            non-trivial — see prs/PR-018-lightgbm-family.md Research findings).
    """
    if cfg.device == "cuda":
        msg = (
            "LightGBM GPU support deferred — see Q-GPU research in "
            "prs/PR-018-lightgbm-family.md. Use device='cpu' or switch "
            "family to xgboost. (Reason: LightGBM-GPU is 8-28x slower than "
            "XGBoost-GPU on workbench-scale data per szilard/GBM-perf "
            "benchmarks; install path via uv is non-trivial — no prebuilt "
            "CUDA wheel on PyPI.)"
        )
        raise NotImplementedError(msg)

    from lightgbm import LGBMClassifier, LGBMRegressor  # noqa: PLC0415

    kwargs = _lgbm_kwargs(cfg, seed=seed)
    task = task_for_metric(cfg.metric)
    base = LGBMClassifier(**kwargs) if task == "classification" else LGBMRegressor(**kwargs)
    return cast(
        "Trainer",
        _LightGBMTrainerShim(base, early_stopping_rounds=cfg.early_stopping_rounds),
    )

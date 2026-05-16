# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Metric registry + Optuna direction lookup (per PR-006 scope).

A single map from metric name → callable + meta keeps PR-006's scoring path
small and testable. The callable signature is intentionally narrow: it takes
the fitted trainer plus the eval pair, since AUC/logloss need ``predict_proba``
and RMSE/MAE need ``predict`` — picking the right one belongs in the registry,
not in the caller.

The file-level ``pyright`` pragma relaxes ``reportUnknownVariableType`` /
``reportUnknownArgumentType`` because ``sklearn.metrics`` ships parameter
stubs typed as ``Unknown`` across every scorer — the return types are well-
defined (each scorer returns ``float`` or a near-numeric union), so the call
sites' ``float(...)`` conversions remain explicit and correct. Library code
elsewhere in ``src/`` stays under the strict default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from sklearn.metrics import (
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    from rux_ml.training.protocol import Trainer

Task = Literal["classification", "regression"]
Direction = Literal["maximize", "minimize"]


class _ScoreFn(Protocol):
    """Score signature: ``(trainer, X_eval, y_eval) -> float``."""

    def __call__(
        self,
        trainer: Trainer,
        x_eval: Any,
        y_eval: ArrayLike,
    ) -> float: ...


@dataclass(frozen=True)
class MetricSpec:
    """One row of the metric registry."""

    name: str
    task: Task
    direction: Direction
    score: _ScoreFn


def _predict_proba(trainer: Trainer, x_eval: Any) -> NDArray[Any]:
    """Cross-family narrowing: classifier-only attribute, accessed defensively.

    Trainer Protocol covers ``fit`` + ``predict`` only; ``predict_proba`` lives
    on classifiers. The metric registry calls this helper exclusively from the
    classification metrics, so the cast is safe by construction.
    """
    proba_fn = cast("Any", trainer).predict_proba
    return cast("NDArray[Any]", proba_fn(x_eval))


def _auc(trainer: Trainer, x_eval: Any, y_eval: ArrayLike) -> float:
    proba = _predict_proba(trainer, x_eval)
    # Binary case: roc_auc_score expects scores for the positive class.
    proba_arr = proba if proba.ndim == 1 else proba[:, 1]
    return float(roc_auc_score(y_eval, proba_arr))


def _logloss(trainer: Trainer, x_eval: Any, y_eval: ArrayLike) -> float:
    proba = _predict_proba(trainer, x_eval)
    return float(log_loss(y_eval, proba))


def _rmse(trainer: Trainer, x_eval: Any, y_eval: ArrayLike) -> float:
    preds = trainer.predict(x_eval)
    # sklearn>=1.6 has root_mean_squared_error; sqrt(MSE) is universal.
    return float(math.sqrt(mean_squared_error(y_eval, preds)))


def _mae(trainer: Trainer, x_eval: Any, y_eval: ArrayLike) -> float:
    preds = trainer.predict(x_eval)
    return float(mean_absolute_error(y_eval, preds))


METRIC_REGISTRY: dict[str, MetricSpec] = {
    "auc": MetricSpec("auc", "classification", "maximize", _auc),
    "logloss": MetricSpec("logloss", "classification", "minimize", _logloss),
    "rmse": MetricSpec("rmse", "regression", "minimize", _rmse),
    "mae": MetricSpec("mae", "regression", "minimize", _mae),
}


def _lookup(name: str) -> MetricSpec:
    try:
        return METRIC_REGISTRY[name]
    except KeyError as exc:
        known = sorted(METRIC_REGISTRY)
        msg = f"unknown metric {name!r}; known: {known}"
        raise ValueError(msg) from exc


def compute_score(metric: str, trainer: Trainer, x_eval: Any, y_eval: ArrayLike) -> float:
    """Score the trainer on ``(x_eval, y_eval)`` using the named metric."""
    return _lookup(metric).score(trainer, x_eval, y_eval)


def task_for_metric(metric: str) -> Task:
    """Return ``"classification"`` or ``"regression"`` for the named metric."""
    return _lookup(metric).task


def optuna_direction(metric: str) -> Direction:
    """Return ``"maximize"`` or ``"minimize"`` for the named metric.

    Used by the CLI to pick the study direction when creating the 1-trial
    Optuna study (per D7 / PR-006).
    """
    return _lookup(metric).direction

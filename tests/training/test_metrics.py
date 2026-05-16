"""Unit tests for the metric registry + Optuna direction lookup."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rux_ml.training.metrics import (
    METRIC_REGISTRY,
    compute_score,
    optuna_direction,
    task_for_metric,
)


def test_registry_keys_match_spec() -> None:
    assert set(METRIC_REGISTRY) == {"auc", "logloss", "rmse", "mae"}


@pytest.mark.parametrize(
    ("metric", "expected_task"),
    [
        ("auc", "classification"),
        ("logloss", "classification"),
        ("rmse", "regression"),
        ("mae", "regression"),
    ],
)
def test_task_for_metric(metric: str, expected_task: str) -> None:
    assert task_for_metric(metric) == expected_task


@pytest.mark.parametrize(
    ("metric", "expected_direction"),
    [("auc", "maximize"), ("logloss", "minimize"), ("rmse", "minimize"), ("mae", "minimize")],
)
def test_optuna_direction(metric: str, expected_direction: str) -> None:
    assert optuna_direction(metric) == expected_direction


def test_unknown_metric_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown metric 'bogus'"):
        task_for_metric("bogus")
    with pytest.raises(ValueError, match="unknown metric 'bogus'"):
        optuna_direction("bogus")


class _PerfectBinaryClassifier:
    """Stub that returns probabilities equal to the true labels — AUC = 1.0."""

    best_iteration_: int | None = None

    def fit(self, X: object, y: object, **_kwargs: object) -> _PerfectBinaryClassifier:
        _ = (X, y)
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.asarray(self._cached)

    def predict_proba(self, X: object) -> np.ndarray:
        _ = X
        # Two-column probabilities for the binary-classification path.
        p1 = np.asarray(self._cached, dtype=float)
        return np.stack([1.0 - p1, p1], axis=1)

    def __init__(self, cached: np.ndarray) -> None:
        self._cached = cached


class _ConstantRegressor:
    """Stub that always predicts a constant value (used for RMSE/MAE tests)."""

    best_iteration_: int | None = None

    def __init__(self, value: float) -> None:
        self._value = value

    def fit(self, X: object, y: object, **_kwargs: object) -> _ConstantRegressor:
        _ = (X, y)
        return self

    def predict(self, X: object) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)  # type: ignore[arg-type]
        return np.full(n, self._value, dtype=float)

    def predict_proba(self, X: object) -> np.ndarray:
        _ = X
        msg = "regressor has no predict_proba"
        raise NotImplementedError(msg)


def test_compute_score_auc_perfect_classifier() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    clf = _PerfectBinaryClassifier(cached=y_true.astype(float))
    score = compute_score("auc", clf, x_eval=np.zeros((len(y_true), 1)), y_eval=y_true)
    assert math.isclose(score, 1.0)


def test_compute_score_rmse_constant_prediction() -> None:
    y_true = np.array([1.0, 3.0, 5.0])  # mean = 3.0
    reg = _ConstantRegressor(value=3.0)
    score = compute_score("rmse", reg, x_eval=np.zeros((3, 1)), y_eval=y_true)
    # rmse = sqrt(((1-3)^2 + 0 + (5-3)^2) / 3) = sqrt(8/3)
    assert math.isclose(score, math.sqrt(8.0 / 3.0), rel_tol=1e-9)


def test_compute_score_mae_constant_prediction() -> None:
    y_true = np.array([1.0, 3.0, 5.0])
    reg = _ConstantRegressor(value=3.0)
    score = compute_score("mae", reg, x_eval=np.zeros((3, 1)), y_eval=y_true)
    # mae = (|1-3| + 0 + |5-3|) / 3 = 4/3
    assert math.isclose(score, 4.0 / 3.0, rel_tol=1e-9)

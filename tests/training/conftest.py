"""Fixtures for the training-layer unit tests."""

from __future__ import annotations

import pytest

from rux_ml.training import XGBoostTraining


@pytest.fixture
def training_cfg_classification() -> XGBoostTraining:
    """Tiny classification-friendly training config (CPU, fast)."""
    return XGBoostTraining(
        device="cpu",
        metric="auc",
        n_estimators=8,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )


@pytest.fixture
def training_cfg_regression() -> XGBoostTraining:
    """Tiny regression-friendly training config (CPU, fast)."""
    return XGBoostTraining(
        device="cpu",
        metric="rmse",
        n_estimators=8,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )

"""Fixtures for the training-layer unit tests."""

from __future__ import annotations

import pytest

from rux_ml.config import TrainingConfig


@pytest.fixture
def training_cfg_classification() -> TrainingConfig:
    """Tiny classification-friendly training config (CPU, fast)."""
    return TrainingConfig(
        device="cpu",
        metric="auc",
        n_estimators=8,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )


@pytest.fixture
def training_cfg_regression() -> TrainingConfig:
    """Tiny regression-friendly training config (CPU, fast)."""
    return TrainingConfig(
        device="cpu",
        metric="rmse",
        n_estimators=8,
        max_depth=3,
        learning_rate=0.3,
        early_stopping_rounds=None,
    )

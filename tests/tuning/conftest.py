"""Shared fixtures for tuning-layer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from rux_ml.config import (
    CatSpec,
    DataConfig,
    FloatSpec,
    IntSpec,
    KFoldCV,
    RuxMLConfig,
    SearchSpec,
    TuningConfig,
    XGBoostTraining,
)
from rux_ml.config.features import FeaturesConfig, FeaturesSpec


@pytest.fixture
def synth_parquet(tmp_path: Path) -> Path:
    """Tiny separable binary classification dataset."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    p = tmp_path / "synth.parquet"
    pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}).write_parquet(p)
    return p


@pytest.fixture
def tune_cfg(synth_parquet: Path) -> RuxMLConfig:
    """Fast tuning config: CPU, tiny model, K=3 folds."""
    search_space: dict[str, SearchSpec] = {
        "training.learning_rate": FloatSpec(low=0.05, high=0.5, log=False),
        "training.max_depth": IntSpec(low=2, high=4),
        "training.subsample": FloatSpec(low=0.7, high=1.0),
        "training.colsample_bytree": CatSpec(choices=[0.8, 1.0]),
    }
    return RuxMLConfig(
        data=DataConfig(source_path=synth_parquet, target_column="y"),
        features=FeaturesConfig(
            spec=FeaturesSpec(numeric_columns=["x1", "x2"], categorical_columns=[]),
        ),
        training=XGBoostTraining(
            device="cpu",
            metric="auc",
            n_estimators=8,
            max_depth=3,
            learning_rate=0.3,
            early_stopping_rounds=None,
        ),
        tuning=TuningConfig(
            sampler="tpe",
            pruner="wilcoxon",
            n_trials=2,
            n_startup_trials=1,
            entropy=42,
        ),
        cv=KFoldCV(n_splits=3, shuffle=True),
        search_space=search_space,
    )

"""Fixtures for features-layer tests."""

from __future__ import annotations

import polars as pl
import pytest

from rux_ml.config import FeaturesConfig, FeaturesSpec


@pytest.fixture
def tiny_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "low_cat": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            # 10 distinct values across 10 rows -> high cardinality for our 3-threshold
            "high_cat": [f"u{i}" for i in range(10)],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def features_cfg_low_threshold() -> FeaturesConfig:
    """Threshold=3 → low_cat (2 unique) passes through; high_cat (10 unique) gets encoded."""
    return FeaturesConfig(
        categorical_low_card_threshold=3,
        spec=FeaturesSpec(
            numeric_columns=["x1", "x2"],
            categorical_columns=["low_cat", "high_cat"],
        ),
    )


@pytest.fixture
def features_cfg_numeric_only() -> FeaturesConfig:
    return FeaturesConfig(
        categorical_low_card_threshold=10,
        spec=FeaturesSpec(numeric_columns=["x1", "x2"]),
    )

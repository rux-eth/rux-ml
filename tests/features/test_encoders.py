"""Categorical encoder decision-rule tests (per D4)."""

from __future__ import annotations

import pytest
from category_encoders.wrapper import NestedCVWrapper

from rux_ml.config import FeaturesConfig
from rux_ml.features.encoders import (
    PASSTHROUGH_TO_XGB_CATEGORICAL,
    make_categorical_encoder,
)


def test_low_cardinality_returns_passthrough() -> None:
    cfg = FeaturesConfig(categorical_low_card_threshold=10)
    encoder = make_categorical_encoder("cat", cardinality=5, cfg=cfg)
    assert encoder == PASSTHROUGH_TO_XGB_CATEGORICAL


def test_high_cardinality_returns_nestedcv_wrapper() -> None:
    cfg = FeaturesConfig(categorical_low_card_threshold=10)
    encoder = make_categorical_encoder("cat", cardinality=500, cfg=cfg)
    assert isinstance(encoder, NestedCVWrapper)


def test_at_threshold_returns_passthrough() -> None:
    """Threshold is inclusive (cardinality ≤ threshold → passthrough)."""
    cfg = FeaturesConfig(categorical_low_card_threshold=10)
    encoder = make_categorical_encoder("cat", cardinality=10, cfg=cfg)
    assert encoder == PASSTHROUGH_TO_XGB_CATEGORICAL


def test_just_above_threshold_returns_encoder() -> None:
    cfg = FeaturesConfig(categorical_low_card_threshold=10)
    encoder = make_categorical_encoder("cat", cardinality=11, cfg=cfg)
    assert isinstance(encoder, NestedCVWrapper)


def test_missing_threshold_raises() -> None:
    """D4 BEST-GUESS: no default — must be set deliberately."""
    cfg = FeaturesConfig(categorical_low_card_threshold=None)
    with pytest.raises(ValueError, match="no default"):
        make_categorical_encoder("cat", cardinality=5, cfg=cfg)

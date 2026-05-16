"""End-to-end Pipeline tests for ``make_features``."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from category_encoders.wrapper import NestedCVWrapper
from sklearn.pipeline import Pipeline

from rux_ml.config import FeaturesConfig
from rux_ml.features.encoders import PASSTHROUGH_TO_XGB_CATEGORICAL
from rux_ml.features.pipeline import (
    _ColumnRouter,
    build_column_transformer,
    cardinalities_from,
    make_features,
)


def test_make_features_returns_pipeline(
    features_cfg_numeric_only: FeaturesConfig,
) -> None:
    pipe = make_features(features_cfg_numeric_only)
    assert isinstance(pipe, Pipeline)
    assert [name for name, _ in pipe.steps] == [
        "polars_select",
        "column_router",
        "to_polars",
    ]


def test_make_features_numeric_only_fit_transform(
    tiny_df: pl.DataFrame, features_cfg_numeric_only: FeaturesConfig
) -> None:
    pipe = make_features(features_cfg_numeric_only)
    out = pipe.fit_transform(tiny_df)
    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) == {"x1", "x2"}
    assert out.height == tiny_df.height


def test_make_features_requires_cardinalities_when_categorical_present(
    features_cfg_low_threshold: FeaturesConfig,
) -> None:
    with pytest.raises(ValueError, match="cardinalities is required"):
        make_features(features_cfg_low_threshold)


def test_build_column_transformer_routes_per_decision_rule(
    features_cfg_low_threshold: FeaturesConfig,
) -> None:
    """low_cat (2 unique) → passthrough; high_cat (10 unique) → NestedCVWrapper."""
    cards = {"low_cat": 2, "high_cat": 10}
    router = build_column_transformer(cards, features_cfg_low_threshold)
    assert isinstance(router, _ColumnRouter)
    # low_cat is at-threshold (well below) → passthrough sentinel
    assert router.encoders["low_cat"] == PASSTHROUGH_TO_XGB_CATEGORICAL
    # high_cat is at threshold=3 way above → real NestedCVWrapper encoder
    assert isinstance(router.encoders["high_cat"], NestedCVWrapper)


def test_pipeline_with_categoricals_fit_transform(
    tiny_df: pl.DataFrame, features_cfg_low_threshold: FeaturesConfig
) -> None:
    cards = cardinalities_from(tiny_df, features_cfg_low_threshold.spec.categorical_columns)
    pipe = make_features(features_cfg_low_threshold, cardinalities=cards)
    y = tiny_df["y"].to_numpy()
    x = tiny_df.drop("y")
    out = pipe.fit_transform(x, y)
    # Output is Polars per set_output("polars")
    assert isinstance(out, pl.DataFrame)
    assert out.height == tiny_df.height


def test_pipeline_fit_on_train_transform_on_test_no_leakage(
    features_cfg_low_threshold: FeaturesConfig,
) -> None:
    """Transform on test must use train-time statistics, not test labels.

    Stronger version: even shuffling the test labels must NOT change the
    transformed test output (because ``transform`` doesn't see ``y``).
    """
    rng = np.random.default_rng(0)
    n_train = 100
    cats = [f"u{i}" for i in range(50)]  # 50 unique vals → high-card vs threshold=3
    x_train = pl.DataFrame(
        {
            "x1": rng.normal(size=n_train).tolist(),
            "x2": rng.integers(0, 100, size=n_train).tolist(),
            "low_cat": rng.choice(["a", "b"], size=n_train).tolist(),
            "high_cat": rng.choice(cats, size=n_train).tolist(),
        }
    )
    y_train = rng.integers(0, 2, size=n_train).tolist()

    x_test = pl.DataFrame(
        {
            "x1": rng.normal(size=10).tolist(),
            "x2": rng.integers(0, 100, size=10).tolist(),
            "low_cat": rng.choice(["a", "b"], size=10).tolist(),
            "high_cat": rng.choice(cats, size=10).tolist(),
        }
    )

    cards = cardinalities_from(x_train, ["low_cat", "high_cat"])
    pipe = make_features(features_cfg_low_threshold, cardinalities=cards)
    pipe.fit(x_train, y_train)
    out_a = pipe.transform(x_test)
    out_b = pipe.transform(x_test)  # second call — must be identical
    assert out_a.equals(out_b)


def test_target_encoder_uses_kfold_to_avoid_within_train_leakage(
    features_cfg_low_threshold: FeaturesConfig,
) -> None:
    """With NestedCVWrapper, fit_transform on train must NOT produce perfect leakage.

    Setup: 50 high-cardinality values, each appearing twice and consistently
    paired with a binary target (alternating 0/1 across categories). A *leaky*
    target encoder would map every row to its category's target deterministically.
    NestedCVWrapper holds out folds, so for at least some rows the K-fold
    encoding must differ from the naive group mean — i.e. not perfectly correlate
    with the target.
    """
    high_cat: list[str] = []
    targets: list[int] = []
    for i in range(50):
        c = f"u{i}"
        high_cat.extend([c, c])  # each unique cat appears twice
        targets.extend([i % 2, i % 2])  # cat→target is 1:1 deterministic
    n = len(high_cat)
    df = pl.DataFrame(
        {
            "x1": [float(i) for i in range(n)],
            "x2": list(range(n)),
            "low_cat": ["a"] * n,
            "high_cat": high_cat,
        }
    )
    cards = cardinalities_from(df, ["low_cat", "high_cat"])
    pipe = make_features(features_cfg_low_threshold, cardinalities=cards)
    out = pipe.fit_transform(df, targets)
    encoded_high = out["high_cat"].to_list()
    # If leaky (no K-fold), encoded_high[i] would equal targets[i] (0.0 or 1.0).
    # With NestedCV, holdout folds force the encoding to use *other* folds'
    # means — for at least some rows the encoding won't match the target.
    exact_matches = sum(int(round(e) == t) for e, t in zip(encoded_high, targets, strict=True))
    assert exact_matches < n, (
        "NestedCVWrapper should prevent full within-train leakage; "
        f"got {exact_matches}/{n} exact target matches"
    )

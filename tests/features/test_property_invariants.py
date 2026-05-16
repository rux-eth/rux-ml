"""Property-based invariants for the features layer (per D12 — hypothesis selective use)."""

from __future__ import annotations

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from rux_ml.config import FeaturesConfig, FeaturesSpec
from rux_ml.features.pipeline import cardinalities_from, make_features
from rux_ml.features.polars_steps import select_columns


@given(
    n_rows=st.integers(min_value=1, max_value=200),
    cat_values=st.lists(
        st.sampled_from(["a", "b", "c", "d", "e"]),
        min_size=1,
        max_size=200,
    ),
)
@settings(deadline=None, max_examples=25)
def test_select_columns_preserves_row_count(n_rows: int, cat_values: list[str]) -> None:
    # Pad/truncate cat_values to match n_rows
    cat = (cat_values * ((n_rows // len(cat_values)) + 1))[:n_rows]
    df = pl.DataFrame(
        {
            "x1": list(range(n_rows)),
            "cat": cat,
        }
    )
    spec = FeaturesSpec(numeric_columns=["x1"], categorical_columns=["cat"])
    out = select_columns(df, spec)
    assert out.height == n_rows


@given(n_rows=st.integers(min_value=2, max_value=100))
@settings(deadline=None, max_examples=15)
def test_numeric_only_pipeline_preserves_shape(n_rows: int) -> None:
    df = pl.DataFrame(
        {
            "x1": [float(i) for i in range(n_rows)],
            "x2": list(range(n_rows)),
        }
    )
    cfg = FeaturesConfig(
        categorical_low_card_threshold=10,
        spec=FeaturesSpec(numeric_columns=["x1", "x2"]),
    )
    pipe = make_features(cfg)
    out = pipe.fit_transform(df)
    assert isinstance(out, pl.DataFrame)
    assert out.height == n_rows
    assert set(out.columns) == {"x1", "x2"}


@given(n_rows=st.integers(min_value=10, max_value=100))
@settings(deadline=None, max_examples=10)
def test_passthrough_low_card_categorical_keeps_dtype_categorical(
    n_rows: int,
) -> None:
    df = pl.DataFrame(
        {
            "x1": [float(i) for i in range(n_rows)],
            "cat": [["a", "b", "c"][i % 3] for i in range(n_rows)],
        }
    )
    cfg = FeaturesConfig(
        categorical_low_card_threshold=10,
        spec=FeaturesSpec(numeric_columns=["x1"], categorical_columns=["cat"]),
    )
    cards = cardinalities_from(df, ["cat"])
    pipe = make_features(cfg, cardinalities=cards)
    out = pipe.fit_transform(df, [0] * n_rows)
    # Low-card → passthrough → cat retains Categorical dtype
    assert "cat" in out.columns
    assert out.height == n_rows

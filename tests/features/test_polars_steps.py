"""Stateless Polars block tests (per D4 — column selection + categorical cast)."""

from __future__ import annotations

import polars as pl
import pytest

from rux_ml.config import FeaturesSpec
from rux_ml.features.polars_steps import select_columns


def test_select_columns_returns_requested_columns_in_order(tiny_df: pl.DataFrame) -> None:
    spec = FeaturesSpec(numeric_columns=["x2", "x1"], categorical_columns=["low_cat"])
    out = select_columns(tiny_df, spec)
    assert out.columns == ["x2", "x1", "low_cat"]


def test_select_columns_casts_categoricals(tiny_df: pl.DataFrame) -> None:
    spec = FeaturesSpec(categorical_columns=["low_cat"])
    out = select_columns(tiny_df, spec)
    assert out.schema["low_cat"] == pl.Categorical


def test_select_columns_raises_on_missing_column(tiny_df: pl.DataFrame) -> None:
    spec = FeaturesSpec(numeric_columns=["nope"])
    with pytest.raises(KeyError, match="missing columns"):
        select_columns(tiny_df, spec)


def test_select_columns_idempotent(tiny_df: pl.DataFrame) -> None:
    spec = FeaturesSpec(numeric_columns=["x1"], categorical_columns=["low_cat"])
    a = select_columns(tiny_df, spec)
    b = select_columns(a, spec)
    assert a.equals(b)


def test_select_columns_skips_recast_when_already_categorical(tiny_df: pl.DataFrame) -> None:
    spec = FeaturesSpec(categorical_columns=["low_cat"])
    once = select_columns(tiny_df, spec)
    twice = select_columns(once, spec)  # should not re-cast or change anything
    assert once.equals(twice)

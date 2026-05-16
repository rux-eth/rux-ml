"""Deterministic split tests."""

from __future__ import annotations

import polars as pl
import pytest

from rux_ml.data.splits import train_val_test_split


def test_split_is_deterministic_for_same_seed(tiny_df: pl.DataFrame) -> None:
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = train_val_test_split(tiny_df, ratios=ratios, seed=42)
    b = train_val_test_split(tiny_df, ratios=ratios, seed=42)
    for key in ("train", "val", "test"):
        assert a[key].equals(b[key])


def test_split_differs_for_different_seeds(tiny_df: pl.DataFrame) -> None:
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = train_val_test_split(tiny_df, ratios=ratios, seed=42)
    b = train_val_test_split(tiny_df, ratios=ratios, seed=43)
    # Shuffle should produce different order for at least one partition on 10 rows
    assert not a["train"].equals(b["train"])


def test_split_preserves_all_rows(tiny_df: pl.DataFrame) -> None:
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    parts = train_val_test_split(tiny_df, ratios=ratios, seed=0)
    total = parts["train"].height + parts["val"].height + parts["test"].height
    assert total == tiny_df.height


def test_split_rejects_bad_ratio_keys(tiny_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="train/val/test"):
        train_val_test_split(tiny_df, ratios={"a": 0.5, "b": 0.5}, seed=0)


def test_split_rejects_ratios_not_summing_to_one(tiny_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        train_val_test_split(
            tiny_df, ratios={"train": 0.5, "val": 0.5, "test": 0.5}, seed=0
        )


def test_split_rejects_negative_ratio(tiny_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        train_val_test_split(
            tiny_df, ratios={"train": 1.1, "val": 0.0, "test": -0.1}, seed=0
        )


def test_split_handles_empty_frame() -> None:
    empty = pl.DataFrame({"x": [], "y": []})
    parts = train_val_test_split(
        empty, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=0
    )
    for key in ("train", "val", "test"):
        assert parts[key].height == 0

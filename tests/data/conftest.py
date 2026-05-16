"""Fixtures for data-layer tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def tiny_df() -> pl.DataFrame:
    """A tiny synthetic dataset with mixed dtypes."""
    return pl.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "cat": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def parquet_file(tmp_path: Path, tiny_df: pl.DataFrame) -> Path:
    p = tmp_path / "tiny.parquet"
    tiny_df.write_parquet(p)
    return p


@pytest.fixture
def parquet_dir(tmp_path: Path, tiny_df: pl.DataFrame) -> Path:
    """Two-part Parquet directory holding the same logical rows split in half."""
    d = tmp_path / "tiny_parts"
    d.mkdir()
    tiny_df.slice(0, 5).write_parquet(d / "part_0.parquet")
    tiny_df.slice(5, 5).write_parquet(d / "part_1.parquet")
    return d


@pytest.fixture
def shuffled_parquet_file(tmp_path: Path, tiny_df: pl.DataFrame) -> Path:
    """Same logical content as `parquet_file` but row order shuffled."""
    p = tmp_path / "tiny_shuffled.parquet"
    tiny_df.sample(fraction=1.0, shuffle=True, seed=999).write_parquet(p)
    return p


@pytest.fixture
def cas_root(tmp_path: Path) -> Path:
    return tmp_path / "cas"


@pytest.fixture
def manifests_root(tmp_path: Path) -> Path:
    return tmp_path / "manifests"

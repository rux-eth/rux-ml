"""Smoke tests for the XGBoost DataIter — verifies wiring, not training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from numpy.typing import NDArray

from rux_ml.data.data_iter import ParquetDataIter


@pytest.fixture
def batch_files(tmp_path: Path) -> list[Path]:
    """Two Parquet batch files for the DataIter."""
    rows_per_batch = 50
    files: list[Path] = []
    for i in range(2):
        df = pl.DataFrame(
            {
                "x1": [float(j + i * rows_per_batch) for j in range(rows_per_batch)],
                "x2": [float(j * 2) for j in range(rows_per_batch)],
                "y": [(j + i) % 2 for j in range(rows_per_batch)],
            }
        )
        p = tmp_path / f"batch_{i}.parquet"
        df.write_parquet(p)
        files.append(p)
    return files


def test_dataiter_rejects_empty_file_list() -> None:
    with pytest.raises(ValueError, match="at least one Parquet file"):
        ParquetDataIter([], "y", cache_prefix="dummy")


def test_dataiter_yields_each_file_once(batch_files: list[Path], tmp_path: Path) -> None:
    """Iterate manually and verify both batches arrive."""
    cache = tmp_path / "cache"
    cache.mkdir()
    it = ParquetDataIter(batch_files, "y", cache_prefix=str(cache / "xgb"))

    captured: list[tuple[int, int]] = []

    def capture(
        *, data: NDArray[np.float64], label: NDArray[Any], **_kwargs: Any
    ) -> None:
        _ = label
        captured.append((int(data.shape[0]), int(data.shape[1])))

    while it.next(capture):
        pass
    assert len(captured) == 2
    assert captured[0] == (50, 2)
    assert captured[1] == (50, 2)


def test_dataiter_reset_restarts_iteration(
    batch_files: list[Path], tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    it = ParquetDataIter(batch_files, "y", cache_prefix=str(cache / "xgb"))

    captured: list[int] = []

    def capture(
        *, data: NDArray[np.float64], label: NDArray[Any], **_kwargs: Any
    ) -> None:
        _ = label
        captured.append(int(data.shape[0]))

    while it.next(capture):
        pass
    it.reset()
    while it.next(capture):
        pass
    assert captured == [50, 50, 50, 50]


def test_dataiter_constructs_extmem_quantile_dmatrix(
    batch_files: list[Path], tmp_path: Path
) -> None:
    """End-to-end smoke: build an ExtMemQuantileDMatrix from the iterator."""
    cache = tmp_path / "cache"
    cache.mkdir()
    it = ParquetDataIter(batch_files, "y", cache_prefix=str(cache / "xgb"))
    dmat = xgb.ExtMemQuantileDMatrix(it, max_bin=64)
    # 100 rows total, 2 feature columns
    assert dmat.num_row() == 100
    assert dmat.num_col() == 2

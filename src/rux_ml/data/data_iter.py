"""XGBoost ``DataIter`` for ``ExtMemQuantileDMatrix`` (per D3 out-of-core path).

Each Parquet file becomes one batch. Callers control batch sizing by
partitioning the source data into appropriately-sized files (NVIDIA recommends
~5-10 GB per batch on a 36 GB host; tuning depends on the dataset).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import xgboost as xgb


class ParquetDataIter(xgb.DataIter):
    """Per-file Parquet batch iterator for ``xgb.ExtMemQuantileDMatrix``.

    Args:
        files: Ordered list of Parquet files; each is one batch.
        target_column: Column to be popped as the label.
        cache_prefix: XGBoost cache directory prefix (must be writable).
    """

    def __init__(
        self,
        files: list[Path],
        target_column: str,
        *,
        cache_prefix: str,
    ) -> None:
        # Init parent first so xgboost.DataIter.__del__ can safely run even if
        # our own validation rejects the inputs.
        super().__init__(cache_prefix=cache_prefix)
        if not files:
            msg = "ParquetDataIter requires at least one Parquet file"
            raise ValueError(msg)
        self._files = files
        self._target_column = target_column
        self._idx = 0

    def reset(self) -> None:
        """Rewind for a fresh pass."""
        self._idx = 0

    def next(self, input_data: Callable[..., None]) -> bool:  # noqa: A003 — XGBoost API name
        """Yield the next batch via ``input_data(data=X, label=y)``.

        Returns ``True`` after yielding so XGBoost calls again; returns
        ``False`` when all files have been consumed (per XGBoost 3.x docs).
        """
        if self._idx >= len(self._files):
            return False
        df = pl.read_parquet(self._files[self._idx])
        y = df[self._target_column].to_numpy()
        x = df.drop(self._target_column).to_numpy()
        input_data(data=x, label=y)
        self._idx += 1
        return True

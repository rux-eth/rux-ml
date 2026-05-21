"""XGBoost ``DataIter`` for ``ExtMemQuantileDMatrix`` (per D3 out-of-core path).

Each Parquet file becomes one batch. Callers control batch sizing by
partitioning the source data into appropriately-sized files (NVIDIA recommends
~5-10 GB per batch on a 36 GB host; tuning depends on the dataset).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import xgboost as xgb

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def single_source_iter(
    df: pl.DataFrame,
    target_column: str,
    *,
    batch_count: int,
    tmp_dir: Path,
    cache_prefix: str,
) -> ParquetDataIter:
    """Chunk a single in-memory frame into N temp Parquets for ExtMem ingest.

    Mirrors the split-on-write pattern in XGBoost 3.2's
    ``demo/guide-python/external_memory.py`` for sources that arrive as a
    single DataFrame (the workbench's common shape after
    ``materialize(load_parquet(...))`` + feature transform). The caller owns
    ``tmp_dir`` lifecycle.
    """
    if batch_count <= 0:
        msg = f"batch_count must be >= 1, got {batch_count}"
        raise ValueError(msg)
    if df.height == 0:
        msg = "single_source_iter requires a non-empty DataFrame"
        raise ValueError(msg)
    if target_column not in df.columns:
        msg = f"target_column={target_column!r} not in DataFrame columns: {df.columns}"
        raise ValueError(msg)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Ceiling division so the final batch absorbs the remainder rather than
    # producing an extra short batch (matches polars iter_slices behavior).
    batch_size = max(1, (df.height + batch_count - 1) // batch_count)
    files: list[Path] = []
    for i, slice_df in enumerate(df.iter_slices(n_rows=batch_size)):
        path = tmp_dir / f"batch_{i}.parquet"
        slice_df.write_parquet(path)
        files.append(path)
    return ParquetDataIter(files=files, target_column=target_column, cache_prefix=cache_prefix)


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

    def next(self, input_data: Callable[..., None]) -> bool:
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

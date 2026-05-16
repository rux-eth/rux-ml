"""XGBoost ingest-path decision rule (per D3).

``select_ingest(x_bytes, data_cfg)`` returns the **DMatrix class** the trainer
should use:

- ``xgb.QuantileDMatrix`` when the estimated X size fits comfortably in VRAM
  (``x_bytes ≲ data_cfg.gpu_in_memory_x_gb_max``)
- ``xgb.ExtMemQuantileDMatrix`` otherwise — host-RAM-cached via the existing
  :class:`rux_ml.data.data_iter.ParquetDataIter` and the optional
  ``MemoryConfig.cache_host_ratio`` knob.

The decision is purely classification — the caller (CLI or future trial
runner) decides how to *construct* the matrix once it knows the path. PR-006
exercises the in-memory branch end-to-end; the ExtMem branch's full execution
path lands as soon as a problem actually exceeds the threshold (the iterator
and ``MemoryConfig.cache_host_ratio`` knob it needs are both already in place).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from xgboost import ExtMemQuantileDMatrix, QuantileDMatrix

if TYPE_CHECKING:
    import polars as pl

    from rux_ml.config import DataConfig

DEFAULT_BYTES_PER_GB: int = 1024**3  # GiB; matches DataConfig.gpu_in_memory_x_gb_max semantics


def estimate_x_bytes(df: pl.DataFrame) -> int:
    """Estimate the byte size of a materialized feature frame.

    Uses Polars' ``estimated_size("b")`` which sums per-column byte costs and
    is comparable to the in-memory footprint XGBoost will see after pandas
    conversion (close enough for the threshold check — the threshold itself
    is a soft heuristic, per D3).
    """
    return int(df.estimated_size("b"))


def select_ingest(
    x_bytes: int,
    data_cfg: DataConfig,
) -> type[QuantileDMatrix] | type[ExtMemQuantileDMatrix]:
    """Return the DMatrix class for the given estimated X size.

    Args:
        x_bytes: Estimated size of the X feature matrix in bytes.
        data_cfg: Data layer config (``gpu_in_memory_x_gb_max`` is the
            threshold above which we switch to the out-of-core path).

    Returns:
        ``xgb.QuantileDMatrix`` when ``x_bytes`` fits under the threshold,
        ``xgb.ExtMemQuantileDMatrix`` otherwise.
    """
    threshold_bytes = int(data_cfg.gpu_in_memory_x_gb_max * DEFAULT_BYTES_PER_GB)
    if x_bytes <= threshold_bytes:
        return QuantileDMatrix
    return ExtMemQuantileDMatrix

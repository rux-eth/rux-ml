"""Unit tests for the D3 ingest decision rule."""

from __future__ import annotations

import polars as pl
from xgboost import ExtMemQuantileDMatrix, QuantileDMatrix

from rux_ml.config import DataConfig
from rux_ml.training import DEFAULT_BYTES_PER_GB, estimate_x_bytes, select_ingest


def test_select_ingest_below_threshold_returns_quantiledmatrix() -> None:
    cfg = DataConfig(gpu_in_memory_x_gb_max=1.0)  # 1 GiB threshold
    # 100 MiB << 1 GiB → in-memory
    chosen = select_ingest(x_bytes=100 * 1024 * 1024, data_cfg=cfg)
    assert chosen is QuantileDMatrix


def test_select_ingest_above_threshold_returns_extmem() -> None:
    cfg = DataConfig(gpu_in_memory_x_gb_max=1.0)
    # 4 GiB > 1 GiB → out-of-core
    chosen = select_ingest(x_bytes=4 * 1024 * 1024 * 1024, data_cfg=cfg)
    assert chosen is ExtMemQuantileDMatrix


def test_select_ingest_at_exact_threshold_uses_in_memory() -> None:
    """Equality with the threshold falls into the in-memory branch (≲)."""
    cfg = DataConfig(gpu_in_memory_x_gb_max=2.0)
    exact_bytes = int(2.0 * DEFAULT_BYTES_PER_GB)
    chosen = select_ingest(x_bytes=exact_bytes, data_cfg=cfg)
    assert chosen is QuantileDMatrix


def test_estimate_x_bytes_returns_positive_int_for_small_df() -> None:
    df = pl.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
    n = estimate_x_bytes(df)
    assert isinstance(n, int)
    assert n > 0

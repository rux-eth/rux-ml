"""Polars/Parquet readers (per D3 — Polars lazy as the default ingest path)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl


def load_parquet(path: Path) -> pl.LazyFrame:
    """Lazy-scan a Parquet file or hive-partitioned directory.

    Returns a ``LazyFrame`` so callers can compose filters/projections before
    materializing. Directory inputs trigger Polars' built-in hive-partition
    discovery.
    """
    return pl.scan_parquet(path)


def materialize(
    lf: pl.LazyFrame,
    *,
    engine: Literal["streaming", "in-memory"] = "streaming",
) -> pl.DataFrame:
    """Collect a LazyFrame.

    ``engine="streaming"`` is the larger-than-RAM safe default per D3; Polars
    falls back to in-memory for streaming-unsupported nodes. Use
    ``engine="in-memory"`` when the data is known to fit and you want the
    legacy planner.
    """
    return lf.collect(engine=engine)


def iter_parquet_files(path: Path) -> list[Path]:
    """Deterministic sorted list of all ``.parquet`` files under ``path``.

    A single file returns ``[path]``; a directory recursively collects every
    ``*.parquet`` under it. Sorting is alphabetical to keep ``bytes_hash``
    composition deterministic (per D9).
    """
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.is_dir():
        msg = f"path {p!s} is neither a file nor a directory"
        raise FileNotFoundError(msg)
    return sorted(p.rglob("*.parquet"))

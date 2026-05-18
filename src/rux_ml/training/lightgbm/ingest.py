"""LightGBM ingest helper (per PR-018 / Q-Ingest research).

LightGBM's ``Dataset`` always quantizes input features to uint8 histograms at
construction (default ``max_bin=255``), giving ~8x memory compression vs raw
float arrays. Per Q-Ingest research findings at LightGBM v4.5.0 source:

- **No tier-switch needed.** Unlike XGBoost (``QuantileDMatrix`` vs
  ``ExtMemQuantileDMatrix`` per D3), LightGBM's binned dataset is small
  enough that workbench-scale data (RAM-bound at 36 GB host) fits without
  out-of-core fallback. A 1Mx50 dataset is ~50 MB after binning.
- **Out-of-core path is file-based**, not host-RAM-tier: ``two_round=True``
  for memory-mapped files; ``Dataset(data=[Sequence(...), ...])`` for
  user-defined chunked readers. Both are deferred to a future PR if needed.
- **Polars input not in v4.5.0** (issue #6204, PR #7264 open as of 2026-05).
  Polars → pandas (or pyarrow.Table) at the boundary; the workbench's
  features layer already does this in
  ``src/rux_ml/features/pipeline.py::_polars_select_then_pandas``.

This module exists for symmetry with ``src/rux_ml/training/xgboost/ingest.py``
and as the home for any future LightGBM-side ingest decision rules; for now
it is a thin pass-through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from rux_ml.config import DataConfig


def build_dataset(
    x: Any,
    y: ArrayLike,
    data_cfg: DataConfig,
    *,
    categorical_feature: Any = "auto",
) -> Any:
    """Build a LightGBM ``Dataset`` from ``(x, y)``.

    Args:
        x: Feature frame (pandas DataFrame after the workbench's Polars→pandas
            shim, or a numpy array). LightGBM auto-detects pandas Categorical
            dtype columns as categorical when ``categorical_feature="auto"``
            (Q-Cat finding — this is exactly what the workbench's
            ``_ColumnRouter`` produces for low-cardinality columns).
        y: Target column.
        data_cfg: Data layer config. Currently unused — kept for signature
            symmetry with the XGBoost ingest helper and as a future hook
            point if a LightGBM-specific decision rule emerges.
        categorical_feature: Pass-through to ``lightgbm.Dataset(...,
            categorical_feature=...)``. Default ``"auto"`` triggers
            LightGBM's pandas-Categorical auto-detection — works for the
            workbench's existing pipeline output. Pass explicit column
            names/indices to override.

    Returns:
        A ``lightgbm.Dataset`` instance ready for training.
    """
    _ = data_cfg  # reserved for future LightGBM-side ingest rules

    from lightgbm import Dataset  # noqa: PLC0415 — lazy import (extra-gated)

    return Dataset(data=x, label=y, categorical_feature=categorical_feature)

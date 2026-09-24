"""Data layer — Polars/Parquet ingest, splits, content-addressed versioning, XGBoost DataIter.

Public surface is intentionally small; most callers reach for the layer
modules directly (e.g. ``from rux_ml.data.versioning import compute_data_hash``).
"""

from rux_ml.data.cv import (
    CombinatorialPurgedSplitter,
    GroupKFoldSplitter,
    KFoldSplitter,
    PanelCombinatorialPurgedSplitter,
    Splitter,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    make_splitter,
)
from rux_ml.data.data_iter import ParquetDataIter, single_source_iter
from rux_ml.data.loaders import load_parquet, materialize
from rux_ml.data.quarantine import OracleQuarantineError, check_oracle_quarantine
from rux_ml.data.splits import make_splits, temporal_train_val_test_split, train_val_test_split
from rux_ml.data.versioning import (
    Manifest,
    compute_data_hash,
    list_manifests,
    read_manifest,
    snapshot,
    write_manifest,
)

__all__ = [
    "CombinatorialPurgedSplitter",
    "GroupKFoldSplitter",
    "KFoldSplitter",
    "Manifest",
    "OracleQuarantineError",
    "PanelCombinatorialPurgedSplitter",
    "ParquetDataIter",
    "Splitter",
    "StratifiedKFoldSplitter",
    "TimeSeriesSplitter",
    "check_oracle_quarantine",
    "compute_data_hash",
    "list_manifests",
    "load_parquet",
    "make_splits",
    "make_splitter",
    "materialize",
    "read_manifest",
    "single_source_iter",
    "snapshot",
    "temporal_train_val_test_split",
    "train_val_test_split",
    "write_manifest",
]

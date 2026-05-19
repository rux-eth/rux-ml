"""Data layer config (per D3, D9, D14; PR-024 split_kind + time_column)."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from rux_ml.config._strict_model import StrictModel


class DataConfig(StrictModel):
    # Set per problem; absent in base.toml on purpose
    source_path: Path | None = None
    target_column: str | None = None

    # Content-addressed cache + manifests (per D9 + D14 Option A)
    cas_root: Path = Path("data/cas")
    manifests_root: Path = Path("data/manifests")

    # XGBoost ingest decision-rule threshold (per D3 ARCHITECTURE.md "Decision Rules")
    # X estimated below this size -> QuantileDMatrix on GPU; above -> ExtMemQuantileDMatrix.
    gpu_in_memory_x_gb_max: float = 18.0

    # Train / val / test ratios (sum should be 1.0)
    split_ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )

    # PR-024: one-off baseline split policy. ``random`` shuffles rows (v0.1
    # default — preserves backward compat); ``time_ordered`` sorts by
    # ``time_column`` and slices into temporally-ordered partitions. Selected
    # at config validation time; ``time_ordered`` requires ``time_column`` to
    # be set. Cross-field validation lives on ``RuxMLConfig`` (data ↔ cv.kind
    # consistency check).
    split_kind: Literal["random", "time_ordered"] = "random"

    # PR-024: column holding the timestamp used by ``time_ordered`` splits and
    # by PR-023's time-aware CV variants (``TimeSeriesSplitCV.time_column``,
    # ``PanelCombinatorialPurgedCV.time_column``). Set per problem; absent in
    # base.toml. None is valid when ``split_kind == "random"``.
    time_column: str | None = None

"""Data layer config (per D3, D9, D14; PR-024 split_kind + time_column; PR-040 oracle)."""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from rux_ml.config._strict_model import StrictModel


class OracleQuarantineConfig(StrictModel):
    """Oracle quarantine at ingest (PR-040; program ACCEPTANCE C13).

    Values live only in ``configs/base.toml`` ``[data.oracle]``, mirrored from the
    harness's ``config/harness.toml`` ``[oracle]``. Both fields are required: an
    empty value would silently disable half the check (``Path(d) / ""`` is ``d``).
    """

    # Column-name prefix of oracle-derived columns (matched case-insensitively,
    # including nested struct/list field names).
    namespace: str = Field(min_length=1)
    # File name that tags a directory as an oracle store (any ancestor, any depth).
    tag_file: str = Field(min_length=1)

    @field_validator("tag_file")
    @classmethod
    def _tag_file_is_a_bare_name(cls, v: str) -> str:
        separators = [sep for sep in ("/", "\\", os.sep, os.altsep) if sep]
        if any(sep in v for sep in separators):
            msg = f"tag_file must be a bare file name, got {v!r}"
            raise ValueError(msg)
        return v


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

    # PR-040: oracle quarantine. None refuses every ingest (fail closed) rather
    # than disabling the check; configs/base.toml sets it. Hash-elided in
    # config/root.py — it only decides whether ingest refuses, never what a
    # passing run computes.
    oracle: OracleQuarantineConfig | None = None

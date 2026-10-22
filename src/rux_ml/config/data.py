"""Data layer config (per D3, D9, D14; PR-024 split_kind + time_column; PR-040 oracle)."""

import datetime as dt
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from rux_ml.config._strict_model import StrictModel

# PR-065: a date-given cut is the date's UTC midnight in epoch ms (the time column's unit,
# as ``m9.h_max_ms``). Calendar constants, not configuration.
_EPOCH = dt.date(1970, 1, 1)
_DAY_MS = 86_400_000
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


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
    # PR-043: column-name prefix of supervised labels (matched case-insensitively);
    # a column in it is consumed only as ``data.target_column``, never a feature.
    label_namespace: str = Field(min_length=1)

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
    # PR-042 adds ``symbol_holdout``: whole groups (coins) per partition, see
    # ``group_column`` / ``symbol_holdout_seed`` below.
    split_kind: Literal["random", "time_ordered", "symbol_holdout"] = "random"

    # PR-024: column holding the timestamp used by ``time_ordered`` splits and
    # by PR-023's time-aware CV variants (``TimeSeriesSplitCV.time_column``,
    # ``PanelCombinatorialPurgedCV.time_column``). Set per problem; absent in
    # base.toml. None is valid when ``split_kind == "random"``.
    time_column: str | None = None

    # PR-042: ``time_ordered`` embargo in ``time_column``'s own units (epoch ms for
    # an Int64 stamp). None = legacy row-count slicing; an int keeps a row only if
    # ``t + split_embargo <`` the next partition's first stamp. Inapplicable to the
    # other kinds (a regime overlay may switch kind without unsetting it).
    split_embargo: int | None = Field(default=None, ge=0)

    # PR-042: ``symbol_holdout`` — the column naming the group (coin) and the seed
    # that fixes each group's partition (config, never the per-trial seed, so every
    # fit of a study holds out the same groups). Both required by that kind.
    group_column: str | None = None
    symbol_holdout_seed: int | None = None

    # PR-046: the nested-prefix learning curve (program D43). Keeps the first
    # fraction of the train partition's unique ``time_column`` stamps; val and test
    # are untouched, so the prefixes of one regime share one OOS set. None = all.
    train_prefix_frac: float | None = Field(default=None, gt=0.0, le=1.0)

    # PR-065 (rumpy program v0.7 D63): the ``[m9]`` time-block cut given as a date — its
    # UTC midnight in epoch ms (``split_cut_stamp``): train keeps ``t + split_embargo <
    # cut``, test ``t >= cut`` (ties to test). A val-less ``[m9]`` ``time_ordered`` split
    # only, without ``train_prefix_frac`` (validators in config/root.py); ``split_ratios``
    # is still sum-checked but no longer places the cut. None = the fraction-chosen cut.
    split_cut_date: dt.date | None = None

    # PR-040: oracle quarantine. None refuses every ingest (fail closed) rather
    # than disabling the check; configs/base.toml sets it. Hash-elided in
    # config/root.py — it only decides whether ingest refuses, never what a
    # passing run computes.
    oracle: OracleQuarantineConfig | None = None

    @field_validator("split_cut_date", mode="before")
    @classmethod
    def _cut_is_a_date(cls, v: object) -> object:
        # pydantic's lax date truncates a datetime (an offset midnight would move the cut by
        # the offset) and reads an int as a Unix stamp: only a date, or an ISO ``YYYY-MM-DD``
        # string (the env / dot-path override routes), is taken.
        if v is None or (isinstance(v, dt.date) and not isinstance(v, dt.datetime)):
            return v
        if isinstance(v, str) and _ISO_DATE.fullmatch(v):
            return v
        msg = (
            f"data.split_cut_date takes a date (YYYY-MM-DD; its UTC midnight is the cut), got {v!r}"
        )
        raise ValueError(msg)

    @property
    def split_cut_stamp(self) -> int | None:
        """PR-065: ``split_cut_date``'s UTC midnight in epoch ms (the cut), or None."""
        if self.split_cut_date is None:
            return None
        return (self.split_cut_date - _EPOCH).days * _DAY_MS

    def refuse_val_with_cut(self) -> None:
        """PR-065: a date cut is val-less only (``split_ratios.val == 0``), refused by name."""
        val = self.split_ratios.get("val")
        if self.split_cut_date is not None and val != 0:
            msg = (
                f"data.split_cut_date is val-less only: data.split_ratios.val must be 0, got {val}"
            )
            raise ValueError(msg)

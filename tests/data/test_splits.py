"""Deterministic split tests."""

from __future__ import annotations

import polars as pl
import pytest

from rux_ml.config import DataConfig, KFoldCV, RuxMLConfig, TimeSeriesSplitCV
from rux_ml.data.splits import (
    make_splits,
    temporal_train_val_test_split,
    train_val_test_split,
)


def test_split_is_deterministic_for_same_seed(tiny_df: pl.DataFrame) -> None:
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = train_val_test_split(tiny_df, ratios=ratios, seed=42)
    b = train_val_test_split(tiny_df, ratios=ratios, seed=42)
    for key in ("train", "val", "test"):
        assert a[key].equals(b[key])


def test_split_differs_for_different_seeds(tiny_df: pl.DataFrame) -> None:
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = train_val_test_split(tiny_df, ratios=ratios, seed=42)
    b = train_val_test_split(tiny_df, ratios=ratios, seed=43)
    # Shuffle should produce different order for at least one partition on 10 rows
    assert not a["train"].equals(b["train"])


def test_split_preserves_all_rows(tiny_df: pl.DataFrame) -> None:
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    parts = train_val_test_split(tiny_df, ratios=ratios, seed=0)
    total = parts["train"].height + parts["val"].height + parts["test"].height
    assert total == tiny_df.height


def test_split_rejects_bad_ratio_keys(tiny_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="train/val/test"):
        train_val_test_split(tiny_df, ratios={"a": 0.5, "b": 0.5}, seed=0)


def test_split_rejects_ratios_not_summing_to_one(tiny_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        train_val_test_split(tiny_df, ratios={"train": 0.5, "val": 0.5, "test": 0.5}, seed=0)


def test_split_rejects_negative_ratio(tiny_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        train_val_test_split(tiny_df, ratios={"train": 1.1, "val": 0.0, "test": -0.1}, seed=0)


def test_split_handles_empty_frame() -> None:
    empty = pl.DataFrame({"x": [], "y": []})
    parts = train_val_test_split(empty, ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    for key in ("train", "val", "test"):
        assert parts[key].height == 0


# ---------- PR-024: temporal_train_val_test_split ----------


def _ts_df(n: int) -> pl.DataFrame:
    """N-row time-ordered frame; the ``ts`` column is intentionally
    out-of-order so the sort step has work to do."""
    return pl.DataFrame(
        {
            "ts": [(i * 13) % n for i in range(n)],
            "x": list(range(n)),
        }
    )


def test_temporal_split_is_time_ordered() -> None:
    df = _ts_df(100)
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    parts = temporal_train_val_test_split(df, time_column="ts", ratios=ratios)
    # Train < val < test in temporal terms (allowing equal boundary timestamps).
    assert parts["train"]["ts"].max() <= parts["val"]["ts"].min()
    assert parts["val"]["ts"].max() <= parts["test"]["ts"].min()


def test_temporal_split_is_deterministic() -> None:
    """No seed; same input → same output. Promote-time reproducibility relies on this."""
    df = _ts_df(80)
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    a = temporal_train_val_test_split(df, time_column="ts", ratios=ratios)
    b = temporal_train_val_test_split(df, time_column="ts", ratios=ratios)
    for key in ("train", "val", "test"):
        assert a[key].equals(b[key])


def test_temporal_split_preserves_all_rows() -> None:
    df = _ts_df(50)
    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    parts = temporal_train_val_test_split(df, time_column="ts", ratios=ratios)
    assert parts["train"].height + parts["val"].height + parts["test"].height == df.height


def test_temporal_split_rejects_missing_time_column() -> None:
    df = _ts_df(10)
    with pytest.raises(ValueError, match="time_column"):
        temporal_train_val_test_split(
            df, time_column="missing_col", ratios={"train": 0.7, "val": 0.15, "test": 0.15}
        )


def test_temporal_split_handles_empty_frame() -> None:
    empty = pl.DataFrame({"ts": [], "x": []})
    parts = temporal_train_val_test_split(
        empty, time_column="ts", ratios={"train": 0.7, "val": 0.15, "test": 0.15}
    )
    for key in ("train", "val", "test"):
        assert parts[key].height == 0


def test_temporal_split_respects_panel_structure() -> None:
    """On a stacked panel (K assets per timestamp), partitions are sliced by
    global row position after sort, so boundary timestamps may overlap. The
    contract is monotone-non-decreasing across train→val→test."""
    rows = []
    for t in range(20):
        for asset in ("A", "B", "C"):
            rows.append({"ts": t, "asset": asset, "x": float(t * 10)})
    df = pl.DataFrame(rows)
    parts = temporal_train_val_test_split(
        df, time_column="ts", ratios={"train": 0.7, "val": 0.15, "test": 0.15}
    )
    # Each partition's timestamps are monotone-non-decreasing relative to the next.
    assert parts["train"]["ts"].max() <= parts["val"]["ts"].min()
    assert parts["val"]["ts"].max() <= parts["test"]["ts"].min()


# ---------- PR-024: make_splits dispatcher ----------


def test_make_splits_dispatches_random_by_default() -> None:
    df = _ts_df(40)
    cfg = RuxMLConfig()  # split_kind defaults to "random"
    random_parts = train_val_test_split(df, ratios=cfg.data.split_ratios, seed=42)
    dispatched = make_splits(cfg, df, seed=42)
    for key in ("train", "val", "test"):
        assert dispatched[key].equals(random_parts[key])


def test_make_splits_dispatches_temporal_when_time_ordered() -> None:
    df = _ts_df(40)
    cfg = RuxMLConfig(
        data=DataConfig(split_kind="time_ordered", time_column="ts"),
    )
    expected = temporal_train_val_test_split(
        df, time_column="ts", ratios=cfg.data.split_ratios
    )
    dispatched = make_splits(cfg, df, seed=42)
    for key in ("train", "val", "test"):
        assert dispatched[key].equals(expected[key])


def test_make_splits_temporal_path_ignores_seed() -> None:
    """Temporal path is deterministic; seed differences must not change output.
    Required for the cli/train.py ↔ registry/promote.py reproducibility contract."""
    df = _ts_df(40)
    cfg = RuxMLConfig(
        data=DataConfig(split_kind="time_ordered", time_column="ts"),
    )
    a = make_splits(cfg, df, seed=1)
    b = make_splits(cfg, df, seed=99999)
    for key in ("train", "val", "test"):
        assert a[key].equals(b[key])


# ---------- PR-024: RuxMLConfig cross-field validator ----------


def test_validator_rejects_time_ordered_without_time_column() -> None:
    with pytest.raises(ValueError, match=r"time_ordered.*requires data\.time_column"):
        RuxMLConfig(data=DataConfig(split_kind="time_ordered"))


def test_validator_rejects_temporal_cv_with_random_split() -> None:
    with pytest.raises(ValueError, match=r"temporal but data\.split_kind"):
        RuxMLConfig(cv=TimeSeriesSplitCV())  # default data.split_kind == "random"


def test_validator_accepts_consistent_time_ordered_pair() -> None:
    cfg = RuxMLConfig(
        data=DataConfig(split_kind="time_ordered", time_column="ts"),
        cv=TimeSeriesSplitCV(),
    )
    assert cfg.data.split_kind == "time_ordered"
    assert cfg.data.time_column == "ts"


def test_validator_accepts_random_split_with_non_temporal_cv() -> None:
    cfg = RuxMLConfig(data=DataConfig(), cv=KFoldCV())
    assert cfg.data.split_kind == "random"

"""Deterministic train/val/test splits.

Per D9, a full ``SeedSequence`` + per-component spawn lands in PR-013;
this PR uses a single integer seed routed through Polars' native shuffle
so PR-013 can plug a derived seed in unchanged.

**One-shot vs repeated CV** (PR-015): ``train_val_test_split`` is a one-shot
convenience that returns Polars DataFrames directly. For repeated K-fold-style
cross-validation (used by PR-007's Optuna objective and any HPO loop), use the
``Splitter`` Protocol in :mod:`rux_ml.data.cv` instead — it yields ``(train_idx,
test_idx)`` row-index pairs (sklearn convention) so the caller controls
materialisation policy.

**Random vs temporal split** (PR-024): :func:`train_val_test_split` shuffles
randomly with a seed — used for IID problems. :func:`temporal_train_val_test_split`
sorts by a timestamp column and slices into contiguous time-ordered partitions
— used for time-series problems. Two separate functions, no kind-knob, per the
≥3-cited convention surveyed in PR-023 Phase 3 Q5 (sktime ``temporal_train_test_split``;
Darts ``TimeSeries.split_before/split_after``; AutoGluon TimeSeriesPredictor;
Nixtla ``mlforecast.cross_validation``; mlfinlab). The ``rux-ml train`` baseline
path picks one via ``cfg.data.split_kind``.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, cast

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rux_ml.config import RuxMLConfig

_TOL = 1e-6


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    if set(ratios) != {"train", "val", "test"}:
        msg = f"split ratios keys must be exactly train/val/test, got {sorted(ratios)}"
        raise ValueError(msg)
    if any(v < 0 for v in ratios.values()):
        msg = f"split ratios must be non-negative, got {dict(ratios)}"
        raise ValueError(msg)
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, abs_tol=_TOL):
        msg = f"split ratios must sum to 1.0, got {total:.6f} from {dict(ratios)}"
        raise ValueError(msg)


def train_val_test_split(
    df: pl.DataFrame,
    *,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Shuffle rows deterministically by ``seed`` and slice into train/val/test.

    ``ratios`` must contain exactly the keys ``train``, ``val``, ``test`` and
    sum to 1.0 (within 1e-6 tolerance).
    """
    _validate_ratios(ratios)
    n = df.height
    if n == 0:
        return {k: df.clone() for k in ("train", "val", "test")}

    shuffled = df.sample(fraction=1.0, shuffle=True, seed=seed)
    train_n = int(n * ratios["train"])
    val_n = int(n * ratios["val"])
    # Test gets whatever is left to ensure all rows are placed.
    test_n = n - train_n - val_n
    return {
        "train": shuffled.slice(0, train_n),
        "val": shuffled.slice(train_n, val_n),
        "test": shuffled.slice(train_n + val_n, test_n),
    }


def temporal_train_val_test_split(
    df: pl.DataFrame,
    *,
    time_column: str,
    ratios: Mapping[str, float],
    embargo: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Sort rows by ``time_column`` and slice into time-ordered train/val/test.

    Deterministic — no seed needed. The output preserves global temporal
    ordering: every row in ``train`` has a timestamp ≤ every row in ``val``,
    and every row in ``val`` has a timestamp ≤ every row in ``test``. On a
    stacked panel (K rows per timestamp) the slice boundaries land on a
    row count, not a timestamp boundary — adjacent partitions may share
    the boundary timestamp; the CV layer's purge / embargo knobs (PR-023
    ``TimeSeriesSplitCV.embargo_time`` or ``PanelCombinatorialPurgedCV``)
    are the right tool when timestamp-atomic separation matters.

    ``ratios`` must contain exactly the keys ``train``, ``val``, ``test`` and
    sum to 1.0 (within 1e-6 tolerance). Convention per the ≥3-cited
    practitioner survey archived in PR-023 Phase 3 Q5.

    **Embargo** (PR-042; program v0.3 D41 / D45 #4): ``embargo=None`` keeps the
    legacy row-count slicing. An integer ``embargo`` (in ``time_column``'s own
    units — epoch ms for an Int64 stamp) keeps a row at ``t`` only if
    ``t + embargo < `` the first stamp of every later non-empty partition, so no
    label horizon up to ``embargo`` reaches past the split. ``embargo=0`` makes
    the separation timestamp-atomic. Integer time columns only; a train
    partition purged to nothing is refused.
    """
    _validate_ratios(ratios)
    if time_column not in df.columns:
        msg = (
            f"temporal_train_val_test_split: time_column={time_column!r} not "
            f"found in DataFrame columns: {df.columns}"
        )
        raise ValueError(msg)

    n = df.height
    if n == 0:
        return {k: df.clone() for k in ("train", "val", "test")}

    sorted_df = df.sort(time_column)
    train_n = int(n * ratios["train"])
    val_n = int(n * ratios["val"])
    test_n = n - train_n - val_n
    parts = {
        "train": sorted_df.slice(0, train_n),
        "val": sorted_df.slice(train_n, val_n),
        "test": sorted_df.slice(train_n + val_n, test_n),
    }
    if embargo is None:
        return parts
    if not df.schema[time_column].is_integer():
        msg = (
            f"split embargo needs an integer time_column; {time_column!r} is "
            f"{df.schema[time_column]} (the embargo is in the column's own units)"
        )
        raise ValueError(msg)
    order = ("train", "val", "test")
    for i, name in enumerate(order[:-1]):
        later = [
            int(cast("int", parts[k][time_column].min())) for k in order[i + 1 :] if parts[k].height
        ]
        if later:
            parts[name] = parts[name].filter(pl.col(time_column) + embargo < min(later))
    if train_n and parts["train"].height == 0:
        msg = f"split embargo {embargo} purged every train row (time_column {time_column!r})"
        raise ValueError(msg)
    return parts


def filter_non_null(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Keep the rows where every listed column is present (not null, not NaN).

    A listed column absent from ``df`` raises ``ValueError`` — a predicate on a
    column the set does not carry would otherwise pass every row silently.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        msg = f"m9.row_filter_non_null {missing} not in the training set columns {df.columns}"
        raise ValueError(msg)
    keep = [
        pl.col(c).is_not_null() & pl.col(c).is_not_nan()
        if df.schema[c].is_float()
        else pl.col(c).is_not_null()
        for c in columns
    ]
    return df.filter(keep)


def _unit_interval(seed: int, group: object) -> float:
    """The group's fixed point in [0, 1): sha256 of ``"<seed>:<group>"``, first 8 bytes."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def symbol_holdout_split(
    df: pl.DataFrame,
    *,
    group_column: str,
    ratios: Mapping[str, float],
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Partition whole groups (coins) into train/val/test (PR-042; program v0.3 D41).

    Each group's partition is a function of ``(seed, group)`` alone: the group's
    point ``u`` in [0, 1) (:func:`_unit_interval`) falls in train below
    ``ratios["train"]``, in val below ``train + val``, else in test. A coin
    therefore holds out identically in every coin universe — every nested prefix
    and every target of a study — which a seeded shuffle of the present coins
    would not give. Partition sizes are ratios in expectation, not exactly.
    Every row of a group travels with it. Refused: a missing or null group
    column, and a partition with a positive ratio that receives no group.
    """
    _validate_ratios(ratios)
    if group_column not in df.columns:
        msg = (
            f"symbol_holdout_split: group_column={group_column!r} not found in "
            f"DataFrame columns: {df.columns}"
        )
        raise ValueError(msg)
    nulls = df[group_column].null_count()
    if nulls:
        msg = f"symbol_holdout_split: {nulls} null value(s) in group_column {group_column!r}"
        raise ValueError(msg)
    cut_train = ratios["train"]
    cut_val = ratios["train"] + ratios["val"]
    members: dict[str, list[object]] = {"train": [], "val": [], "test": []}
    for group in df[group_column].unique().sort().to_list():
        u = _unit_interval(seed, group)
        members["train" if u < cut_train else "val" if u < cut_val else "test"].append(group)
    if df.height:
        empty = [k for k in ("train", "val", "test") if ratios[k] > 0 and not members[k]]
        if empty:
            msg = (
                f"symbol_holdout_split: no group assigned to {empty} "
                f"({df[group_column].n_unique()} groups, ratios {dict(ratios)}, seed {seed})"
            )
            raise ValueError(msg)
    return {k: df.filter(pl.col(group_column).is_in(members[k])) for k in members}


def make_splits(cfg: RuxMLConfig, df: pl.DataFrame, *, seed: int) -> dict[str, pl.DataFrame]:
    """Dispatcher used by ``cli/train.py`` and ``registry/promote.py``.

    Selects between :func:`train_val_test_split` and
    :func:`temporal_train_val_test_split` based on ``cfg.data.split_kind``.
    The ``RuxMLConfig`` model_validator enforces that ``time_ordered``
    implies ``cfg.data.time_column`` is set, so the temporal branch is safe
    to reach without re-validating here.

    ``[m9] row_filter_non_null`` (program PR-024 A9) is applied first, so every
    consumer of the split — train, tune, promote, score — sees the same rows.
    """
    if cfg.m9 is not None and cfg.m9.row_filter_non_null:
        df = filter_non_null(df, cfg.m9.row_filter_non_null)
    if cfg.data.split_kind == "time_ordered":
        # validator guarantees time_column is set; assert defensively for type
        # narrowing without runtime cost in the happy path.
        time_column = cfg.data.time_column
        assert time_column is not None  # validator-enforced precondition
        return temporal_train_val_test_split(
            df,
            time_column=time_column,
            ratios=cfg.data.split_ratios,
            embargo=cfg.data.split_embargo,
        )
    if cfg.data.split_kind == "symbol_holdout":
        # PR-042: the assignment seed is config (stable across trials), never the
        # per-trial ``seed`` — every fit of a study must hold out the same coins.
        group_column = cfg.data.group_column
        holdout_seed = cfg.data.symbol_holdout_seed
        assert group_column is not None and holdout_seed is not None  # validator-enforced
        return symbol_holdout_split(
            df, group_column=group_column, ratios=cfg.data.split_ratios, seed=holdout_seed
        )
    return train_val_test_split(df, ratios=cfg.data.split_ratios, seed=seed)


def split_definition(
    cfg: RuxMLConfig, df: pl.DataFrame, splits: Mapping[str, pl.DataFrame]
) -> dict[str, object]:
    """The one-off split as a record (PR-042; program C9 "the fold definitions").

    ``symbol_holdout``: the group column, the seed and the sorted groups per
    partition. ``time_ordered``: the time column, the embargo and the rows the
    embargo purged. Every kind: the rows per partition.
    """
    rows = {k: splits[k].height for k in ("train", "val", "test")}
    kind = cfg.data.split_kind
    if kind == "symbol_holdout":
        col = cfg.data.group_column
        assert col is not None  # validator-enforced
        return {
            "kind": kind,
            "group_column": col,
            "symbol_holdout_seed": cfg.data.symbol_holdout_seed,
            "groups": {k: splits[k][col].unique().sort().to_list() for k in rows},
            "rows": rows,
        }
    if kind == "time_ordered":
        return {
            "kind": kind,
            "time_column": cfg.data.time_column,
            "split_embargo": cfg.data.split_embargo,
            "rows": rows,
            "purged_rows": df.height - sum(rows.values()),
        }
    return {"kind": kind, "rows": rows}

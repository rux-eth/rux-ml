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

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

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
    return {
        "train": sorted_df.slice(0, train_n),
        "val": sorted_df.slice(train_n, val_n),
        "test": sorted_df.slice(train_n + val_n, test_n),
    }


def make_splits(
    cfg: RuxMLConfig, df: pl.DataFrame, *, seed: int
) -> dict[str, pl.DataFrame]:
    """Dispatcher used by ``cli/train.py`` and ``registry/promote.py``.

    Selects between :func:`train_val_test_split` and
    :func:`temporal_train_val_test_split` based on ``cfg.data.split_kind``.
    The ``RuxMLConfig`` model_validator enforces that ``time_ordered``
    implies ``cfg.data.time_column`` is set, so the temporal branch is safe
    to reach without re-validating here.
    """
    if cfg.data.split_kind == "time_ordered":
        # validator guarantees time_column is set; assert defensively for type
        # narrowing without runtime cost in the happy path.
        time_column = cfg.data.time_column
        assert time_column is not None  # validator-enforced precondition
        return temporal_train_val_test_split(
            df, time_column=time_column, ratios=cfg.data.split_ratios
        )
    return train_val_test_split(df, ratios=cfg.data.split_ratios, seed=seed)

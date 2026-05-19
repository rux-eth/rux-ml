# pyright: reportUnknownMemberType=false, reportReturnType=false
"""CV strategy — ``Splitter`` ``typing.Protocol`` + concrete strategies.

Per PR-015 (Tier-2 research, locked-in 2026-05-16) extended by PR-023 (panel-
aware CV, locked-in 2026-05-19):

- **Input shape:** Splitter accepts ``pl.DataFrame`` for ``X`` (matches PR-005's
  Polars convention). Internally extracts ``.height`` and ``.to_numpy()`` only
  where sklearn delegation requires arrays (Stratified, Group, CPCV, panel-CPCV).
- **Output shape:** ``(np.ndarray, np.ndarray)`` row-index pairs per sklearn
  convention — repeated CV stays in indices, not materialised DataFrames, so
  PR-007's objective owns the slice-and-train policy.
- **ExtMem compatibility:** each concrete Splitter declares
  ``extmem_compatible: ClassVar[bool]``. The training/objective layer checks
  this when ExtMem is the active ingest path and raises ``NotImplementedError``
  for incompatible pairings (per sub-decision C1; materialised fallback is
  deferred to a follow-up PR).
- **Groups column-to-array pattern:** ``GroupKFoldCV.groups_column`` references
  a column on the DataFrame; the **caller** resolves it to ``np.ndarray`` via
  ``df[col].to_numpy()`` and passes it to ``split(..., groups=arr)``. The
  Splitter never holds DataFrame state.
- **Time / asset columns (PR-023):** ``TimeSeriesSplitCV.time_column`` and the
  two ``PanelCombinatorialPurgedCV`` columns are resolved inside the splitter
  at ``.split()`` time because their translation depends on the timestamp
  distribution of the input. The splitter holds only field references, not
  DataFrame state.
- **Seed:** every Splitter accepts ``seed: int | None``. KFold / StratifiedKFold
  thread it through sklearn's ``random_state`` when ``shuffle=True``. The
  deterministic strategies (TimeSeriesSplit, GroupKFold, CPCV, panel-CPCV)
  ignore it. PR-013 plugs a ``SeedSequence(entropy).spawn()``-derived
  ``cv_seed`` here unchanged.
- **CPCV:** wraps ``skfolio.model_selection.CombinatorialPurgedCV`` which yields
  ``(train, list[test_path])`` per combinatorial split. We flatten the test
  paths into a single ``np.ndarray`` per yield so the output stays sklearn-
  shaped; per-path decomposition is out of scope for HPO scoring at v0.
- **Panel CPCV (PR-023 D1):** folds over the unique sorted timestamps from
  ``time_column``. skfolio CPCV runs on the timestamp axis (purge / embargo
  in timestamp units); each yielded fold maps back to row indices via the
  row-to-timestamp-index map. Per-asset purge is automatic — dropping a
  timestamp from train removes every asset row at that timestamp.

The file-level ``pyright`` pragma relaxes ``reportUnknownMemberType`` /
``reportReturnType`` because ``sklearn.model_selection`` ships parameter stubs
typed as ``Unknown`` (same pattern as ``training/metrics.py``); the integer
dtype skfolio yields (``np.integer[Any]``) is incompatible with the protocol's
``np.int_`` under strict covariance even though both encode integer indices at
runtime. Tests assert the runtime contract.
"""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import numpy as np
import pandas as pd  # used only by TimeSeriesSplitter._effective_gap for Timedelta parsing
import polars as pl
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from rux_ml.config import (
    CombinatorialPurgedCV,
    CVConfig,
    GroupKFoldCV,
    KFoldCV,
    PanelCombinatorialPurgedCV,
    StratifiedKFoldCV,
    TimeSeriesSplitCV,
)

_MIN_TS_FOR_DELTA: int = 2  # need >=2 unique timestamps to compute median delta

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class Splitter(Protocol):
    """Universal CV splitter surface (sklearn-shape, Polars-in).

    ``extmem_compatible`` is the only Splitter-level decision the training
    layer reads to decide whether the current ingest path supports this
    strategy. ``False`` causes a ``NotImplementedError`` when paired with
    ``ExtMemQuantileDMatrix`` until the materialised fallback lands.
    """

    extmem_compatible: ClassVar[bool]

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]: ...

    def get_n_splits(self) -> int: ...


# ---------- Concrete strategies ----------


def _dummy(n: int) -> NDArray[np.float64]:
    """Sklearn splitters only read ``.shape[0]`` for index generation; a tiny
    zero-array is sufficient and avoids materialising the real X."""
    return np.zeros(n, dtype=np.float64)


class KFoldSplitter:
    """IID K-fold (sklearn ``KFold``). Not ExtMem-compatible (intra-partition shuffle)."""

    extmem_compatible: ClassVar[bool] = False

    def __init__(self, n_splits: int, shuffle: bool, seed: int | None = None) -> None:
        self._inner = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        _ = (y, groups)
        yield from self._inner.split(_dummy(X.height))

    def get_n_splits(self) -> int:
        return int(self._inner.get_n_splits())


class StratifiedKFoldSplitter:
    """Class-balance-preserving K-fold (sklearn ``StratifiedKFold``).

    Requires ``y`` at ``.split()`` time. Not ExtMem-compatible.
    """

    extmem_compatible: ClassVar[bool] = False

    def __init__(self, n_splits: int, shuffle: bool, seed: int | None = None) -> None:
        self._inner = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=seed if shuffle else None,
        )

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        _ = groups
        if y is None:
            msg = "StratifiedKFoldSplitter requires y (target series) at split time"
            raise ValueError(msg)
        yield from self._inner.split(_dummy(X.height), y.to_numpy())

    def get_n_splits(self) -> int:
        return int(self._inner.get_n_splits())


class TimeSeriesSplitter:
    """Walk-forward (sklearn ``TimeSeriesSplit``).

    ``max_train_size=None`` → expanding window; ``int`` → rolling window of
    that size (Q1.b decision rule). ``gap`` excludes adjacent train-end /
    test-start samples (Q2.a simple-case embargo).

    **Panel-aware embargo (PR-023 D2)**: when ``embargo_time`` is supplied
    it wins over ``gap``:

    - ``int`` → used directly as the row-count gap (same semantics as ``gap``).
    - ``str`` (e.g., ``"24h"``) → requires ``time_column`` to be set on the
      input DataFrame; parsed via ``pandas.Timedelta`` and translated to a
      row-count gap by computing the **median delta between unique sorted
      timestamps** and the **mean rows-per-unique-timestamp**. Conservative
      approximation on irregular bars — flagged as ``best-guess`` per D3
      irregular-bar caveat.

    Inner sklearn ``TimeSeriesSplit`` is rebuilt at ``.split()`` time so the
    effective gap can depend on the input DataFrame's timestamps.

    **ExtMem-compatible** under the documented assumption that input partitions
    are time-ordered — the only Splitter strategy compatible with the file-
    level ExtMem path in v0. Time-unit embargo additionally requires
    ``time_column`` to be materialisable at split time; the workbench's
    materialised loader (``data.load_parquet → data.materialize``) already
    delivers a Polars DataFrame.
    """

    extmem_compatible: ClassVar[bool] = True

    def __init__(
        self,
        n_splits: int,
        gap: int,
        max_train_size: int | None,
        time_column: str | None = None,
        embargo_time: int | str | None = None,
        time_unit: Literal["ns", "us", "ms", "s"] | None = None,
        seed: int | None = None,
    ) -> None:
        _ = seed  # TimeSeriesSplit is deterministic.
        self._n_splits = n_splits
        self._gap = gap
        self._max_train_size = max_train_size
        self._time_column = time_column
        self._embargo_time = embargo_time
        self._time_unit: Literal["ns", "us", "ms", "s"] | None = time_unit

    def _effective_gap(self, X: pl.DataFrame) -> int:
        if self._embargo_time is None:
            return self._gap
        if isinstance(self._embargo_time, int):
            return self._embargo_time
        # str path: time-unit embargo translated via the DataFrame's timestamps.
        if self._time_column is None:
            msg = (
                "TimeSeriesSplitCV.embargo_time as duration string requires "
                "time_column to be set"
            )
            raise ValueError(msg)
        if self._time_column not in X.columns:
            msg = (
                f"TimeSeriesSplitCV.time_column={self._time_column!r} not found "
                f"in DataFrame columns: {X.columns}"
            )
            raise ValueError(msg)

        # PR-027 behavior matrix: explicit time_unit handling for pl.Int64
        # columns; pl.Datetime columns carry their own unit and reject
        # time_unit as user-confusion (fail-fast per PR-024 convention).
        col = X[self._time_column]
        is_int = col.dtype == pl.Int64
        is_datetime = col.dtype.is_temporal()
        if is_int and self._time_unit is None:
            msg = (
                f"TimeSeriesSplitCV.time_column={self._time_column!r} is pl.Int64 "
                f"and requires TimeSeriesSplitCV.time_unit to be set. Pick one of "
                f"'ns', 'us', 'ms', 's' per the integer's interpretation "
                f"(e.g. 's' for Unix-seconds)."
            )
            raise ValueError(msg)
        if is_datetime and self._time_unit is not None:
            msg = (
                f"TimeSeriesSplitCV.time_unit={self._time_unit!r} is meaningless "
                f"on a pl.Datetime column (the column carries its own unit). "
                f"Remove time_unit from the config."
            )
            raise ValueError(msg)

        duration_ns: int = int(pd.Timedelta(self._embargo_time).value)
        times_ns: NDArray[np.int64]
        if is_int:
            # Promote Int64 -> Datetime("ns") via polars' explicit-unit cast.
            # pl.from_epoch is the polars-canonical primitive for this.
            # `is_int` implies `self._time_unit is not None` (rejected above).
            assert self._time_unit is not None
            datetime_col = pl.from_epoch(col, time_unit=self._time_unit)
            times_ns = datetime_col.cast(pl.Datetime("ns")).to_numpy().astype(np.int64)
        else:
            times_ns = col.cast(pl.Datetime("ns")).to_numpy().astype(np.int64)

        unique_ns: NDArray[np.int64] = np.unique(times_ns)
        if unique_ns.size < _MIN_TS_FOR_DELTA:
            return 0
        median_delta_ns = int(np.median(np.diff(unique_ns)))
        if median_delta_ns <= 0:
            msg = "TimeSeriesSplitCV.time_column has non-positive median timestamp delta"
            raise ValueError(msg)
        unique_ts_to_skip = int(np.ceil(duration_ns / median_delta_ns))
        rows_per_ts = X.height / unique_ns.size
        return int(np.ceil(unique_ts_to_skip * rows_per_ts))

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        _ = (y, groups)
        inner = TimeSeriesSplit(
            n_splits=self._n_splits,
            gap=self._effective_gap(X),
            max_train_size=self._max_train_size,
        )
        yield from inner.split(_dummy(X.height))

    def get_n_splits(self) -> int:
        return self._n_splits


class GroupKFoldSplitter:
    """Group-leakage-preventing K-fold (sklearn ``GroupKFold``).

    ``groups`` must be supplied to ``.split()`` as a pre-resolved ``np.ndarray``
    (per the column-to-array convention — the caller extracts
    ``df[cfg.cv.groups_column].to_numpy()`` before calling). Not ExtMem-
    compatible at v0 (would require partition-aligned groups).
    """

    extmem_compatible: ClassVar[bool] = False

    def __init__(self, n_splits: int, seed: int | None = None) -> None:
        _ = seed  # GroupKFold is deterministic.
        self._inner = GroupKFold(n_splits=n_splits)

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        _ = y
        if groups is None:
            msg = (
                "GroupKFoldSplitter requires `groups` at split time; resolve it "
                "from `cfg.cv.groups_column` via `df[col].to_numpy()` before calling."
            )
            raise ValueError(msg)
        yield from self._inner.split(_dummy(X.height), groups=groups)

    def get_n_splits(self) -> int:
        return int(self._inner.get_n_splits())


class CombinatorialPurgedSplitter:
    """AFML CPCV with purge + one-sided post-test embargo.

    Wraps ``skfolio.model_selection.CombinatorialPurgedCV``. The upstream
    yields ``(train, list[test_path])`` per combinatorial split; we flatten
    the test-path list to a single ``np.ndarray`` so the output matches the
    sklearn ``(train, test)`` shape. Per-path decomposition (multiple
    backtest paths from one CV) is out of scope for HPO scoring at v0.

    **Ergonomic knobs (PR-023 D3)**: when ``target_horizon_bars`` or
    ``embargo_pct`` are non-zero AND their row-count siblings (``purged_size``
    / ``embargo_size``) are zero, they convert internally:

    - ``effective_purged_size = target_horizon_bars`` (symmetric, conservative-
      correct on regular bars per D4).
    - ``effective_embargo_size = int(N * embargo_pct)`` (AFML Snippet 7.3).

    When both are supplied, the row-count value wins. Inner skfolio splitter
    is built at ``.split()`` time so ``embargo_pct`` can resolve against ``N``.

    Not ExtMem-compatible (row-level purge/embargo logic).
    """

    extmem_compatible: ClassVar[bool] = False

    def __init__(
        self,
        n_folds: int,
        n_test_folds: int,
        purged_size: int,
        embargo_size: int,
        target_horizon_bars: int = 0,
        embargo_pct: float = 0.0,
        seed: int | None = None,
    ) -> None:
        _ = seed  # CPCV is deterministic.
        self._n_folds = n_folds
        self._n_test_folds = n_test_folds
        self._purged_size = purged_size
        self._embargo_size = embargo_size
        self._target_horizon_bars = target_horizon_bars
        self._embargo_pct = embargo_pct
        self._n_splits = comb(n_folds, n_test_folds)

    def _build_inner(self, n_rows: int):
        from skfolio.model_selection import (  # noqa: PLC0415
            CombinatorialPurgedCV as _SkfolioCPCV,
        )

        effective_purged = (
            self._purged_size if self._purged_size > 0 else self._target_horizon_bars
        )
        effective_embargo = (
            self._embargo_size if self._embargo_size > 0 else int(n_rows * self._embargo_pct)
        )
        return _SkfolioCPCV(
            n_folds=self._n_folds,
            n_test_folds=self._n_test_folds,
            purged_size=effective_purged,
            embargo_size=effective_embargo,
        )

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        _ = (y, groups)
        inner = self._build_inner(X.height)
        for train_idx, test_paths in inner.split(_dummy(X.height)):
            # skfolio yields a list of arrays (one per combinatorial test path);
            # flatten to a single ndarray for sklearn-shape conformance.
            test_idx = np.concatenate(test_paths) if test_paths else np.array([], dtype=np.int64)
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return self._n_splits


class PanelCombinatorialPurgedSplitter:
    """Panel-aware CPCV (PR-023 D1).

    Folds over the unique sorted timestamps read from ``time_column``; skfolio
    CPCV runs on the timestamp axis (so ``purged_size`` and ``embargo_size``
    operate in **timestamp units**, not rows). Each timestamp-level fold is
    mapped back to row indices by selecting every row whose timestamp is in
    that fold — per-asset purge is automatic (drop a timestamp from train →
    drop every asset's row at that timestamp).

    ``target_horizon_bars`` and ``embargo_pct`` are interpreted in timestamp
    units: ``purged_size = target_horizon_bars`` (number of unique timestamps
    to drop on each side of every test fold) and ``embargo_size = int(
    n_unique_timestamps * embargo_pct)``.

    ``asset_column`` is required and validated at split time so the wrapper
    can fail loudly when given mismatched data (and so combinatorial paths
    are decomposable downstream via row-index ``groupby(asset)``).

    Convention: mlfinlab ``StackedCombinatorialPurgedKFold`` + Numerai era-CV.

    Not ExtMem-compatible (requires the full ``time_column`` materialised at
    split time).
    """

    extmem_compatible: ClassVar[bool] = False

    def __init__(
        self,
        n_folds: int,
        n_test_folds: int,
        time_column: str,
        asset_column: str,
        target_horizon_bars: int = 0,
        embargo_pct: float = 0.0,
        seed: int | None = None,
    ) -> None:
        _ = seed  # CPCV is deterministic.
        self._n_folds = n_folds
        self._n_test_folds = n_test_folds
        self._time_column = time_column
        self._asset_column = asset_column
        self._target_horizon_bars = target_horizon_bars
        self._embargo_pct = embargo_pct
        self._n_splits = comb(n_folds, n_test_folds)

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        _ = (y, groups)
        for col in (self._time_column, self._asset_column):
            if col not in X.columns:
                msg = (
                    f"PanelCombinatorialPurgedCV requires column {col!r}; "
                    f"DataFrame has {X.columns}"
                )
                raise ValueError(msg)

        times = X[self._time_column].to_numpy()
        unique_ts = np.unique(times)  # np.unique returns a sorted array
        n_unique = unique_ts.size
        if n_unique < self._n_folds:
            msg = (
                f"PanelCombinatorialPurgedCV needs at least n_folds={self._n_folds} "
                f"unique timestamps in {self._time_column!r}; got {n_unique}"
            )
            raise ValueError(msg)

        embargo_ts = int(n_unique * self._embargo_pct)

        from skfolio.model_selection import (  # noqa: PLC0415
            CombinatorialPurgedCV as _SkfolioCPCV,
        )

        inner = _SkfolioCPCV(
            n_folds=self._n_folds,
            n_test_folds=self._n_test_folds,
            purged_size=self._target_horizon_bars,
            embargo_size=embargo_ts,
        )

        # Map each row to its timestamp's position in unique_ts.
        row_ts_idx = np.searchsorted(unique_ts, times)

        for train_ts_idx, test_ts_paths in inner.split(_dummy(n_unique)):
            train_mask = np.isin(row_ts_idx, train_ts_idx)
            train_rows = np.flatnonzero(train_mask).astype(np.int64)
            test_ts_concat = (
                np.concatenate(test_ts_paths)
                if test_ts_paths
                else np.array([], dtype=np.int64)
            )
            test_mask = np.isin(row_ts_idx, test_ts_concat)
            test_rows = np.flatnonzero(test_mask).astype(np.int64)
            yield train_rows, test_rows

    def get_n_splits(self) -> int:
        return self._n_splits


# ---------- Factory ----------


def make_splitter(cfg: CVConfig, *, seed: int | None = None) -> Splitter:
    """Construct the concrete Splitter for ``cfg``.

    ``seed`` flows into sklearn ``random_state`` for shuffle-based strategies
    (KFold, StratifiedKFold); the deterministic strategies (TimeSeriesSplit,
    GroupKFold, CPCV, panel-CPCV) ignore it. PR-013's ``cv_seed`` plugs in
    here unchanged.
    """
    match cfg:
        case KFoldCV():
            return KFoldSplitter(n_splits=cfg.n_splits, shuffle=cfg.shuffle, seed=seed)
        case StratifiedKFoldCV():
            return StratifiedKFoldSplitter(
                n_splits=cfg.n_splits, shuffle=cfg.shuffle, seed=seed
            )
        case TimeSeriesSplitCV():
            return TimeSeriesSplitter(
                n_splits=cfg.n_splits,
                gap=cfg.gap,
                max_train_size=cfg.max_train_size,
                time_column=cfg.time_column,
                embargo_time=cfg.embargo_time,
                time_unit=cfg.time_unit,
                seed=seed,
            )
        case GroupKFoldCV():
            return GroupKFoldSplitter(n_splits=cfg.n_splits, seed=seed)
        case CombinatorialPurgedCV():
            return CombinatorialPurgedSplitter(
                n_folds=cfg.n_folds,
                n_test_folds=cfg.n_test_folds,
                purged_size=cfg.purged_size,
                embargo_size=cfg.embargo_size,
                target_horizon_bars=cfg.target_horizon_bars,
                embargo_pct=cfg.embargo_pct,
                seed=seed,
            )
        case PanelCombinatorialPurgedCV():
            return PanelCombinatorialPurgedSplitter(
                n_folds=cfg.n_folds,
                n_test_folds=cfg.n_test_folds,
                time_column=cfg.time_column,
                asset_column=cfg.asset_column,
                target_horizon_bars=cfg.target_horizon_bars,
                embargo_pct=cfg.embargo_pct,
                seed=seed,
            )

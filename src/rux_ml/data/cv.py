# pyright: reportUnknownMemberType=false, reportReturnType=false
"""CV strategy — ``Splitter`` ``typing.Protocol`` + concrete strategies.

Per PR-015 (Tier-2 research, locked-in 2026-05-16):

- **Input shape:** Splitter accepts ``pl.DataFrame`` for ``X`` (matches PR-005's
  Polars convention). Internally extracts ``.height`` and ``.to_numpy()`` only
  where sklearn delegation requires arrays (Stratified, Group, CPCV).
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
- **Seed:** every Splitter accepts ``seed: int | None``. KFold / StratifiedKFold
  thread it through sklearn's ``random_state`` when ``shuffle=True``. The
  deterministic strategies (TimeSeriesSplit, GroupKFold, CPCV) ignore it. PR-013
  will plug a ``SeedSequence(entropy).spawn()``-derived ``cv_seed`` here
  unchanged.
- **CPCV:** wraps ``skfolio.model_selection.CombinatorialPurgedCV`` which yields
  ``(train, list[test_path])`` per combinatorial split. We flatten the test
  paths into a single ``np.ndarray`` per yield so the output stays sklearn-
  shaped; per-path decomposition is out of scope for HPO scoring at v0.

The file-level ``pyright`` pragma relaxes ``reportUnknownMemberType`` /
``reportReturnType`` because ``sklearn.model_selection`` ships parameter stubs
typed as ``Unknown`` (same pattern as ``training/metrics.py``); the integer
dtype skfolio yields (``np.integer[Any]``) is incompatible with the protocol's
``np.int_`` under strict covariance even though both encode integer indices at
runtime. Tests assert the runtime contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np
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
    StratifiedKFoldCV,
    TimeSeriesSplitCV,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import polars as pl
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

    **ExtMem-compatible** under the documented assumption that input partitions
    are time-ordered — the only Splitter strategy compatible with the file-
    level ExtMem path in v0.
    """

    extmem_compatible: ClassVar[bool] = True

    def __init__(
        self,
        n_splits: int,
        gap: int,
        max_train_size: int | None,
        seed: int | None = None,
    ) -> None:
        _ = seed  # TimeSeriesSplit is deterministic.
        self._inner = TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
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

    Not ExtMem-compatible (row-level purge/embargo logic).
    """

    extmem_compatible: ClassVar[bool] = False

    def __init__(
        self,
        n_folds: int,
        n_test_folds: int,
        purged_size: int,
        embargo_size: int,
        seed: int | None = None,
    ) -> None:
        _ = seed  # CPCV is deterministic.
        # Imported lazily so the dependency cost is only paid when CPCV is selected.
        from skfolio.model_selection import (  # noqa: PLC0415
            CombinatorialPurgedCV as _SkfolioCPCV,
        )

        self._inner = _SkfolioCPCV(
            n_folds=n_folds,
            n_test_folds=n_test_folds,
            purged_size=purged_size,
            embargo_size=embargo_size,
        )

    def split(
        self,
        X: pl.DataFrame,
        y: pl.Series | None = None,
        *,
        groups: NDArray[np.int_] | None = None,
    ) -> Iterator[tuple[NDArray[np.int_], NDArray[np.int_]]]:
        _ = (y, groups)
        for train_idx, test_paths in self._inner.split(_dummy(X.height)):
            # skfolio yields a list of arrays (one per combinatorial test path);
            # flatten to a single ndarray for sklearn-shape conformance.
            test_idx = np.concatenate(test_paths) if test_paths else np.array([], dtype=np.int64)
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        return int(self._inner.get_n_splits())


# ---------- Factory ----------


def make_splitter(cfg: CVConfig, *, seed: int | None = None) -> Splitter:
    """Construct the concrete Splitter for ``cfg``.

    ``seed`` flows into sklearn ``random_state`` for shuffle-based strategies
    (KFold, StratifiedKFold); the deterministic strategies (TimeSeriesSplit,
    GroupKFold, CPCV) ignore it. PR-013's ``cv_seed`` plugs in here unchanged.
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
                seed=seed,
            )

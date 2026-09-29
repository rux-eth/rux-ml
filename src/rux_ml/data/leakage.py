"""The leakage audit of a one-off split (PR-045; program v0.3 D41, ACCEPTANCE C9).

C9's two leakage tests, as measurements on the partitions a fit actually used:

- **stamps** — "no training stamp inside a test stamp's embargo": a row of a later
  partition violates when a stamp of the partitions fitted before it lies within
  ``window`` of its own stamp (``|s - t| <= window``). Val is checked against train;
  test against train + val (val early-stops the fit, so it is fitted data too).
- **groups** — "no holdout coin's row in train": the number of groups present in
  two partitions, for each pair.

The audit is independent of the splitters (nearest fitted stamp by binary search
over the sorted unique stamps; set intersection of groups), so it checks them.
:func:`assert_regime_clean` refuses a fit whose regime promises a separation the
audit does not find; the row-random regime (diagnostic) promises none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


class LeakageError(ValueError):
    """A split broke the separation its regime promises (a ``ValueError``: exit 2)."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"leakage: {detail} — refused")


def _within(
    later: NDArray[Any], weights: NDArray[Any] | None, ref: NDArray[Any], window: int
) -> int:
    """Rows (or their summed ``weights``) of ``later`` stamped within ``window`` of the
    nearest stamp in ``ref`` (sorted, unique)."""
    if not later.size or not ref.size:
        return 0
    i = np.searchsorted(ref, later)
    left = np.abs(later - ref[np.clip(i - 1, 0, ref.size - 1)])
    right = np.abs(ref[np.clip(i, 0, ref.size - 1)] - later)
    hit = np.minimum(left, right) <= window
    return int(hit.sum()) if weights is None else int(weights[hit].sum())


def _violations(later: pl.DataFrame, fitted: list[pl.DataFrame], col: str, window: int) -> int:
    stamps = [f[col].to_numpy() for f in fitted if f.height]
    if not later.height or not stamps:
        return 0
    return _within(later[col].to_numpy(), None, np.unique(np.concatenate(stamps)), window)


def leakage_audit(
    splits: Mapping[str, pl.DataFrame],
    *,
    time_column: str | None,
    group_column: str | None,
    window: int | None,
) -> dict[str, Any]:
    """Count stamp-window violations and group overlaps; an axis without its column
    (or, for stamps, without a window) is reported ``None`` — not audited."""
    stamps: dict[str, int] | None = None
    if time_column is not None and window is not None:
        stamps = {
            "val_vs_train": _violations(splits["val"], [splits["train"]], time_column, window),
            "test_vs_fitted": _violations(
                splits["test"], [splits["train"], splits["val"]], time_column, window
            ),
        }
    groups: dict[str, int] | None = None
    if group_column is not None:
        sets = {k: set(splits[k][group_column].unique().to_list()) for k in splits}
        groups = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in _PAIRS}
    return {
        "time_column": time_column,
        "window": window,
        "stamp_violations": stamps,
        "group_column": group_column,
        "group_overlap": groups,
    }


def leakage_audit_counts(
    parts: Mapping[str, pl.DataFrame],
    *,
    time_column: str | None,
    group_column: str | None,
    window: int | None,
    weight: str = "rows",
) -> dict[str, Any]:
    """:func:`leakage_audit` from row counts (PR-054): each partition as a tally of
    ``(time_column, group_column, weight)`` — the per-day batch fit never holds its
    partitions — giving the same record as the audit of the partitions' rows."""
    live = {k: v.filter(pl.col(weight) > 0) for k, v in parts.items()}
    stamps: dict[str, int] | None = None
    if time_column is not None and window is not None:

        def fitted(names: tuple[str, ...]) -> NDArray[Any]:
            arrays = [live[k][time_column].to_numpy() for k in names if live[k].height]
            return np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.int64)

        def count(later: str, names: tuple[str, ...]) -> int:
            p = live[later]
            return _within(p[time_column].to_numpy(), p[weight].to_numpy(), fitted(names), window)

        stamps = {
            "val_vs_train": count("val", ("train",)),
            "test_vs_fitted": count("test", ("train", "val")),
        }
    groups: dict[str, int] | None = None
    if group_column is not None:
        sets = {k: set(live[k][group_column].unique().to_list()) for k in live}
        groups = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in _PAIRS}
    return {
        "time_column": time_column,
        "window": window,
        "stamp_violations": stamps,
        "group_column": group_column,
        "group_overlap": groups,
    }


def assert_regime_clean(split_kind: str, audit: Mapping[str, Any]) -> None:
    """Refuse a split whose regime's promise the audit does not confirm.

    ``time_ordered``: no fitted stamp within the window of a later row's stamp.
    ``symbol_holdout``: no group in two partitions. ``random``: nothing promised.
    An axis the regime promises but the audit could not measure is refused too.
    """
    if split_kind == "time_ordered":
        v = audit["stamp_violations"]
        if v is None:
            msg = "time_ordered split: stamps not audited (no time_column or window)"
            raise LeakageError(msg)
        if any(v.values()):
            msg = f"time_ordered split: stamp violations {v} within window {audit['window']}"
            raise LeakageError(msg)
    elif split_kind == "symbol_holdout":
        g = audit["group_overlap"]
        if g is None:
            msg = "symbol_holdout split: groups not audited (no group_column)"
            raise LeakageError(msg)
        if any(g.values()):
            msg = f"symbol_holdout split: group overlap {g}"
            raise LeakageError(msg)

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

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))


class LeakageError(ValueError):
    """A split broke the separation its regime promises (a ``ValueError``: exit 2)."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"leakage: {detail} — refused")


def _violations(later: pl.DataFrame, fitted: list[pl.DataFrame], col: str, window: int) -> int:
    stamps = [f[col].to_numpy() for f in fitted if f.height]
    if not later.height or not stamps:
        return 0
    ref = np.unique(np.concatenate(stamps))
    s = later[col].to_numpy()
    i = np.searchsorted(ref, s)
    left = np.abs(s - ref[np.clip(i - 1, 0, ref.size - 1)])
    right = np.abs(ref[np.clip(i, 0, ref.size - 1)] - s)
    return int((np.minimum(left, right) <= window).sum())


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

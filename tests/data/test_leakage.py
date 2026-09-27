"""PR-045: the leakage audit of a one-off split (program v0.3 C9's leakage tests).

C9: "leakage tests on the real set (no holdout coin's row in train; no training
stamp inside a test stamp's embargo)". The audit is a different algorithm from
the splitters (nearest fitted stamp by binary search; group-set intersection), so
it checks them rather than restating them. Expected counts are hand-computed.
"""

from __future__ import annotations

import polars as pl
import pytest

from rux_ml.data.leakage import LeakageError, assert_regime_clean, leakage_audit
from rux_ml.data.splits import (
    symbol_holdout_split,
    temporal_train_val_test_split,
    train_val_test_split,
)

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def _frame(stamps: list[int], coins: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"t": stamps, "coin": coins, "x": [0.0] * len(stamps)})


def test_hand_computed_counts() -> None:
    """Fitted (train + val) stamps {0, 10, 20} and {22}; test stamps {25, 26, 60}, window 5:
    25 and 26 lie within 5 of 22 (and 25 of 20), 60 of nothing -> 2 test rows violate.
    Val stamp 22 lies within 5 of train stamp 20 -> 1 val row violates.
    Coins: train {A, B}, val {B}, test {C} -> overlaps train&val = 1, others 0."""
    splits = {
        "train": _frame([0, 10, 20], ["A", "A", "B"]),
        "val": _frame([22], ["B"]),
        "test": _frame([25, 26, 60], ["C", "C", "C"]),
    }
    a = leakage_audit(splits, time_column="t", group_column="coin", window=5)
    assert a["stamp_violations"] == {"val_vs_train": 1, "test_vs_fitted": 2}
    assert a["group_overlap"] == {"train&val": 1, "train&test": 0, "val&test": 0}
    assert a["window"] == 5


def _panel() -> pl.DataFrame:
    coins = [f"C{i:02d}" for i in range(24)]
    return pl.DataFrame(
        {
            "t": [s * 10 for s in range(50) for _ in coins],
            "coin": [c for _ in range(50) for c in coins],
            "x": [0.0] * (50 * len(coins)),
        }
    )


def test_row_random_leaks_on_both_axes() -> None:
    parts = train_val_test_split(_panel(), ratios=RATIOS, seed=1)
    a = leakage_audit(parts, time_column="t", group_column="coin", window=30)
    assert a["stamp_violations"]["test_vs_fitted"] > 0
    assert a["group_overlap"]["train&test"] > 0
    assert_regime_clean("random", a)  # the diagnostic regime promises nothing


def test_time_block_purged_has_no_fitted_stamp_inside_a_test_stamps_window() -> None:
    parts = temporal_train_val_test_split(_panel(), time_column="t", ratios=RATIOS, embargo=30)
    a = leakage_audit(parts, time_column="t", group_column="coin", window=30)
    assert a["stamp_violations"] == {"val_vs_train": 0, "test_vs_fitted": 0}
    assert_regime_clean("time_ordered", a)
    # without the embargo the boundary stamps leak
    legacy = temporal_train_val_test_split(_panel(), time_column="t", ratios=RATIOS)
    b = leakage_audit(legacy, time_column="t", group_column="coin", window=30)
    assert b["stamp_violations"]["test_vs_fitted"] > 0
    with pytest.raises(LeakageError, match="stamp"):
        assert_regime_clean("time_ordered", b)


def test_symbol_holdout_has_no_holdout_coin_in_train() -> None:
    parts = symbol_holdout_split(_panel(), group_column="coin", ratios=RATIOS, seed=3)
    a = leakage_audit(parts, time_column="t", group_column="coin", window=30)
    assert a["group_overlap"] == {"train&val": 0, "train&test": 0, "val&test": 0}
    assert a["stamp_violations"]["test_vs_fitted"] > 0  # contemporaneous by design
    assert_regime_clean("symbol_holdout", a)
    leaky = {**parts, "test": pl.concat([parts["test"], parts["train"].head(1)])}
    with pytest.raises(LeakageError, match="group"):
        assert_regime_clean(
            "symbol_holdout", leakage_audit(leaky, time_column="t", group_column="coin", window=30)
        )


def test_axes_without_a_column_are_not_audited() -> None:
    parts = train_val_test_split(_panel(), ratios=RATIOS, seed=1)
    a = leakage_audit(parts, time_column=None, group_column=None, window=None)
    assert a["stamp_violations"] is None and a["group_overlap"] is None
    with pytest.raises(LeakageError, match="not audited"):
        assert_regime_clean("time_ordered", a)

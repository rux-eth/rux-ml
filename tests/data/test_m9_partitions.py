"""PR-054 (program PR-027 A1 / A3 / A4): the M9 partition plan, its keyed hash, and the
records built from its per-batch tally instead of from materialised partitions.

The plan is computed from batches (one per day file on the fit path, the whole frame on
the in-memory path); these tests pin that the two layouts give the same plan, that the
plan reproduces rux-ml's splitters where the rule is unchanged, and that the tally-built
leakage audit and label diagnostics equal the frame-built ones.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import pytest

from rux_ml.cli.train import _label_diagnostics  # pyright: ignore[reportPrivateUsage]
from rux_ml.config import DataConfig, M9Config, RuxMLConfig
from rux_ml.data import make_splits, split_definition, symbol_holdout_split
from rux_ml.data.leakage import leakage_audit, leakage_audit_counts
from rux_ml.data.partitions import (
    FOLDS,
    FrameBatches,
    diagnostics_by_batches,
    plan_partitions,
    row_random_unit,
    split_record,
    splitmix64,
)
from rux_ml.data.splits import train_prefix
from tests.data.test_m9_membership_rules import EMBARGO, KEYS, RATIOS, SEED, _mix, _panel, _unit

DIAG = ["d_float", "d_int"]


def _frame() -> pl.DataFrame:
    # 97 stamps: a day boundary inside the panel (96 per day); twelve coins, so the
    # symbol holdout at the M9 seed populates every partition (val SUI, test ADA / AVAX)
    coins = ("BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "XRP", "ADA", "SUI", "HYPE", "BNB", "LTC")
    f = _panel(n_stamps=97, coins=coins)
    n = f.height
    return f.with_columns(
        pl.Series("d_float", [None if i % 5 == 0 else math.sin(i) for i in range(n)]),
        pl.Series("d_int", [i % 11 for i in range(n)], dtype=pl.Int16),
        (pl.col("stamp_ms") // 86_400_000).alias("day"),
    )


def _cfg(split_kind: str, prefix: float | None = None) -> RuxMLConfig:
    # the M9 problems set the group column (the coin) for every regime: the audit reads it
    extra: dict[str, Any] = {"group_column": "coin", "symbol_holdout_seed": 20260926}
    if split_kind == "time_ordered":
        extra["split_embargo"] = EMBARGO
    return RuxMLConfig(
        data=DataConfig(
            target_column="y",
            split_kind=split_kind,  # type: ignore[arg-type]
            time_column="stamp_ms",
            split_ratios=RATIOS,
            train_prefix_frac=prefix,
            **extra,
        ),
        m9=M9Config(
            row_filter_non_null=["y"],
            h_max_ms=EMBARGO,
            row_key_columns=KEYS,
            diagnostic_columns=DIAG,
        ),
    )


def _kept(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("y").is_not_null() & pl.col("y").is_not_nan())


def _by_day(frame: pl.DataFrame) -> FrameBatches:
    return FrameBatches([d for _, d in _kept(frame).group_by("day", maintain_order=True)])


REGIMES = ["time_ordered", "symbol_holdout", "random"]


def test_splitmix64_matches_the_reference_and_the_published_first_output() -> None:
    xs = np.array([0, 1, 2**63, 2**64 - 1, 0x9E3779B97F4A7C15], dtype=np.uint64)
    assert splitmix64(xs).tolist() == [_mix(int(x)) for x in xs]
    assert _mix(0) == 0xE220A8397B1DCDAF  # splitmix64 seeded with 0: the first output


def test_row_random_unit_is_the_reference_bit_for_bit() -> None:
    frame = _panel()
    got = row_random_unit(frame, KEYS, SEED)
    assert got.tolist() == [_unit(r, SEED) for r in frame.iter_rows(named=True)]


@pytest.mark.parametrize(
    ("column", "message"),
    [
        (pl.Series("coin", ["BTC", None]), "null"),
        (pl.Series("p_bp", [1.0, float("nan")]), "NaN"),
        (pl.Series("p_bp", [[1.0], [2.0]]), "dtype"),
    ],
)
def test_a_row_key_that_cannot_identify_a_row_is_refused(column: pl.Series, message: str) -> None:
    frame = pl.DataFrame({"stamp_ms": [1, 2]}).with_columns(column)
    with pytest.raises(ValueError, match=message):
        row_random_unit(frame, ["stamp_ms", column.name], SEED)


@pytest.mark.parametrize("prefix", [None, 0.5])
@pytest.mark.parametrize("regime", REGIMES)
def test_the_plan_over_day_batches_equals_the_plan_over_the_frame(
    regime: str, prefix: float | None
) -> None:
    frame = _frame()
    cfg = _cfg(regime, prefix)
    whole = plan_partitions(cfg, FrameBatches([_kept(frame)]), seed=SEED)
    days = plan_partitions(cfg, _by_day(frame), seed=SEED)
    assert len(_by_day(frame).frames) == 2
    assert np.array_equal(whole.assign(_kept(frame)), days.assign(_kept(frame)))
    assert whole.tally.sort(whole.tally.columns).equals(days.tally.sort(days.tally.columns))
    record = split_record(cfg, days, rows_before_filter=frame.height)
    assert record == split_record(cfg, whole, rows_before_filter=frame.height)
    # the in-memory split and its record are the same rule applied to the whole frame
    parts = make_splits(cfg, frame, seed=SEED)
    fold = days.assign(_kept(frame))
    for i, k in enumerate(FOLDS):
        assert (
            parts[k]["row"].to_list() == _kept(frame).filter(pl.Series(fold == i))["row"].to_list()
        )
    assert record == split_definition(cfg, frame, parts, seed=SEED)


def test_the_symbol_plan_is_the_symbol_holdout_split() -> None:
    frame = _frame()
    parts = make_splits(_cfg("symbol_holdout"), frame, seed=SEED)
    ref = symbol_holdout_split(_kept(frame), group_column="coin", ratios=RATIOS, seed=20260926)
    assert all(parts[k]["row"].to_list() == ref[k]["row"].to_list() for k in FOLDS)


def test_the_time_prefix_is_train_prefix_of_the_plans_train() -> None:
    frame = _frame()
    full = make_splits(_cfg("time_ordered"), frame, seed=SEED)
    half = make_splits(_cfg("time_ordered", 0.5), frame, seed=SEED)
    ref = train_prefix(full["train"], time_column="stamp_ms", frac=0.5)
    assert half["train"]["row"].to_list() == ref["row"].to_list()
    assert half["val"].equals(full["val"]) and half["test"].equals(full["test"])


@pytest.mark.parametrize("regime", REGIMES)
def test_the_tally_audit_equals_the_audit_of_the_partitions(regime: str) -> None:
    frame = _frame()
    cfg = _cfg(regime)
    plan = plan_partitions(cfg, _by_day(frame), seed=SEED)
    parts = make_splits(cfg, frame, seed=SEED)
    for group in (None, "coin"):
        kw: dict[str, Any] = {"time_column": "stamp_ms", "group_column": group, "window": EMBARGO}
        assert leakage_audit_counts(plan.parts(), **kw) == leakage_audit(parts, **kw)


@pytest.mark.parametrize("regime", REGIMES)
def test_the_diagnostics_pass_equals_the_frame_summary(regime: str) -> None:
    frame = _frame()
    cfg = _cfg(regime)
    plan = plan_partitions(cfg, _by_day(frame), seed=SEED)
    got = diagnostics_by_batches(_by_day(frame), plan, DIAG)
    want = _label_diagnostics(make_splits(cfg, frame, seed=SEED), DIAG)
    assert set(got) == set(want) == {"train", "val"}
    for fold, cols in want.items():
        for c, stats in cols.items():
            for k, v in stats.items():
                g = got[fold][c][k]
                assert (g is None) == (v is None), (fold, c, k)
                if v is not None and g is not None:
                    assert g == pytest.approx(v, rel=1e-12, abs=1e-12), (fold, c, k)
    with pytest.raises(ValueError, match="diagnostic_columns"):
        diagnostics_by_batches(_by_day(frame), plan, ["y__absent"])

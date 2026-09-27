"""PR-042: the symbol-holdout one-off split kind and the embargoed temporal split.

rux-capital program v0.3 D41: three evaluation regimes — row-random (diagnostic),
time-block purged with an embargo >= the longest label horizon, and symbol holdout
("a new one-off rux-ml split kind by coin"). The expected assignments below are
computed from the documented contract (sha256 of ``"<seed>:<group>"`` -> [0, 1)
against the cumulative ratios), independently of the implementation.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from rux_ml.config import DataConfig, M9Config, RuxMLConfig
from rux_ml.config.root import cfg_hash, layer_cfg_hash
from rux_ml.data.splits import (
    make_splits,
    split_definition,
    symbol_holdout_split,
    temporal_train_val_test_split,
)

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
COINS = [f"C{i:02d}" for i in range(30)]


def _expected_part(seed: int, coin: str, ratios: dict[str, float]) -> str:
    u = int.from_bytes(hashlib.sha256(f"{seed}:{coin}".encode()).digest()[:8], "big") / 2**64
    if u < ratios["train"]:
        return "train"
    if u < ratios["train"] + ratios["val"]:
        return "val"
    return "test"


def _panel(coins: list[str], stamps: int = 4) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stamp_ms": [t for t in range(stamps) for _ in coins],
            "coin": [c for _ in range(stamps) for c in coins],
            "x": [float(i) for i in range(stamps * len(coins))],
        }
    )


# ---------- symbol holdout ----------


def test_each_coin_lands_whole_in_the_partition_its_hash_names() -> None:
    parts = symbol_holdout_split(_panel(COINS), group_column="coin", ratios=RATIOS, seed=7)
    for name, frame in parts.items():
        for coin in frame["coin"].unique().to_list():
            assert _expected_part(7, coin, RATIOS) == name, (coin, name)
    assert sum(f.height for f in parts.values()) == 4 * len(COINS)
    seen = [set(f["coin"].to_list()) for f in parts.values()]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    # every row of a coin travels with it
    for frame in parts.values():
        assert (frame.group_by("coin").len()["len"] == 4).all()


def test_a_coins_partition_does_not_depend_on_the_universe_or_row_order() -> None:
    """Nested prefixes and the three targets see different coin universes; a coin
    must hold out the same way in every one of them (the learning curve's premise)."""
    full = symbol_holdout_split(_panel(COINS), group_column="coin", ratios=RATIOS, seed=7)
    sub = symbol_holdout_split(
        _panel(COINS[::3]).reverse(), group_column="coin", ratios=RATIOS, seed=7
    )
    for name in ("train", "val", "test"):
        assert set(sub[name]["coin"].to_list()) <= set(full[name]["coin"].to_list())


def test_the_seed_is_the_assignment() -> None:
    a = symbol_holdout_split(_panel(COINS), group_column="coin", ratios=RATIOS, seed=7)
    b = symbol_holdout_split(_panel(COINS), group_column="coin", ratios=RATIOS, seed=8)
    assert set(a["test"]["coin"].to_list()) != set(b["test"]["coin"].to_list())


def test_symbol_holdout_refuses_a_missing_or_null_group_column() -> None:
    with pytest.raises(ValueError, match="group_column"):
        symbol_holdout_split(_panel(COINS), group_column="asset", ratios=RATIOS, seed=7)
    df = _panel(COINS).with_columns(
        pl.when(pl.col("x") == 0.0).then(None).otherwise(pl.col("coin")).alias("coin")
    )
    with pytest.raises(ValueError, match="null"):
        symbol_holdout_split(df, group_column="coin", ratios=RATIOS, seed=7)


def test_symbol_holdout_refuses_an_empty_partition_with_a_positive_ratio() -> None:
    """Two coins cannot fill three partitions: a holdout with no test coin is refused."""
    with pytest.raises(ValueError, match="no group"):
        symbol_holdout_split(_panel(["A", "B"]), group_column="coin", ratios=RATIOS, seed=7)


# ---------- the embargoed temporal split ----------


def test_embargo_purges_rows_whose_horizon_reaches_the_next_partition() -> None:
    """Integer stamps 0..19, one row each, ratios .6/.2/.2 -> train 0..11, val 12..15,
    test 16..19 by row count. Embargo e keeps a row at t only if t + e < the next
    partition's first stamp (program D45 #4's 't + h_max < the split')."""
    df = pl.DataFrame({"t": list(range(20)), "x": [0.0] * 20})
    parts = temporal_train_val_test_split(df, time_column="t", ratios=RATIOS, embargo=3)
    assert parts["train"]["t"].to_list() == list(range(9))  # 8 + 3 < 12, 9 + 3 = 12 purged
    assert parts["val"]["t"].to_list() == [12]  # 12 + 3 < 16; 13 + 3 = 16 purged
    assert parts["test"]["t"].to_list() == list(range(16, 20))


def test_embargo_zero_separates_a_shared_boundary_stamp_and_none_keeps_legacy_slicing() -> None:
    """A stacked panel: two rows per stamp, 5 stamps; row-count slicing at .5 / .3 / .2
    puts one row of stamp 2 in train and the other in val — the legacy behaviour shares
    a boundary stamp; embargo 0 makes the separation timestamp-atomic."""
    df = pl.DataFrame({"t": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4], "x": [0.0] * 10})
    ratios = {"train": 0.5, "val": 0.3, "test": 0.2}
    legacy = temporal_train_val_test_split(df, time_column="t", ratios=ratios)
    assert legacy["train"]["t"].to_list() == [0, 0, 1, 1, 2]
    assert legacy["val"]["t"].to_list() == [2, 3, 3]
    purged = temporal_train_val_test_split(df, time_column="t", ratios=ratios, embargo=0)
    assert purged["train"]["t"].to_list() == [0, 0, 1, 1]
    assert purged["val"]["t"].to_list() == [2, 3, 3]
    assert purged["test"]["t"].to_list() == [4, 4]


def test_embargo_refuses_a_non_integer_time_column_and_a_fully_purged_train() -> None:
    df = pl.DataFrame({"t": [float(i) for i in range(20)], "x": [0.0] * 20})
    with pytest.raises(ValueError, match="integer"):
        temporal_train_val_test_split(df, time_column="t", ratios=RATIOS, embargo=3)
    df = pl.DataFrame({"t": list(range(20)), "x": [0.0] * 20})
    with pytest.raises(ValueError, match="purged every train row"):
        temporal_train_val_test_split(df, time_column="t", ratios=RATIOS, embargo=100)


# ---------- config + dispatch ----------


def _cfg(**data: object) -> RuxMLConfig:
    return RuxMLConfig(data=DataConfig(target_column="y", **data))  # type: ignore[arg-type]


def test_config_requires_group_column_and_seed_for_symbol_holdout() -> None:
    with pytest.raises(ValidationError, match="group_column"):
        _cfg(split_kind="symbol_holdout", symbol_holdout_seed=1)
    with pytest.raises(ValidationError, match="symbol_holdout_seed"):
        _cfg(split_kind="symbol_holdout", group_column="coin")
    with pytest.raises(ValidationError):
        _cfg(split_kind="time_ordered", time_column="t", split_embargo=-1)


def test_make_splits_dispatches_the_new_kinds() -> None:
    cfg = _cfg(
        split_kind="symbol_holdout", group_column="coin", symbol_holdout_seed=7, split_ratios=RATIOS
    )
    parts = make_splits(cfg, _panel(COINS), seed=123)  # the trial seed is NOT the assignment
    direct = symbol_holdout_split(_panel(COINS), group_column="coin", ratios=RATIOS, seed=7)
    assert all(parts[k].equals(direct[k]) for k in ("train", "val", "test"))
    cfg = _cfg(split_kind="time_ordered", time_column="t", split_embargo=3, split_ratios=RATIOS)
    df = pl.DataFrame({"t": list(range(20)), "x": [0.0] * 20})
    assert make_splits(cfg, df, seed=0)["train"].height == 9


def test_new_fields_are_hash_neutral_while_unset_and_identity_once_set() -> None:
    base = _cfg()
    assert layer_cfg_hash(base, "data") == layer_cfg_hash(
        RuxMLConfig(data=DataConfig(target_column="y")), "data"
    )
    for extra in (
        {"split_kind": "symbol_holdout", "group_column": "coin", "symbol_holdout_seed": 1},
        {"split_kind": "time_ordered", "time_column": "t", "split_embargo": 5},
    ):
        assert cfg_hash(_cfg(**extra)) != cfg_hash(base)


# ---------- [m9] h_max_ms: the embargo covers the longest label horizon ----------


def test_m9_time_block_regime_requires_an_embargo_of_at_least_h_max() -> None:
    kw = {"split_kind": "time_ordered", "time_column": "t"}
    with pytest.raises(ValidationError, match="h_max_ms"):
        RuxMLConfig(data=DataConfig(target_column="y", **kw), m9=M9Config())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="split_embargo"):
        RuxMLConfig(
            data=DataConfig(target_column="y", split_embargo=99, **kw),  # type: ignore[arg-type]
            m9=M9Config(h_max_ms=100),
        )
    ok = RuxMLConfig(
        data=DataConfig(target_column="y", split_embargo=100, **kw),  # type: ignore[arg-type]
        m9=M9Config(h_max_ms=100),
    )
    assert ok.m9 is not None and ok.m9.h_max_ms == 100
    # row-random (diagnostic) and symbol holdout carry no time boundary
    RuxMLConfig(data=DataConfig(target_column="y"), m9=M9Config(h_max_ms=100))


def test_repo_m9_problems_embargo_at_least_their_h_max(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2] / "configs"
    for problem in ("m9_fill_frac", "m9_markout_bp", "m9_walk_bp"):
        cfg = RuxMLConfig.from_layers(
            repo / "base.toml", problem=problem, problems_dir=repo / "problems"
        )
        assert cfg.m9 is not None and cfg.m9.h_max_ms is not None
        assert cfg.data.split_embargo is not None
        assert cfg.data.split_embargo >= cfg.m9.h_max_ms
        assert cfg.data.group_column == "coin"
        assert cfg.data.symbol_holdout_seed is not None


# ---------- h_max is the label's REACH (program PR-024 §Findings Q5, amendment A3) ----------

# A filled order's markout reads mid(tau + h_mk) with tau <= t + L + h_fill, so the label
# reads prices up to t + reach. Every term is an existing program key (A3, operator-approved
# 2026-09-26); the constants below are the test's independent derivation of the pin.
ENTRY_LATENCY_MS = 1_000  # entry_ns: the order reaches the book 1 s after the decision
RESTING_HORIZON_MS = 14_400_000  # the order rests 4 h (ladder_ttl_intervals x timeframe)
MAX_MARKOUT_HORIZON_MS = 14_400_000  # max(markout_horizons_ms) = 4 h (program D45 #3)
LABEL_GRID_MS = 900_000  # the 15-min label grid
MIN_EMBARGO_BARS = 33  # ceil(28,801,000 / 900,000), program PR-024 A3 (not 16)


def test_repo_m9_problems_h_max_is_the_label_reach_not_the_markout_horizon() -> None:
    reach = ENTRY_LATENCY_MS + RESTING_HORIZON_MS + MAX_MARKOUT_HORIZON_MS
    assert reach == 28_801_000
    assert math.ceil(reach / LABEL_GRID_MS) == MIN_EMBARGO_BARS
    repo = Path(__file__).resolve().parents[2] / "configs"
    for problem in ("m9_fill_frac", "m9_markout_bp", "m9_walk_bp"):
        cfg = RuxMLConfig.from_layers(
            repo / "base.toml", problem=problem, problems_dir=repo / "problems"
        )
        assert cfg.m9 is not None and cfg.data.split_embargo is not None
        assert cfg.m9.h_max_ms == reach, problem
        assert cfg.data.split_embargo >= reach, problem
        assert math.ceil(cfg.data.split_embargo / LABEL_GRID_MS) >= MIN_EMBARGO_BARS, problem


def test_a_label_stamped_between_the_markout_horizon_and_the_reach_is_purged() -> None:
    """The A3 leak: a row stamped 4-8 h before the next partition survives a 4 h embargo
    while its label reads past the boundary; the reach-sized embargo purges it."""
    hour = 3_600_000
    boundary = 100 * hour
    stamps = [0, boundary - 6 * hour, *[boundary + i * hour for i in range(18)]]
    df = pl.DataFrame({"t": stamps, "y": [float(i) for i in range(len(stamps))]})
    ratios = {"train": 0.1, "val": 0.45, "test": 0.45}
    four_h = temporal_train_val_test_split(
        df, time_column="t", ratios=ratios, embargo=MAX_MARKOUT_HORIZON_MS
    )
    assert boundary - 6 * hour in four_h["train"]["t"].to_list()  # the leak
    reach = ENTRY_LATENCY_MS + RESTING_HORIZON_MS + MAX_MARKOUT_HORIZON_MS
    full = temporal_train_val_test_split(df, time_column="t", ratios=ratios, embargo=reach)
    assert full["train"]["t"].to_list() == [0]


def test_split_definition_does_not_count_row_filtered_rows_as_purged() -> None:
    """``[m9] row_filter_non_null`` drops rows before the split (program PR-024 A9);
    the record reports them apart from the rows the embargo purged."""
    df = pl.DataFrame(
        {"t": list(range(20)), "y": [None if i % 4 == 0 else float(i) for i in range(20)]}
    )
    data = DataConfig(
        target_column="y", split_kind="time_ordered", time_column="t", split_embargo=1
    )
    cfg = RuxMLConfig(data=data, m9=M9Config(h_max_ms=1, row_filter_non_null=["y"]))
    splits = make_splits(cfg, df, seed=0)
    record = split_definition(cfg, df, splits)
    kept = sum(p.height for p in splits.values())
    assert record["row_filter_dropped"] == 5
    assert record["purged_rows"] == 15 - kept

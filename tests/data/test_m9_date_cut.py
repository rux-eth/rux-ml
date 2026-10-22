"""PR-065 (rumpy program v0.7 D63): the ``[m9]`` time-block split with the cut given as a date.

``data.split_cut_date`` replaces the fraction-chosen cut of a val-less ``[m9]``
``time_ordered`` split with a date's UTC midnight: train = rows with ``stamp + embargo <
cut``, test = rows with ``stamp >= cut`` (ties to test), the rest purged — the fold rule
of PR-054 unchanged, only its cut's source. These tests pin the date mode to the
fraction mode wherever the two cuts coincide, the per-day batches to the whole frame,
the boundary rows, the refusals, hash neutrality, and the committed fixture a harness
test replays through ``partition_at_cut``.
"""

from __future__ import annotations

import datetime as dt
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from rux_ml.config import DataConfig, M9Config, RuxMLConfig
from rux_ml.config.root import cfg_hash, layer_cfg_hash
from rux_ml.data import make_splits, split_definition
from rux_ml.data.partitions import (
    FOLDS,
    PURGED,
    TEST,
    TRAIN,
    FrameBatches,
    plan_partitions,
    split_record,
)
from tests.config.test_oracle_config import MIN_DATA_CFG_HASH, MIN_ROOT_CFG_HASH
from tests.data import pr065_fixture as fx

DAY = fx.DAY_MS
VALLESS = {"train": 0.7, "val": 0.0, "test": 0.3}
DATE_KEYS = ("source", "cut_date", "cut_stamp")


def _cfg(
    cut: dt.date | str | None,
    *,
    h_days: int = 7,
    ratios: dict[str, float] | None = None,
    kind: str = "time_ordered",
    prefix: float | None = None,
    m9: bool = True,
) -> RuxMLConfig:
    extra: dict[str, Any] = {}
    if kind == "symbol_holdout":
        extra = {"group_column": "coin", "symbol_holdout_seed": 1}
    return RuxMLConfig(
        data=DataConfig(
            target_column="y",
            split_kind=kind,  # type: ignore[arg-type]
            time_column="stamp_ms",
            split_ratios=dict(ratios or VALLESS),
            split_embargo=h_days * DAY,
            split_cut_date=cut,  # type: ignore[arg-type]
            train_prefix_frac=prefix,
            **extra,
        ),
        m9=M9Config(row_filter_non_null=["y"], h_max_ms=h_days * DAY) if m9 else None,
    )


def _events() -> pl.DataFrame:
    frame = fx.tables()["events"]
    return frame.with_columns((pl.col("stamp_ms") // DAY).alias("day"))


def _kept(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("y").is_not_null())


def _by_day(frame: pl.DataFrame) -> FrameBatches:
    return FrameBatches([d for _, d in _kept(frame).group_by("day", maintain_order=True)])


def _without_date_keys(record: dict[str, object]) -> dict[str, object]:
    cuts = dict(record["cuts"])  # type: ignore[call-overload]
    return {**record, "cuts": {k: v for k, v in cuts.items() if k not in DATE_KEYS}}


# ---------------------------------------------------------------- the field


def test_the_cut_date_is_its_utc_midnight_in_epoch_ms() -> None:
    assert DataConfig(split_cut_date=dt.date(2025, 11, 26)).split_cut_stamp == 1_764_115_200_000
    assert DataConfig(split_cut_date="2025-11-26").split_cut_stamp == 1_764_115_200_000  # type: ignore[arg-type]
    toml = tomllib.loads("[data]\nsplit_cut_date = 2025-11-26\n")["data"]
    assert DataConfig.model_validate(toml).split_cut_date == dt.date(2025, 11, 26)
    assert DataConfig().split_cut_date is None and DataConfig().split_cut_stamp is None


_BASE_TOML = """
[data]
target_column = "y"
split_kind = "time_ordered"
time_column = "stamp_ms"
split_embargo = 86400000
split_ratios = { train = 0.7, val = 0.0, test = 0.3 }
split_cut_date = 2020-01-31

[m9]
row_filter_non_null = ["y"]
h_max_ms = 86400000
"""


def test_the_cut_date_loads_from_toml_env_and_a_dot_path_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.toml"
    base.write_text(_BASE_TOML)
    assert RuxMLConfig.from_layers(base).data.split_cut_stamp == fx.stamp_of(dt.date(2020, 1, 31))
    over = RuxMLConfig.from_layers(base, overrides={"data.split_cut_date": "2020-02-01"})
    assert over.data.split_cut_stamp == fx.stamp_of(dt.date(2020, 2, 1))
    monkeypatch.setenv("RUXML_DATA__SPLIT_CUT_DATE", "2020-02-02")
    assert RuxMLConfig.from_layers(base).data.split_cut_stamp == fx.stamp_of(dt.date(2020, 2, 2))


@pytest.mark.parametrize(
    "value",
    [
        dt.datetime(2025, 11, 26),  # a naive datetime
        dt.datetime(2025, 11, 26, tzinfo=dt.timezone(dt.timedelta(hours=-6))),
        1_764_115_200_000,
        1_764_115_200,
        "2025-11-26T00:00:00",
        "20251126",
    ],
)
def test_a_datetime_an_int_or_a_non_iso_string_is_refused(value: object) -> None:
    with pytest.raises(ValidationError, match="split_cut_date takes a date"):
        DataConfig(split_cut_date=value)  # type: ignore[arg-type]


def test_the_field_is_hash_neutral_while_unset_and_identity_once_set() -> None:
    unset = RuxMLConfig(data=DataConfig(source_path="x.parquet", target_column="y"))
    assert layer_cfg_hash(unset, "data") == MIN_DATA_CFG_HASH
    assert cfg_hash(unset) == MIN_ROOT_CFG_HASH
    a, b = _cfg(dt.date(2020, 1, 31)), _cfg(dt.date(2020, 2, 1))
    assert layer_cfg_hash(a, "data") != layer_cfg_hash(_cfg(None), "data")
    assert layer_cfg_hash(a, "data") != layer_cfg_hash(b, "data")


# ---------------------------------------------------------------- refusals at load


def test_a_val_partition_is_refused_by_name() -> None:
    with pytest.raises(ValidationError, match=r"split_cut_date is val-less only.*val must be 0"):
        _cfg(dt.date(2020, 1, 31), ratios={"train": 0.7, "val": 0.15, "test": 0.15})


@pytest.mark.parametrize("kind", ["random", "symbol_holdout"])
def test_a_kind_other_than_time_ordered_is_refused(kind: str) -> None:
    msg = f"split_cut_date needs data.split_kind 'time_ordered', got '{kind}'"
    with pytest.raises(ValidationError, match=msg):
        _cfg(dt.date(2020, 1, 31), kind=kind)


def test_a_config_without_m9_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"split_cut_date needs an \[m9\] problem"):
        _cfg(dt.date(2020, 1, 31), m9=False)


def test_a_train_prefix_is_refused() -> None:
    with pytest.raises(
        ValidationError, match=r"split_cut_date does not take data\.train_prefix_frac"
    ):
        _cfg(dt.date(2020, 1, 31), prefix=0.5)


# ---------------------------------------------------------------- the rule


@pytest.mark.parametrize("train", [0.3, 0.5, 0.7, 0.85])
@pytest.mark.parametrize("h_days", [1, 7])
def test_date_mode_equals_fraction_mode_at_the_fraction_chosen_cut(
    train: float, h_days: int
) -> None:
    frame = _events()
    kept = _kept(frame)
    ratios = {"train": train, "val": 0.0, "test": 1.0 - train}
    by_frac = plan_partitions(
        _cfg(None, h_days=h_days, ratios=ratios), FrameBatches([kept]), seed=0
    )
    cut = by_frac.test_first_stamp
    assert cut is not None and cut % DAY == 0  # event stamps: the fraction lands on a midnight
    cfg = _cfg(fx.EPOCH + dt.timedelta(days=cut // DAY), h_days=h_days, ratios=ratios)
    by_date = plan_partitions(cfg, FrameBatches([kept]), seed=0)
    assert (by_date.val_first_stamp, by_date.test_first_stamp) == (cut, cut)
    assert np.array_equal(by_date.assign(kept), by_frac.assign(kept))
    assert by_date.tally.sort(by_date.tally.columns).equals(
        by_frac.tally.sort(by_frac.tally.columns)
    )
    frac_parts = make_splits(_cfg(None, h_days=h_days, ratios=ratios), frame, seed=0)
    date_parts = make_splits(cfg, frame, seed=0)
    assert all(date_parts[k].equals(frac_parts[k]) for k in FOLDS)
    frac_record = split_record(
        _cfg(None, h_days=h_days, ratios=ratios), by_frac, rows_before_filter=frame.height
    )
    date_record = split_record(cfg, by_date, rows_before_filter=frame.height)
    assert _without_date_keys(date_record) == frac_record
    assert "source" not in frac_record["cuts"]  # type: ignore[operator]  # fraction records unchanged


@pytest.mark.parametrize("h_days", [1, 7, 14])
def test_the_plan_over_day_batches_equals_the_plan_over_the_frame(h_days: int) -> None:
    frame = _events()
    cfg = _cfg(dt.date(2020, 2, 1), h_days=h_days)
    whole = plan_partitions(cfg, FrameBatches([_kept(frame)]), seed=0)
    days = plan_partitions(cfg, _by_day(frame), seed=0)
    assert len(_by_day(frame).frames) == 60
    assert np.array_equal(whole.assign(_kept(frame)), days.assign(_kept(frame)))
    assert whole.tally.sort(whole.tally.columns).equals(days.tally.sort(days.tally.columns))
    record = split_record(cfg, days, rows_before_filter=frame.height)
    assert record == split_record(cfg, whole, rows_before_filter=frame.height)
    assert record == split_definition(cfg, frame, make_splits(cfg, frame, seed=0), seed=0)


@pytest.mark.parametrize("h_days", [1, 7, 14])
def test_ties_go_to_test_the_cut_minus_h_row_is_purged_and_nulls_are_nowhere(
    h_days: int,
) -> None:
    cut_day = dt.date(2020, 2, 1)
    cut, emb = fx.stamp_of(cut_day), h_days * DAY
    stamps = [
        cut,
        cut,
        cut,
        cut - emb,
        cut - emb - 1,
        cut - 1,
        cut + 1,
        cut - emb - DAY,
        cut - emb - DAY,
    ]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, None]
    frame = pl.DataFrame(
        {"stamp_ms": pl.Series(stamps, dtype=pl.Int64), "coin": [f"C{i}" for i in range(9)], "y": y}
    )
    cfg = _cfg(cut_day, h_days=h_days)
    fold = plan_partitions(cfg, FrameBatches([_kept(frame)]), seed=0).assign(_kept(frame))
    want = [TEST, TEST, TEST, PURGED, TRAIN, PURGED, TEST, TRAIN]
    assert fold.tolist() == want
    parts = make_splits(cfg, frame, seed=0)
    assert parts["val"].height == 0
    assert sum(parts[k].height for k in FOLDS) == 6  # two purged, the null row in no partition
    assert parts["train"]["coin"].to_list() == ["C4", "C7"]


def test_the_record_writes_the_date_and_the_stamp() -> None:
    frame = _events()
    cfg = _cfg(dt.date(2020, 2, 1))
    record = split_definition(cfg, frame, make_splits(cfg, frame, seed=0), seed=0)
    stamp = fx.stamp_of(dt.date(2020, 2, 1))
    assert record["cuts"] == {
        "val_first_stamp": stamp,
        "test_first_stamp": stamp,
        "boundary_ties": "later_partition",
        "source": "date",
        "cut_date": "2020-02-01",
        "cut_stamp": stamp,
    }
    assert record["rows"]["val"] == 0  # type: ignore[index]
    assert record["split_embargo"] == 7 * DAY


# ---------------------------------------------------------------- refusals in the plan


def test_rows_below_the_cut_all_purged_is_the_embargo_refusal() -> None:
    frame = fx.tables()["small_n1"]  # one row at 2020-01-11
    with pytest.raises(ValueError, match=r"split embargo 604800000 purged every train row"):
        make_splits(_cfg(dt.date(2020, 1, 12)), frame, seed=0)


@pytest.mark.parametrize("cut_day", [dt.date(2020, 1, 11), dt.date(2019, 6, 1)])
def test_no_row_below_the_cut_is_the_empty_train_refusal(cut_day: dt.date) -> None:
    frame = fx.tables()["small_n1"]
    with pytest.raises(ValueError, match=f"split cut date {cut_day.isoformat()} .* train is empty"):
        make_splits(_cfg(cut_day), frame, seed=0)


def test_an_empty_frame_is_the_empty_train_refusal() -> None:
    frame = fx.tables()["small_n1"].clear()
    with pytest.raises(ValueError, match="train is empty"):
        make_splits(_cfg(dt.date(2020, 1, 12)), frame, seed=0)


def test_the_plan_refuses_a_val_partition_set_after_load() -> None:
    cfg = _cfg(dt.date(2020, 2, 1))
    cfg.data.split_ratios = {"train": 0.6, "val": 0.1, "test": 0.3}  # skips the root validator
    with pytest.raises(ValueError, match="split_cut_date is val-less only"):
        plan_partitions(cfg, FrameBatches([_kept(_events())]), seed=0)


def test_the_ratios_are_still_sum_checked() -> None:
    cfg = _cfg(dt.date(2020, 2, 1), ratios={"train": 0.7, "val": 0.0, "test": 0.2})
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        plan_partitions(cfg, FrameBatches([_kept(_events())]), seed=0)


# ---------------------------------------------------------------- the committed fixture


def test_the_committed_tables_are_the_generated_tables() -> None:
    committed = fx.read_tables()
    generated = fx.tables()
    assert sorted(committed) == sorted(generated)
    for name, frame in generated.items():
        assert committed[name].equals(frame), name
        assert committed[name]["y"].is_nan().sum() == 0  # the harness reads validity as not-null


def test_the_committed_expected_is_rux_mls_date_mode_output() -> None:
    committed = json.loads(fx.EXPECTED.read_text())
    assert committed == fx.expected(fx.read_tables())
    entries = [v for k, v in committed.items() if k != "_contract"]
    refused = [e["refused"] for e in entries if "refused" in e]
    assert {"embargo_purged_train", "empty_train"} <= set(refused)
    fitted = [e for e in entries if "refused" not in e]
    assert all(e["val"]["rows"] == 0 for e in fitted)
    assert any(e["test"]["rows"] == 0 for e in fitted)  # a cut after every row
    assert len(entries) == len(fx.cases())

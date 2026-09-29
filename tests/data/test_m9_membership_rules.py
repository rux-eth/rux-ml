"""PR-054 (program PR-027 A1 / A4): an [m9] partition is a rule of the row.

Written against the public surface only (``make_splits``, ``split_definition``,
``RuxMLConfig``) with an independent pure-Python reference of the keyed hash, so on
the pre-PR code these tests fail on the behaviour, not on an import:

- **row-random (A4, a ruling):** a row's partition is fixed by a version-stable keyed
  hash of its keys (splitmix64 over the integer-encoded keys, seeded by the split seed),
  not by ``df.sample(shuffle=True)`` — the partition membership changes, and becomes
  reproducible from the row alone, across row orders and library versions;
- **time-block (A1):** the partitions are stamp predicates, so the rows tied on a
  boundary stamp go to the later partition (the row-count slice split them by an
  unstable sort and purged the earlier side);
- the membership rule is recorded in ``split_definition``.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import polars as pl
import pytest

from rux_ml.config import DataConfig, M9Config, RuxMLConfig
from rux_ml.data import make_splits, split_definition, train_val_test_split

KEYS = ["stamp_ms", "coin", "side", "kind", "p_bp", "q_usd", "h_ms"]
SEED = 20260928
RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
EMBARGO = 1_800_000  # two 15-minute stamps

_MASK = (1 << 64) - 1


def _mix(x: int) -> int:
    """splitmix64's finaliser (Vigna), in Python integers: the reference."""
    z = (x + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def _encode(v: object) -> int:
    if isinstance(v, bool):
        raise TypeError(v)
    if isinstance(v, int):
        return v & _MASK  # two's complement
    if isinstance(v, float):
        return struct.unpack(">Q", struct.pack(">d", v + 0.0))[0]  # -0.0 is 0.0
    if isinstance(v, str):
        return int.from_bytes(hashlib.sha256(v.encode()).digest()[:8], "big")
    raise TypeError(type(v))


def _unit(row: dict[str, Any], seed: int) -> float:
    h = _mix(seed)
    for k in KEYS:
        h = _mix(h ^ _encode(row[k]))
    return (h >> 11) / 2**53


def _panel(n_stamps: int = 41, coins: tuple[str, ...] = ("BTC", "ETH", "SOL")) -> pl.DataFrame:
    """A stacked panel: 2 rows per stamp and coin (the sides), a NaN / null target on a
    deterministic subset, one ``-0.0`` price."""
    rows = [(t, c, s) for t in range(n_stamps) for c in coins for s in (1, -1)]
    n = len(rows)
    y: list[float | None] = [
        None if i % 13 == 0 else float("nan") if i % 17 == 0 else (i % 7) / 7 for i in range(n)
    ]
    return pl.DataFrame(
        {
            "stamp_ms": pl.Series(
                [1_700_000_000_000 + t * 900_000 for t, _, _ in rows], dtype=pl.Int64
            ),
            "coin": [c for _, c, _ in rows],
            "side": pl.Series([s for _, _, s in rows], dtype=pl.Int8),
            "kind": ["post_only"] * n,
            "p_bp": [-0.0 if i == 5 else [0.0, 15.0, 30.0][i % 3] for i in range(n)],
            "q_usd": [[100.0, 300.0][i % 2] for i in range(n)],
            "h_ms": pl.Series([14_400_000] * n, dtype=pl.Int64),
            "x": [float(i) for i in range(n)],
            "y": y,
            "row": list(range(n)),
        }
    )


def _cfg(split_kind: str, **m9: Any) -> RuxMLConfig:
    extra: dict[str, Any] = {"split_embargo": EMBARGO} if split_kind == "time_ordered" else {}
    return RuxMLConfig(
        data=DataConfig(
            target_column="y",
            split_kind=split_kind,  # type: ignore[arg-type]
            time_column="stamp_ms",
            split_ratios=RATIOS,
            **extra,
        ),
        m9=M9Config(row_filter_non_null=["y"], h_max_ms=EMBARGO, **m9),
    )


def _rows(parts: dict[str, pl.DataFrame]) -> dict[str, set[int]]:
    return {k: set(v["row"].to_list()) for k, v in parts.items()}


def test_m9_row_random_membership_is_the_keyed_hash_of_the_row() -> None:
    frame = _panel()
    parts = make_splits(_cfg("random", row_key_columns=KEYS), frame, seed=SEED)
    kept = frame.filter(pl.col("y").is_not_null() & pl.col("y").is_not_nan())
    want: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}
    for row in kept.iter_rows(named=True):
        u = _unit(row, SEED)
        fold = (
            "train"
            if u < RATIOS["train"]
            else "val"
            if u < RATIOS["train"] + RATIOS["val"]
            else "test"
        )
        want[fold].add(row["row"])
    assert _rows(parts) == want
    assert all(want.values())  # every partition is populated on the fixture


def test_m9_row_random_membership_does_not_depend_on_row_order_or_the_seed_bag_alone() -> None:
    frame = _panel()
    cfg = _cfg("random", row_key_columns=KEYS)
    forward = _rows(make_splits(cfg, frame, seed=SEED))
    backward = _rows(make_splits(cfg, frame.reverse(), seed=SEED))
    assert forward == backward  # a shuffle of the frame would move rows
    assert _rows(make_splits(cfg, frame, seed=SEED + 1)) != forward  # the seed keys the hash


def test_a_non_m9_random_split_keeps_the_seeded_shuffle() -> None:
    frame = _panel().drop_nulls("y")
    cfg = RuxMLConfig(data=DataConfig(target_column="y", split_ratios=RATIOS))
    got = make_splits(cfg, frame, seed=SEED)
    ref = train_val_test_split(frame, ratios=RATIOS, seed=SEED)
    assert all(got[k].equals(ref[k]) for k in ref)


def test_m9_random_without_row_key_columns_is_refused_at_the_split() -> None:
    with pytest.raises(ValueError, match="row_key_columns"):
        make_splits(_cfg("random"), _panel(), seed=SEED)


def _time_cuts(frame: pl.DataFrame) -> tuple[int, int]:
    kept = frame.filter(pl.col("y").is_not_null() & pl.col("y").is_not_nan())
    st = sorted(kept["stamp_ms"].to_list())
    tr_n = int(len(st) * RATIOS["train"])
    va_n = int(len(st) * RATIOS["val"])
    return st[tr_n], st[tr_n + va_n]


def test_m9_time_block_partitions_are_stamp_predicates_with_ties_to_the_later_partition() -> None:
    frame = _panel()
    t_val, t_test = _time_cuts(frame)
    kept = frame.filter(pl.col("y").is_not_null() & pl.col("y").is_not_nan())
    # the fixture's cuts land inside a stamp: rows tied on a boundary stamp exist on both sides
    st = sorted(kept["stamp_ms"].to_list())
    tr_n = int(len(st) * RATIOS["train"])
    assert st[tr_n - 1] == st[tr_n] == t_val
    s = pl.col("stamp_ms")
    want = {
        "train": kept.filter(s + EMBARGO < t_val),
        "val": kept.filter((s >= t_val) & (s + EMBARGO < t_test)),
        "test": kept.filter(s >= t_test),
    }
    parts = make_splits(_cfg("time_ordered"), frame, seed=SEED)
    assert _rows(parts) == _rows(want)
    assert set(kept.filter(s == t_val)["row"].to_list()) <= _rows(parts)["val"]


@pytest.mark.parametrize("split_kind", ["random", "time_ordered"])
def test_m9_split_definition_records_the_membership_rule(split_kind: str) -> None:
    frame = _panel()
    cfg = _cfg(split_kind, row_key_columns=KEYS)
    parts = make_splits(cfg, frame, seed=SEED)
    record = split_definition(cfg, frame, parts, seed=SEED)
    assert record["rows"] == {k: v.height for k, v in parts.items()}
    if split_kind == "random":
        assert record["membership"] == {
            "rule": "keyed_hash",
            "hash": "splitmix64",
            "key_columns": KEYS,
            "seed": SEED,
        }
    else:
        t_val, t_test = _time_cuts(frame)
        assert record["cuts"] == {
            "val_first_stamp": t_val,
            "test_first_stamp": t_test,
            "boundary_ties": "later_partition",
        }

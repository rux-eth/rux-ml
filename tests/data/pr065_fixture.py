"""PR-065: the committed fixture of the ``[m9]`` date-cut split, for a harness replay.

The rumpy harness re-implements this split as ``probe/partition.py`` ``partition_at_cut``
(program PR-102). As PR-082 did for the fraction split (harness ``tests/fixtures/pr082/``),
this module writes made-up parquet tables and rux-ml's own date-mode output over them —
``make_splits`` on the in-memory ``[m9]`` path, read back from the committed files — to
``tests/fixtures/pr065/``, so a harness test can replay the tables through
``partition_at_cut`` and compare row for row. ``tests/data/test_m9_date_cut.py`` pins the
committed files to this module's output.

Regenerate (only when the rule changes on purpose)::

    uv run python -m tests.data.pr065_fixture
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from rux_ml.config import DataConfig, M9Config, RuxMLConfig
from rux_ml.data import make_splits

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pr065"
EXPECTED = FIXTURE / "partition_at_cut_expected.json"
DAY_MS = 86_400_000
EPOCH = dt.date(1970, 1, 1)
H_DAYS = (1, 7, 14)
RATIOS = {"train": 0.7, "val": 0.0, "test": 0.3}  # val-less; train / test unused for the cut
D0 = dt.date(2020, 1, 1)  # made-up days, far from any real window
COINS = tuple(f"C{i:02d}" for i in range(6))

CONTRACT = (
    "rux-ml PR-065 date-cut split (data.split_cut_date, val-less [m9] time_ordered) over the "
    "tables in partition/, computed by rux-ml's make_splits. Key '<table>|h<h>|cut<YYYY-MM-DD>': "
    "split_embargo = h * 86,400,000 ms, cut = the date's UTC midnight in epoch ms. A row is valid "
    "iff y is not null (the tables carry no NaN: pyarrow's is_valid counts NaN valid, rux-ml's "
    "row filter does not). Row key = 'stamp_ms|coin' or 'stamp_ms|coin|event_id' when the table "
    "has event_id; sha256 over the sorted keys joined by newlines. 'refused' cases: "
    "'embargo_purged_train' = rows lie below the cut but the embargo keeps none (harness: "
    "PartitionRefused 'partition_embargo_purged_train'); 'empty_train' = no valid row below the "
    "cut (rux-ml refuses a fit; harness partition_at_cut returns an empty train, no refusal)."
)


def stamp_of(day: dt.date) -> int:
    """A day's UTC midnight in epoch ms."""
    return (day - EPOCH).days * DAY_MS


def _y(i: int) -> float | None:
    return None if i % 7 == 3 else ((i * 37) % 101) / 101


def tables() -> dict[str, pl.DataFrame]:
    """The made-up tables: event stamps ``(day + 1) · DAY``, intra-day stamps, ties on
    the cut and on ``cut - h · DAY``, and small n."""
    out: dict[str, pl.DataFrame] = {}
    rows: list[tuple[int, str, str]] = []
    for d in range(60):  # 1-3 events per (day, coin), stamped at the next midnight
        for c, coin in enumerate(COINS):
            rows.extend(
                (stamp_of(D0 + dt.timedelta(days=d + 1)), coin, f"e{k}")
                for k in range(1 + (d + c) % 3)
            )
    out["events"] = _frame(rows)
    quarter = DAY_MS // 96
    out["intraday"] = _frame(
        [(stamp_of(D0) + q * quarter, coin, "") for q in range(10 * 96) for coin in COINS[:3]]
    )
    cut = stamp_of(D0 + dt.timedelta(days=30))
    near = [cut + k * DAY_MS for k in range(-20, 6)]
    near += [cut - h * DAY_MS + e for h in H_DAYS for e in (-1, 0, 1)] + [cut - 1, cut + 1]
    out["tie_on_cut"] = _frame([(s, coin, "") for s in sorted(set(near)) for coin in COINS[:4]])
    for n in (1, 2, 3):
        out[f"small_n{n}"] = _frame(
            [(stamp_of(D0 + dt.timedelta(days=10 + 3 * i)), COINS[i], "") for i in range(n)]
        )
    return out


def _frame(rows: list[tuple[int, str, str]]) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "stamp_ms": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "coin": pl.Series([r[1] for r in rows], dtype=pl.String),
            "y": pl.Series([_y(i) for i in range(len(rows))], dtype=pl.Float64),
        }
    )
    if any(r[2] for r in rows):
        frame = frame.with_columns(pl.Series("event_id", [r[2] for r in rows], dtype=pl.String))
    return frame


def cases() -> list[tuple[str, int, dt.date]]:
    """(table, h days, cut date): inside the data, on the ties, before every row (empty
    train), after every row (empty test), and where the embargo purges all of train."""
    day = lambda k: D0 + dt.timedelta(days=k)  # noqa: E731
    cuts = {
        "events": [day(0), day(20), day(31), day(45), day(70)],
        "intraday": [day(0), day(3), day(5), day(9), day(20)],
        "tie_on_cut": [day(10), day(16), day(23), day(29), day(30), day(31), day(40)],
        "small_n1": [day(10), day(11), day(12)],
        "small_n2": [day(13), day(14), day(20), day(30)],
        "small_n3": [day(13), day(17), day(30)],
    }
    return [(t, h, c) for t, cs in cuts.items() for h in H_DAYS for c in cs]


def config(h_days: int, cut: dt.date) -> RuxMLConfig:
    embargo = h_days * DAY_MS
    return RuxMLConfig(
        data=DataConfig(
            target_column="y",
            split_kind="time_ordered",
            time_column="stamp_ms",
            split_ratios=dict(RATIOS),
            split_embargo=embargo,
            split_cut_date=cut,
        ),
        m9=M9Config(row_filter_non_null=["y"], h_max_ms=embargo),
    )


def _keys(frame: pl.DataFrame) -> list[str]:
    cols = [c for c in ("stamp_ms", "coin", "event_id") if c in frame.columns]
    return ["|".join(str(v) for v in row) for row in frame.select(cols).iter_rows()]


def digest(frame: pl.DataFrame) -> str:
    return hashlib.sha256("\n".join(sorted(_keys(frame))).encode()).hexdigest()


def expected(source: dict[str, pl.DataFrame]) -> dict[str, Any]:
    """rux-ml's date-mode partitions of every case, keyed as the contract says."""
    out: dict[str, Any] = {"_contract": CONTRACT}
    for name, h, cut in cases():
        frame = source[name]
        entry: dict[str, Any] = {
            "n": int(frame["y"].is_not_null().sum()),
            "cut": stamp_of(cut),
            "cut_date": cut.isoformat(),
        }
        try:
            parts = make_splits(config(h, cut), frame, seed=0)
        except ValueError as e:
            text = str(e)
            if "purged every train row" in text:
                entry["refused"] = "embargo_purged_train"
            elif "train is empty" in text:
                entry["refused"] = "empty_train"
            else:
                raise
        else:
            for part in ("train", "val", "test"):
                entry[part] = {"rows": parts[part].height, "sha256": digest(parts[part])}
        out[f"{name}|h{h}|cut{cut.isoformat()}"] = entry
    return out


def read_tables() -> dict[str, pl.DataFrame]:
    return {p.stem: pl.read_parquet(p) for p in sorted((FIXTURE / "partition").glob("*.parquet"))}


def write() -> None:
    (FIXTURE / "partition").mkdir(parents=True, exist_ok=True)
    for name, frame in tables().items():
        frame.write_parquet(FIXTURE / "partition" / f"{name}.parquet")
    EXPECTED.write_text(json.dumps(expected(read_tables()), indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    write()

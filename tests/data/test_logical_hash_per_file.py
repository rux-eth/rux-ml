"""PR-053: the logical data hash computed file by file equals the whole-source digest, bit for bit.

Program PR-027 A2 (Phase 3 Round 2b, ``q3_hash_stream``): the whole-source computation (one scan
over every file, one ``struct(all cols).hash`` and one streaming sort) held 18.0 GiB on the 156-day
fill view and left about 400 B per row resident before the fit. Recomputed per file, the same
digest came out at 7.3 GiB. Row hashes are row-local and the final sort removes order, so the
digest cannot depend on how the rows are split into files.

The reference is ``versioning._logical_hash_whole_source``, the pre-PR-053 body kept verbatim.
The pinned digests were computed by that code on ``dev`` @ ``9e3f0c3`` under Polars 1.40.1. Polars
guarantees its row hash only within one version, so the pins skip under any other version.
"""

from __future__ import annotations

import datetime as dt
import json
import tracemalloc
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

from rux_ml.data import versioning
from rux_ml.data.loaders import iter_parquet_files
from rux_ml.runs import data_hashes
from tests.conftest import repo_oracle_cfg

if TYPE_CHECKING:
    from collections.abc import Callable

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "fixtures" / "golden_v1"
PINNED_POLARS = "1.40.1"
N = 120


def _edge_frame(n: int = N, seed: int = 0) -> pl.DataFrame:
    """Every dtype family a training set could carry, with the float and null edge values."""
    rng = np.random.default_rng(seed)
    f64 = rng.normal(size=n)
    f64[:6] = [np.nan, -0.0, 0.0, np.inf, -np.inf, 1e-308]
    k = np.arange(n)
    return pl.DataFrame(
        {
            "stamp_ms": pl.Series(1_700_000_000_000 + k * 60_000, dtype=pl.Int64),
            "i8": pl.Series((k % 256) - 128, dtype=pl.Int8),
            "u64": pl.Series([2**64 - 1 - j for j in range(n)], dtype=pl.UInt64),
            "f64": f64,
            "f32": pl.Series(f64, dtype=pl.Float32),
            "f64_null": [None if j % 7 == 0 else float(v) for j, v in enumerate(f64)],
            "b": [None if j % 5 == 0 else bool(j % 2) for j in range(n)],
            "s": [
                None if j % 11 == 0 else ("" if j % 13 == 0 else f"c{j % 4}-é✓") for j in range(n)
            ],
            "cat": pl.Series([f"coin{j % 3}" for j in range(n)], dtype=pl.Categorical),
            "enum": pl.Series(
                [("fill", "walk")[j % 2] for j in range(n)], dtype=pl.Enum(["fill", "walk"])
            ),
            "date": [dt.date(2026, 1, 1) + dt.timedelta(days=int(j)) for j in k],
            "ts": pl.Series(1_700_000_000_000 + k, dtype=pl.Datetime("ms", "UTC")),
            "dur": pl.Series(k * 1_000, dtype=pl.Duration("ms")),
            "lst": [[float(j), None] if j % 3 else [] for j in range(n)],
            "st": [{"a": int(j), "b": None if j % 4 == 0 else f"x{j}"} for j in range(n)],
            "bin": [bytes([j % 256]) * (j % 3) for j in range(n)],
        }
    )


def _layouts() -> dict[str, Callable[[pl.DataFrame], dict[str, pl.DataFrame]]]:
    """The same logical rows (or a stated multiset of them) in different physical layouts."""

    def reorder(df: pl.DataFrame, shift: int) -> pl.DataFrame:
        names = df.columns
        return df.select(names[shift:] + names[:shift])

    return {
        "one_file": lambda df: {"all.parquet": df},
        "three_files": lambda df: {
            "a.parquet": df[:40],
            "b.parquet": df[40:90],
            "c.parquet": df[90:],
        },
        "shuffled_uneven_split": lambda df: {
            f"p{i}.parquet": df.sample(fraction=1.0, shuffle=True, seed=1)[lo:hi]
            for i, (lo, hi) in enumerate([(0, 7), (7, 8), (8, 70), (70, N)])
        },
        "column_order_per_file": lambda df: {
            "a.parquet": df[:50],
            "b.parquet": reorder(df[50:100], 5),
            "c.parquet": df[100:].select(df.columns[::-1]),
        },
        "empty_file_first": lambda df: {"a.parquet": df.clear(), "b.parquet": df},
        "empty_file_middle": lambda df: {
            "a.parquet": df[:60],
            "b.parquet": df.clear(),
            "c.parquet": df[60:],
        },
        "only_empty": lambda df: {"a.parquet": df.clear()},
        "duplicates_across_files": lambda df: {"a.parquet": df, "b.parquet": df.head(30)},
        "hive_layout": lambda df: {"k=0/p.parquet": df[:60], "k=1/p.parquet": df[60:]},
        "nested_dirs": lambda df: {
            "a.parquet": df[:30],
            "sub/b.parquet": df[30:80],
            "sub/deeper/c.parquet": df[80:],
        },
    }


def _write(root: Path, parts: dict[str, pl.DataFrame]) -> list[Path]:
    for rel, df in parts.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
    return iter_parquet_files(root)


def _source(tmp_path: Path, layout: str) -> list[Path]:
    return _write(tmp_path / layout, _layouts()[layout](_edge_frame()))


@pytest.mark.parametrize("layout", list(_layouts()))
def test_the_per_file_digest_equals_the_whole_source_digest(tmp_path: Path, layout: str) -> None:
    """Digest, row count and schema: the runtime path equals the pre-PR-053 computation."""
    files = _source(tmp_path, layout)
    assert versioning._logical_hash(files) == versioning._logical_hash_whole_source(files)


def test_the_digest_does_not_depend_on_the_split_into_files(tmp_path: Path) -> None:
    one = versioning._logical_hash(_source(tmp_path, "one_file"))
    for layout in (
        "three_files",
        "shuffled_uneven_split",
        "column_order_per_file",
        "hive_layout",
        "nested_dirs",
    ):
        assert versioning._logical_hash(_source(tmp_path, layout)) == one, layout


def test_a_duplicated_row_is_counted_twice(tmp_path: Path) -> None:
    one = versioning._logical_hash(_source(tmp_path, "one_file"))
    dup = versioning._logical_hash(_source(tmp_path, "duplicates_across_files"))
    assert dup[1] == one[1] + 30
    assert dup[0] != one[0]


@pytest.mark.skipif(pl.__version__ != PINNED_POLARS, reason=f"pinned under Polars {PINNED_POLARS}")
def test_the_recorded_golden_digest_is_reproduced() -> None:
    """``golden_v1/manifest.json`` was written by an earlier rux-ml. Its hashes still hold."""
    recorded = json.loads((GOLDEN / "manifest.json").read_text())
    hashes = data_hashes(GOLDEN / "synthetic.parquet", oracle=repo_oracle_cfg())
    assert hashes["data_logical_hash"] == recorded["data_logical_hash"]
    assert hashes["data_bytes_hash"] == recorded["data_bytes_hash"]


PINNED_EDGE = {
    "three_files": "1cf2a9741daa38783a4adab4605d22a8301bd803f5327374d4e31fdc9745cbff",
    "only_empty": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # sha256(b"")
}


@pytest.mark.skipif(pl.__version__ != PINNED_POLARS, reason=f"pinned under Polars {PINNED_POLARS}")
@pytest.mark.parametrize("layout", list(PINNED_EDGE))
def test_the_digest_recorded_by_the_old_code_is_reproduced(tmp_path: Path, layout: str) -> None:
    assert versioning._logical_hash(_source(tmp_path, layout))[0] == PINNED_EDGE[layout]


def _later_file_variants() -> dict[str, tuple[pl.DataFrame, bool]]:
    """A second file that differs from the first in schema: (frame, whether one scan over every
    file accepts it). Read from Polars 1.40.1; the test asserts the old path agrees."""
    df = _edge_frame()[60:]
    swapped = pl.struct(pl.col("st").struct.field("b"), pl.col("st").struct.field("a"))
    return {
        "column_order": (df.select(df.columns[::-1]), True),
        "struct_field_order": (df.with_columns(st=swapped), True),
        "timezone_naive": (df.with_columns(pl.col("ts").dt.replace_time_zone(None)), True),
        "timezone_other": (
            df.with_columns(pl.col("ts").dt.convert_time_zone("America/Chicago")),
            False,
        ),
        "time_unit": (df.with_columns(pl.col("ts").dt.cast_time_unit("us")), False),
        "int16_for_int8": (df.with_columns(pl.col("i8").cast(pl.Int16)), False),
        "float32_for_float64": (df.with_columns(pl.col("f64").cast(pl.Float32)), False),
        "string_for_categorical": (df.with_columns(pl.col("cat").cast(pl.String)), False),
        "enum_categories": (
            df.with_columns(pl.col("enum").cast(pl.String).cast(pl.Enum(["walk", "fill"]))),
            False,
        ),
        "struct_inner_dtype": (
            df.with_columns(
                st=pl.struct(
                    pl.col("st").struct.field("a").cast(pl.Int32), pl.col("st").struct.field("b")
                )
            ),
            False,
        ),
        "missing_column": (df.drop("f64_null"), False),
        "extra_column": (df.with_columns(extra=pl.lit(1)), False),
    }


@pytest.mark.parametrize("case", list(_later_file_variants()))
def test_a_later_file_with_another_schema_is_treated_as_the_whole_source_scan_treats_it(
    tmp_path: Path, case: str
) -> None:
    """One scan over every file checks each later file against the first file's schema. It
    accepts some differences and refuses the rest. Hashing each file on its own would accept
    them all, so the per-file path scans each file with the first file's schema."""
    frame, accepted = _later_file_variants()[case]
    files = _write(tmp_path, {"a.parquet": _edge_frame()[:60], "b.parquet": frame})
    if accepted:
        assert versioning._logical_hash(files) == versioning._logical_hash_whole_source(files)
        return
    errors = (pl.exceptions.SchemaError, pl.exceptions.ColumnNotFoundError)
    with pytest.raises(errors) as old:
        versioning._logical_hash_whole_source(files)
    with pytest.raises(errors) as new:
        versioning._logical_hash(files)
    assert type(new.value) is type(old.value)
    assert "b.parquet" in str(new.value)


def test_a_null_typed_column_then_a_typed_one_is_refused(tmp_path: Path) -> None:
    files = _write(
        tmp_path,
        {
            "a.parquet": pl.DataFrame({"x": [1, 2], "n": pl.Series([None, None], dtype=pl.Null)}),
            "b.parquet": pl.DataFrame({"x": [3, 4], "n": pl.Series([1, None], dtype=pl.Int64)}),
        },
    )
    with pytest.raises(pl.exceptions.SchemaError):
        versioning._logical_hash_whole_source(files)
    with pytest.raises(pl.exceptions.SchemaError):
        versioning._logical_hash(files)


def test_no_files_is_refused() -> None:
    with pytest.raises(FileNotFoundError, match="no parquet files found"):
        versioning._logical_hash([])


def test_polars_is_handed_one_file_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The memory bound: no scan names more than one file, and no collected frame holds more
    than one file's rows. The whole-source path fails both (one scan of the list; one frame of
    every row)."""
    files = _source(tmp_path, "three_files")
    scanned: list[Any] = []
    heights: list[int] = []
    scan, collect = pl.scan_parquet, pl.LazyFrame.collect

    def spy_scan(source: Any, *args: Any, **kwargs: Any) -> pl.LazyFrame:
        scanned.append(source)
        return scan(source, *args, **kwargs)

    def spy_collect(self: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        out = collect(self, *args, **kwargs)
        heights.append(out.height)
        return out

    monkeypatch.setattr(pl, "scan_parquet", spy_scan)
    monkeypatch.setattr(pl.LazyFrame, "collect", spy_collect)
    _, rows, _ = versioning._logical_hash(files)
    assert rows == N
    assert all(isinstance(s, (str, Path)) for s in scanned), (
        f"a scan named several files: {scanned}"
    )
    assert {Path(s) for s in scanned} == set(files)
    assert max(heights) <= 50, (
        f"a collected frame held {max(heights)} rows; the largest file has 50"
    )


def test_the_hash_array_does_not_outlive_the_call(tmp_path: Path) -> None:
    """The row-hash array is released before the caller goes on to the fit."""
    rows = 300_000
    files = _write(
        tmp_path,
        {
            f"{d}.parquet": pl.DataFrame({"x": np.arange(d * rows // 3, (d + 1) * rows // 3)})
            for d in range(3)
        },
    )
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        _, got, _ = versioning._logical_hash(files)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert got == rows
    assert peak - before >= 8 * rows  # the array was traced, so the check below can see it
    assert current - before < rows  # and it is gone (< 1 B per row left)

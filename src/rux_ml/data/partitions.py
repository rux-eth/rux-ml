"""An ``[m9]`` split as a rule of the row, planned from batches (PR-054; program PR-027 A1/A3/A4).

The in-memory splitters (:mod:`rux_ml.data.splits`) need the whole frame: a row-count
slice of the stamp-sorted frame, a seeded shuffle. An ``[m9]`` fit at 156 days cannot
hold that frame (program PR-027 R2-b: 52 to 137 GiB against a 20 GiB scope), so its
partitions are **rules each row can be checked against on its own** — per-day batches
then carry only their own rows of a partition to XGBoost:

- **time-block** (``time_ordered``): stamp predicates. The first stamps of val and test
  are the stamps at sorted row positions ``int(n·train)`` and ``int(n·(train+val))`` —
  the same cuts as the row-count slice — and train / val keep a row only when
  ``t + embargo`` lies before the next partition's first stamp. Rows tied on a boundary
  stamp go to the **later** partition (the row-count slice split them by an unstable
  sort and purged the earlier side); every other row is where the slice put it.
- **symbol holdout**: each group's fixed point in [0, 1) (:func:`unit_interval`), as
  :func:`rux_ml.data.splits.symbol_holdout_split` — unchanged.
- **row-random** (program PR-027 A4, an operator ruling of 2026-09-28): a version-stable
  keyed hash of the row's keys (``[m9] row_key_columns``) — splitmix64 over the
  integer-encoded keys, seeded by the split seed — in place of ``df.sample(shuffle=True)``.
  **This changes partition membership** (it is reproducible from the row alone, across
  row orders, batchings, and Polars / NumPy versions, so the harness can recompute it);
  partition sizes are the ratios in expectation, not exactly.
- the nested-prefix cut (``data.train_prefix_frac``): train keeps the first
  ``ceil(frac · n)`` of its unique stamps, as :func:`rux_ml.data.splits.train_prefix`.

:func:`plan_partitions` computes the rule's global inputs (the stamp cuts, the group
membership) and a per-batch tally ``(fold, stamp, group, rows)`` in passes that read the
key columns only; :meth:`PartitionPlan.assign` then gives every row of any batch its
fold. The in-memory path (:func:`rux_ml.data.splits.make_splits`) runs the same plan over
the whole frame as one batch, so both paths share the membership by construction. The
split record, the leakage audit and the label diagnostics (A3: their own pass, never in a
training frame) are built from the tally and per-batch passes, never from partitions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import polars as pl

from rux_ml.data.loaders import iter_parquet_files
from rux_ml.data.quarantine import check_oracle_quarantine

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from rux_ml.config import RuxMLConfig
    from rux_ml.config.data import OracleQuarantineConfig

_TOL = 1e-6
FOLDS: tuple[str, str, str] = ("train", "val", "test")
TRAIN, VAL, TEST = 0, 1, 2
PURGED = -1  # in no partition: the embargo, or the train prefix cut
ROWS = "rows"  # the tally's count column

# splitmix64's finaliser (S. Vigna, public domain): the increment and the two multipliers.
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_MUL2 = np.uint64(0x94D049BB133111EB)
_S11, _S27, _S30, _S31 = (np.uint64(s) for s in (11, 27, 30, 31))


def validate_ratios(ratios: Mapping[str, float]) -> None:
    if set(ratios) != set(FOLDS):
        msg = f"split ratios keys must be exactly train/val/test, got {sorted(ratios)}"
        raise ValueError(msg)
    if any(v < 0 for v in ratios.values()):
        msg = f"split ratios must be non-negative, got {dict(ratios)}"
        raise ValueError(msg)
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, abs_tol=_TOL):
        msg = f"split ratios must sum to 1.0, got {total:.6f} from {dict(ratios)}"
        raise ValueError(msg)


def unit_interval(seed: int, group: object) -> float:
    """The group's fixed point in [0, 1): sha256 of ``"<seed>:<group>"``, first 8 bytes."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def non_null_predicate(schema: Mapping[str, pl.DataType], columns: Sequence[str]) -> list[pl.Expr]:
    """``[m9] row_filter_non_null`` as predicates: every listed column present (not null,
    not NaN). A listed column absent from ``schema`` raises ``ValueError``."""
    missing = [c for c in columns if c not in schema]
    if missing:
        msg = f"m9.row_filter_non_null {missing} not in the training set columns {list(schema)}"
        raise ValueError(msg)
    return [
        pl.col(c).is_not_null() & pl.col(c).is_not_nan()
        if schema[c].is_float()
        else pl.col(c).is_not_null()
        for c in columns
    ]


# ---------------------------------------------------------------- the keyed hash (A4)


def splitmix64(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """splitmix64's finaliser, vectorised; uint64 array arithmetic wraps modulo 2**64."""
    z = np.asarray(x, dtype=np.uint64) + _GAMMA
    z = (z ^ (z >> _S30)) * _MUL1
    z = (z ^ (z >> _S27)) * _MUL2
    return z ^ (z >> _S31)


def _key_bits(s: pl.Series) -> NDArray[np.uint64]:
    """A key column as 64-bit words: integers two's complement, floats their IEEE-754
    binary64 bits (``-0.0`` as ``0.0``), strings / categoricals the first 8 bytes of
    their UTF-8 sha256 (big-endian). Nulls, NaN and other dtypes are refused: a row
    key that cannot tell rows apart would fix membership by accident."""
    name, dt = s.name, s.dtype
    if s.null_count():
        msg = f"row key {name!r} has {s.null_count()} null value(s): it cannot identify a row"
        raise ValueError(msg)
    if dt.is_integer():
        if dt == pl.UInt64:
            return s.to_numpy().astype(np.uint64)
        return np.ascontiguousarray(s.cast(pl.Int64).to_numpy()).view(np.uint64)
    if dt.is_float():
        a = s.cast(pl.Float64).to_numpy()
        if np.isnan(a).any():
            msg = f"row key {name!r} has NaN: it cannot identify a row"
            raise ValueError(msg)
        return (a + 0.0).view(np.uint64)
    if isinstance(dt, (pl.String, pl.Categorical, pl.Enum)):
        text = s.cast(pl.String)
        words = {
            v: int.from_bytes(hashlib.sha256(v.encode()).digest()[:8], "big")
            for v in text.unique().to_list()
        }
        return text.replace_strict(words, return_dtype=pl.UInt64).to_numpy().astype(np.uint64)
    msg = f"row key {name!r} has dtype {dt}, which the keyed hash does not encode"
    raise ValueError(msg)


def row_random_unit(
    frame: pl.DataFrame, key_columns: Sequence[str], seed: int
) -> NDArray[np.float64]:
    """Each row's point in [0, 1): ``h = mix(seed)``, then ``h = mix(h ^ key)`` per key
    column in order, and the top 53 bits of ``h`` over ``2**53`` (exact in binary64)."""
    if not key_columns:
        msg = "row_random_unit needs at least one key column (m9.row_key_columns)"
        raise ValueError(msg)
    missing = [c for c in key_columns if c not in frame.columns]
    if missing:
        msg = f"m9.row_key_columns {missing} not in the training set columns {frame.columns}"
        raise ValueError(msg)
    start = splitmix64(np.array([seed], dtype=np.uint64))[0]
    h = np.full(frame.height, start, dtype=np.uint64)
    for c in key_columns:
        h = splitmix64(h ^ _key_bits(frame[c]))
    return (h >> _S11).astype(np.float64) * 2.0**-53


# ---------------------------------------------------------------- batches


class Batches(Protocol):
    """A source read one batch at a time, the row filter already applied."""

    @property
    def n_batches(self) -> int: ...

    def columns(self) -> list[str]: ...

    def frame(self, index: int, columns: Sequence[str]) -> pl.DataFrame: ...


@dataclass(frozen=True)
class FrameBatches:
    """In-memory frames as batches (the in-memory path: one frame; tests: several)."""

    frames: list[pl.DataFrame]

    @property
    def n_batches(self) -> int:
        return len(self.frames)

    def columns(self) -> list[str]:
        return self.frames[0].columns if self.frames else []

    def frame(self, index: int, columns: Sequence[str]) -> pl.DataFrame:
        return self.frames[index].select(list(columns))


class SourceBatches:
    """An M9 subtree's day files (``<set>/<target>/<day>.parquet``), one batch per file.

    Each read projects the requested columns and applies ``[m9] row_filter_non_null``
    in the scan. Every file is scanned with the first file's schema, so a file that one
    scan over the directory would refuse (a missing column, another dtype) is refused
    here too (the per-file rule of PR-053). The oracle quarantine runs first, as for
    every training-set read (PR-040).
    """

    def __init__(
        self, path: Path, *, oracle: OracleQuarantineConfig | None, row_filter: Sequence[str]
    ) -> None:
        check_oracle_quarantine(path, oracle)
        self.path = Path(path)
        self.files = iter_parquet_files(self.path)
        if not self.files:
            msg = f"no parquet file under {self.path}"
            raise ValueError(msg)
        self._schema = pl.scan_parquet(self.files[0]).collect_schema()
        self._row_filter = list(row_filter)

    @property
    def n_batches(self) -> int:
        return len(self.files)

    def columns(self) -> list[str]:
        return self._schema.names()

    def frame(self, index: int, columns: Sequence[str]) -> pl.DataFrame:
        lf = pl.scan_parquet(self.files[index], schema=self._schema)
        if self._row_filter:
            lf = lf.filter(non_null_predicate(self._schema, self._row_filter))
        return lf.select(list(columns)).collect()

    def rows_before_filter(self) -> int:
        return sum(
            int(pl.scan_parquet(f, schema=self._schema).select(pl.len()).collect().item())
            for f in self.files
        )


def _dedupe(columns: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(columns))


# ---------------------------------------------------------------- the plan


@dataclass
class PartitionPlan:
    """The rule's inputs, plus the per-batch tally of ``(fold, stamp, group) → rows``."""

    kind: str
    ratios: dict[str, float]
    time_column: str | None
    group_column: str | None
    embargo: int | None = None
    val_first_stamp: int | None = None
    test_first_stamp: int | None = None
    members: dict[Any, int] | None = None
    holdout_seed: int | None = None
    key_columns: tuple[str, ...] = ()
    seed: int | None = None
    prefix_frac: float | None = None
    prefix_last_stamp: int | None = None
    tally: pl.DataFrame = field(default_factory=pl.DataFrame)
    batch_rows: list[tuple[int, int, int]] = field(default_factory=list)
    filtered_rows: int = 0

    @property
    def tally_keys(self) -> list[str]:
        return _dedupe([c for c in (self.time_column, self.group_column) if c is not None])

    @property
    def columns(self) -> list[str]:
        """The columns :meth:`assign` and the tally read."""
        return _dedupe([*self.tally_keys, *self.key_columns])

    def assign(self, frame: pl.DataFrame) -> NDArray[np.int8]:
        """Every row's fold: ``TRAIN`` / ``VAL`` / ``TEST``, or ``PURGED``."""
        if self.kind == "time_ordered":
            fold = self._time_folds(frame)
        elif self.kind == "symbol_holdout":
            assert self.group_column is not None and self.members is not None
            fold = np.array(
                frame[self.group_column]
                .replace_strict(self.members, return_dtype=pl.Int8)
                .to_numpy(),
                dtype=np.int8,
            )
        else:
            assert self.seed is not None
            u = row_random_unit(frame, self.key_columns, self.seed)
            cut_train = self.ratios["train"]
            cut_val = self.ratios["train"] + self.ratios["val"]
            fold = np.where(u < cut_train, TRAIN, np.where(u < cut_val, VAL, TEST)).astype(np.int8)
        if self.prefix_last_stamp is not None:
            assert self.time_column is not None
            s = frame[self.time_column].to_numpy()
            fold[(fold == TRAIN) & (s > self.prefix_last_stamp)] = PURGED
        return fold

    def _time_folds(self, frame: pl.DataFrame) -> NDArray[np.int8]:
        assert self.time_column is not None and self.embargo is not None
        s = frame[self.time_column].to_numpy()
        emb, t_val, t_test = self.embargo, self.val_first_stamp, self.test_first_stamp
        fold = np.full(frame.height, PURGED, dtype=np.int8)
        fold[np.ones(frame.height, dtype=bool) if t_val is None else s + emb < t_val] = TRAIN
        if t_val is not None:
            fold[(s >= t_val) if t_test is None else (s >= t_val) & (s + emb < t_test)] = VAL
        if t_test is not None:
            fold[s >= t_test] = TEST
        return fold

    def rows(self, fold: int) -> int:
        return int(self.tally.filter(pl.col("fold") == fold)[ROWS].sum())

    def parts(self) -> dict[str, pl.DataFrame]:
        """The tally per partition: ``(stamp, group, rows)``, for the leakage audit."""
        return {k: self.tally.filter(pl.col("fold") == i).drop("fold") for i, k in enumerate(FOLDS)}


def _tally(fold: NDArray[np.int8], frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    by = pl.DataFrame({"fold": pl.Series(fold, dtype=pl.Int8)})
    if keys:
        by = pl.concat([by, frame.select(keys)], how="horizontal")
    return by.group_by(["fold", *keys]).agg(pl.len().cast(pl.Int64).alias(ROWS))


def _fold_rows(tally: pl.DataFrame) -> tuple[int, int, int]:
    def rows(fold: int) -> int:
        return int(tally.filter(pl.col("fold") == fold)[ROWS].sum())

    return rows(TRAIN), rows(VAL), rows(TEST)


def plan_partitions(cfg: RuxMLConfig, batches: Batches, *, seed: int | None) -> PartitionPlan:
    """The ``[m9]`` partition rule for ``cfg.data.split_kind`` over ``batches``
    (row filter applied), with its per-batch tally. ``seed`` keys the row-random hash."""
    data = cfg.data
    ratios = dict(data.split_ratios)
    validate_ratios(ratios)
    names = batches.columns()
    plan = PartitionPlan(
        kind=data.split_kind,
        ratios=ratios,
        time_column=data.time_column,
        group_column=data.group_column,
        prefix_frac=data.train_prefix_frac,
    )
    train_n = 0
    if plan.kind == "time_ordered":
        train_n = _plan_time(plan, batches, names, data.split_embargo)
    elif plan.kind == "symbol_holdout":
        _plan_groups(plan, batches, names, data.symbol_holdout_seed)
    else:
        keys = cfg.m9.row_key_columns if cfg.m9 is not None else []
        if not keys:
            msg = "an [m9] row-random split needs m9.row_key_columns (program PR-027 A4)"
            raise ValueError(msg)
        if seed is None:
            msg = "an [m9] row-random split needs the split seed (it keys the row hash)"
            raise ValueError(msg)
        plan.key_columns, plan.seed = tuple(keys), seed

    keys = plan.tally_keys
    tallies = [_tally(plan.assign(f), f, keys) for f in _frames(batches, plan.columns)]
    if plan.kind == "time_ordered" and train_n and not sum(_fold_rows(t)[TRAIN] for t in tallies):
        t = plan.time_column
        msg = f"split embargo {plan.embargo} purged every train row (time_column {t!r})"
        raise ValueError(msg)
    if plan.prefix_frac is not None:
        tallies = _cut_prefix(plan, tallies)
    plan.batch_rows = [_fold_rows(t) for t in tallies]
    plan.tally = (
        pl.concat(tallies).group_by(["fold", *keys]).agg(pl.col(ROWS).sum())
        if tallies
        else pl.DataFrame(schema={"fold": pl.Int8, ROWS: pl.Int64})
    )
    plan.filtered_rows = int(plan.tally[ROWS].sum()) if plan.tally.height else 0
    return plan


def _frames(batches: Batches, columns: list[str]) -> list[pl.DataFrame]:
    return [batches.frame(i, columns) for i in range(batches.n_batches)]


def _plan_time(plan: PartitionPlan, batches: Batches, names: list[str], embargo: int | None) -> int:
    t = plan.time_column
    assert t is not None  # RuxMLConfig: time_ordered requires data.time_column
    if t not in names:
        msg = (
            f"temporal_train_val_test_split: time_column={t!r} not found in "
            f"DataFrame columns: {names}"
        )
        raise ValueError(msg)
    if embargo is None:
        msg = "an [m9] time-ordered split needs data.split_embargo (>= m9.h_max_ms)"
        raise ValueError(msg)
    parts = [batches.frame(i, [t]).group_by(t).len() for i in range(batches.n_batches)]
    counts = pl.concat(parts).group_by(t).agg(pl.col("len").sum()).sort(t)
    if not counts.schema[t].is_integer():
        msg = (
            f"split embargo needs an integer time_column; {t!r} is {counts.schema[t]} "
            "(the embargo is in the column's own units)"
        )
        raise ValueError(msg)
    cum = counts["len"].cum_sum().to_numpy()
    stamps = counts[t].to_numpy()
    n = int(cum[-1]) if cum.size else 0
    train_n = int(n * plan.ratios["train"])
    val_n = int(n * plan.ratios["val"])

    def stamp_at(position: int) -> int | None:
        # the stamp at sorted row position ``position``: the first whose running count exceeds it
        return int(stamps[np.searchsorted(cum, position, side="right")]) if position < n else None

    plan.embargo = embargo
    plan.val_first_stamp = stamp_at(train_n)
    plan.test_first_stamp = stamp_at(train_n + val_n)
    return train_n


def _plan_groups(
    plan: PartitionPlan, batches: Batches, names: list[str], holdout_seed: int | None
) -> None:
    g = plan.group_column
    assert g is not None and holdout_seed is not None  # RuxMLConfig validator
    if g not in names:
        msg = f"symbol_holdout_split: group_column={g!r} not found in DataFrame columns: {names}"
        raise ValueError(msg)
    levels: set[Any] = set()
    nulls = rows = 0
    for i in range(batches.n_batches):
        col = batches.frame(i, [g])[g]
        nulls += col.null_count()
        rows += col.len()
        levels.update(col.drop_nulls().unique().to_list())
    if nulls:
        msg = f"symbol_holdout_split: {nulls} null value(s) in group_column {g!r}"
        raise ValueError(msg)
    cut_train = plan.ratios["train"]
    cut_val = plan.ratios["train"] + plan.ratios["val"]
    members: dict[Any, int] = {}
    for group in sorted(levels):
        u = unit_interval(holdout_seed, group)
        members[group] = TRAIN if u < cut_train else VAL if u < cut_val else TEST
    if rows:
        empty = [k for i, k in enumerate(FOLDS) if plan.ratios[k] > 0 and i not in members.values()]
        if empty:
            msg = (
                f"symbol_holdout_split: no group assigned to {empty} "
                f"({len(levels)} groups, ratios {plan.ratios}, seed {holdout_seed})"
            )
            raise ValueError(msg)
    plan.members, plan.holdout_seed = members, holdout_seed


def _cut_prefix(plan: PartitionPlan, tallies: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """The nested-prefix cut on train (PR-046): the first ``ceil(frac · n)`` unique stamps."""
    t, frac = plan.time_column, plan.prefix_frac
    assert t is not None and frac is not None  # RuxMLConfig: a prefix requires data.time_column
    if not 0.0 < frac <= 1.0:
        msg = f"train_prefix: frac must be in (0, 1], got {frac}"
        raise ValueError(msg)
    train = [x.filter(pl.col("fold") == TRAIN)[t] for x in tallies]
    stamps = pl.concat(train).unique().sort() if train else pl.Series(t, [], dtype=pl.Int64)
    if stamps.len() == 0 or frac == 1.0:
        return tallies
    last = int(stamps[math.ceil(frac * stamps.len()) - 1])
    plan.prefix_last_stamp = last
    cut = (
        pl.when((pl.col("fold") == TRAIN) & (pl.col(t) > last))
        .then(PURGED)
        .otherwise(pl.col("fold"))
    )
    keys = plan.tally_keys
    return [
        x.with_columns(cut.cast(pl.Int8).alias("fold"))
        .group_by(["fold", *keys])
        .agg(pl.col(ROWS).sum())
        for x in tallies
    ]


# ---------------------------------------------------------------- records from the tally


def split_record(
    cfg: RuxMLConfig, plan: PartitionPlan, *, rows_before_filter: int
) -> dict[str, object]:
    """The split as a record (PR-042's ``split_definition`` keys) plus the membership rule:
    the stamp cuts (time-block), the keyed hash (row-random), the prefix's last stamp."""
    rows = {k: plan.rows(i) for i, k in enumerate(FOLDS)}
    filtered: dict[str, object] = {}
    if cfg.m9 is not None and cfg.m9.row_filter_non_null:
        filtered = {
            "row_filter_non_null": cfg.m9.row_filter_non_null,
            "row_filter_dropped": rows_before_filter - plan.filtered_rows,
        }
    out: dict[str, object]
    if plan.kind == "symbol_holdout":
        g = plan.group_column
        assert g is not None
        out = {
            "kind": plan.kind,
            "group_column": g,
            "symbol_holdout_seed": plan.holdout_seed,
            "groups": {
                k: plan.tally.filter((pl.col("fold") == i) & (pl.col(ROWS) > 0))[g]
                .unique()
                .sort()
                .to_list()
                for i, k in enumerate(FOLDS)
            },
            "rows": rows,
            **filtered,
        }
    elif plan.kind == "time_ordered":
        out = {
            "kind": plan.kind,
            "time_column": plan.time_column,
            "split_embargo": plan.embargo,
            "rows": rows,
            "purged_rows": plan.filtered_rows - sum(rows.values()),
            "cuts": {
                "val_first_stamp": plan.val_first_stamp,
                "test_first_stamp": plan.test_first_stamp,
                "boundary_ties": "later_partition",
            },
            **filtered,
        }
    else:
        out = {
            "kind": plan.kind,
            "rows": rows,
            "membership": {
                "rule": "keyed_hash",
                "hash": "splitmix64",
                "key_columns": list(plan.key_columns),
                "seed": plan.seed,
            },
            **filtered,
        }
    if plan.prefix_frac is not None:
        out["train_prefix_frac"] = plan.prefix_frac
        out["train_prefix_last_stamp"] = plan.prefix_last_stamp
    return out


@dataclass
class _Moments:
    """Count, mean, M2, min, max of a column's values, merged batch by batch (Chan et al.)."""

    n: int = 0
    nulls: int = 0
    mean: float = 0.0
    m2: float = 0.0
    lo: float = math.inf
    hi: float = -math.inf

    def add(self, a: NDArray[np.float64]) -> None:
        valid = a[~np.isnan(a)]
        self.nulls += int(a.size - valid.size)
        nb = int(valid.size)
        if not nb:
            return
        mb = float(valid.mean())
        m2b = float(np.square(valid - mb).sum())
        n = self.n + nb
        delta = mb - self.mean
        self.mean = mb if not self.n else self.mean + delta * nb / n
        self.m2 = m2b if not self.n else self.m2 + m2b + delta * delta * self.n * nb / n
        self.n = n
        self.lo = min(self.lo, float(valid.min()))
        self.hi = max(self.hi, float(valid.max()))

    def summary(self) -> dict[str, float | None]:
        some = self.n > 0
        return {
            "n": float(self.n),
            "null_count": float(self.nulls),
            "mean": self.mean if some else None,
            "std": math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else None,
            "min": self.lo if some else None,
            "max": self.hi if some else None,
        }


def diagnostics_by_batches(
    batches: Batches, plan: PartitionPlan, columns: Sequence[str]
) -> dict[str, dict[str, dict[str, float | None]]]:
    """``[m9] diagnostic_columns`` summarised per fold (train, val) in their own pass (A3):
    ``{n, null_count, mean, std, min, max}`` as ``cli.train._label_diagnostics``, with
    the moments merged batch by batch. A listed column absent from the set is refused."""
    names = batches.columns()
    missing = [c for c in columns if c not in names]
    if missing:
        msg = f"m9.diagnostic_columns {missing} not in the training set columns {names}"
        raise ValueError(msg)
    acc = {fold: {c: _Moments() for c in columns} for fold in FOLDS[:2]}
    if columns:
        for i in range(batches.n_batches):
            if not any(plan.batch_rows[i][:2]):
                continue
            frame = batches.frame(i, _dedupe([*plan.columns, *columns]))
            fold = plan.assign(frame)
            for f, name in enumerate(FOLDS[:2]):
                part = frame.filter(pl.Series(fold == f))
                for c in columns:
                    acc[name][c].add(part[c].cast(pl.Float64).to_numpy())
    return {name: {c: m.summary() for c, m in cols.items()} for name, cols in acc.items()}

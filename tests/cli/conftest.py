"""Fixtures for CLI tests."""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from tests.conftest import m9_column_dtypes, m9_subtree


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------- PR-043 / PR-044 / PR-048: a synthetic set in the harness's M9 layout ----------

REPO = Path(__file__).resolve().parents[2]
DAY_MS = 86_400_000
GRID_MS = 900_000  # a 15-minute label grid, as program D45 #3's label_grid_ms
# 12 coins: under the M9 configs' symbol_holdout_seed every partition gets one.
COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "XRP", "ADA", "SUI", "HYPE", "BNB", "LTC"]
# The schema's outcome codes (``outcomes``): ok, alo_expired, no_book, below_one_lot.
OK, ALO_EXPIRED, NO_BOOK, BELOW_ONE_LOT = 0, 1, 3, 4
_POLARS_DTYPES = {
    "int64": pl.Int64, "int8": pl.Int8, "float64": pl.Float64, "string": pl.String,
    "bool": pl.Boolean,
}  # fmt: skip


def _c6_day(subtree: str, day: int, rng: np.random.Generator) -> pl.DataFrame:
    """One day of ``<set>/<subtree>/<day>.parquet`` with exactly the schema's columns and
    dtypes (tests/fixtures/m9_training_schema.json). Faithful to the materializer's
    writing rules: rows whose order was never placed (``no_book``, ``below_one_lot``) or
    rested nowhere (``alo_expired``) are WRITTEN with NaN targets; the markout is null on
    an unfilled rung (A9); ``y__walk_bp`` is null where an IOC filled nothing (≈ 2 %)."""
    entry = m9_subtree(subtree)
    stamps = [day * DAY_MS + i * GRID_MS for i in range(DAY_MS // GRID_MS)]
    rows = [(t, c, s) for t in stamps for c in COINS for s in (1, -1)]
    n = len(rows)
    q = rng.choice([100.0, 300.0, 1000.0, 3000.0], size=n)
    spread = rng.gamma(2.0, 1.0, size=n)
    cols: dict[str, object] = {
        "stamp_ms": [r[0] for r in rows],
        "coin": [r[1] for r in rows],
        "side": [r[2] for r in rows],
        "kind": [entry["kind"]] * n,
        "p_bp": rng.choice([0.0, 15.0, 30.0], size=n) if subtree == "fill" else [entry["p_bp"]] * n,
        "q_usd": q,
        "h_ms": [entry["h_ms"]] * n,
    }
    for f in entry["feat"]:
        cols[f["name"]] = rng.normal(0.0, 1.0, size=n)
    cols["feat__spread_bp"] = spread
    nan = float("nan")
    if subtree == "fill":
        outcome = rng.choice(
            [OK, ALO_EXPIRED, NO_BOOK, BELOW_ONE_LOT], p=[0.85, 0.05, 0.05, 0.05], size=n
        )
        ok = outcome == OK
        fill = np.clip(rng.beta(2.0, 2.0, size=n) - 0.2, 0.0, 1.0)
        filled = ok & (fill > 0.0)
        markout = rng.normal(0.0, 3.0, size=n)
        for c in entry["y"]:
            name = c["name"]
            if name.startswith("y__fill_frac"):
                h = c["horizon_ms"] / (entry["h_ms"] + 1_000)  # the curve rises to the label
                cols[name] = np.where(ok, fill * min(h, 1.0), nan)
            elif name.startswith("y__markout_bp"):
                cols[name] = [float(m) if f else None for f, m in zip(filled, markout, strict=True)]
        cols["y__capture_bp"] = np.where(filled, rng.normal(1.0, 0.5, size=n), nan)
        cols["y__ttf_first_ms"] = np.where(filled, rng.integers(1_000, entry["h_ms"], size=n), -1)
        cols["y__ttf_full_ms"] = np.where(filled & (fill >= 0.8), cols["y__ttf_first_ms"], -1)
        cols["y__queue_ahead_at_ack"] = np.where(ok, rng.gamma(2.0, 100.0, size=n), nan)
        cols["y__through"] = filled & (fill >= 0.8)
    else:
        outcome = rng.choice([OK, NO_BOOK, BELOW_ONE_LOT], p=[0.9, 0.05, 0.05], size=n)
        ok = outcome == OK
        nothing = ok & (rng.random(size=n) < 0.02)  # the stale-touch IOCs that fill nothing
        walk = 0.5 * spread + 0.001 * q + rng.normal(0.0, 0.2, size=n)
        cols["y__walk_bp"] = [
            None if z else (float(w) if o else nan)
            for o, z, w in zip(ok, nothing, walk, strict=True)
        ]
        cols["y__walk_filled_frac"] = np.where(ok, np.where(nothing, 0.0, 1.0), nan)
        cols["y__half_spread_bp"] = np.where(ok, 0.5 * spread, nan)
        cols["y__censored_beyond_book"] = (ok & (q >= 3000.0)).astype(np.int8)
        cols["y__levels_walked"] = np.where(ok & ~nothing, rng.integers(1, 5, size=n), 0)
    cols["y__outcome"] = outcome.astype(np.int8)
    dtypes = m9_column_dtypes(subtree)
    return pl.DataFrame(
        {
            c: pl.Series(c, cols[c], dtype=_POLARS_DTYPES[t], nan_to_null=False)
            for c, t in dtypes.items()
        }
    )


@pytest.fixture
def c6_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The set root: ``fill/<day>.parquet`` and ``walk/<day>.parquet``, flat (no hive key)."""
    monkeypatch.chdir(REPO)  # --problem resolves configs/problems against the cwd
    rng = np.random.default_rng(0)
    root = tmp_path / "training_root" / "set"
    for subtree in ("fill", "walk"):
        (root / subtree).mkdir(parents=True)
        for day in range(20_500, 20_503):
            iso = (dt.date(1970, 1, 1) + dt.timedelta(days=day)).isoformat()
            _c6_day(subtree, day, rng).write_parquet(root / subtree / f"{iso}.parquet")
    return root


def problem_subtree(problem: str) -> str:
    """The subtree a repo M9 problem reads (the last part of its ``data.source_path``)."""
    toml = tomllib.loads((REPO / "configs" / "problems" / f"{problem}.toml").read_text())
    return Path(toml["data"]["source_path"]).name


def m9_argv(tmp: Path, source: Path, problem: str, *extra: str) -> list[str]:
    """The CLI argv of a repo M9 problem on the set rooted at ``source`` (its own subtree)."""
    sets = {
        "data.source_path": str(source / problem_subtree(problem)),
        "data.cas_root": str(tmp / "cas"),
        "data.manifests_root": str(tmp / "manifests"),
        "training.device": "cpu",
        "runs.storage_url": f"sqlite:///{tmp}/studies/studies.db",
        "runs.artifacts_root": str(tmp / "studies" / "artifacts"),
        "registry.root": str(tmp / "registry"),
    }
    argv = ["--config", str(REPO / "configs" / "base.toml"), "--problem", problem]
    for k, v in sets.items():
        argv += ["--set", f"{k}={v}"]
    return [*argv, *extra]

"""Fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------- PR-043 / PR-044: a synthetic set with the program's C6 layout ----------

REPO = Path(__file__).resolve().parents[2]
DAY_MS = 86_400_000
GRID_MS = 900_000  # a 15-minute label grid, as program D45 #3's label_grid_ms


def _c6_day(day: int, rng: np.random.Generator) -> pl.DataFrame:
    stamps = [day * DAY_MS + i * GRID_MS for i in range(DAY_MS // GRID_MS)]
    coins = ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK"]
    rows = [(t, c, s) for t in stamps for c in coins for s in ("buy", "sell")]
    n = len(rows)
    q = rng.choice([100.0, 300.0, 1000.0, 3000.0], size=n)
    spread = rng.gamma(2.0, 1.0, size=n)
    fill = np.clip(rng.beta(2.0, 2.0, size=n) - 0.2, 0.0, 1.0)
    markout = rng.normal(0.0, 3.0, size=n)
    return pl.DataFrame(
        {
            "stamp_ms": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "coin": [r[1] for r in rows],
            "side": [r[2] for r in rows],
            "kind": rng.choice(["alo", "taker"], size=n).tolist(),
            "p_bp": rng.choice([0.0, 5.0, 10.0], size=n),
            "q_usd": q,
            "h_ms": rng.choice([60_000, 600_000], size=n).astype(np.int64),
            "feat__spread_bp": spread,
            "y__fill_frac": fill,
            # undefined on an unfilled rung (program D37 #2): null
            "y__markout_bp": [
                None if f == 0.0 else float(m) for f, m in zip(fill, markout, strict=True)
            ],
            "y__walk_bp": 0.5 * spread + 0.001 * q + rng.normal(0.0, 0.2, size=n),
        }
    )


@pytest.fixture
def c6_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(REPO)  # --problem resolves configs/problems against the cwd
    rng = np.random.default_rng(0)
    root = tmp_path / "training_root" / "set"
    root.mkdir(parents=True)
    for day in range(3):
        _c6_day(20_500 + day, rng).write_parquet(root / f"day={day}.parquet")
    return root


def m9_argv(tmp: Path, source: Path, problem: str, *extra: str) -> list[str]:
    sets = {
        "data.source_path": str(source),
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

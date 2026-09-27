"""End-to-end: a ``y__``-as-feature set and an ``oracle__`` set are refused with exit 2 (PR-043).

rux-capital program v0.3 ACCEPTANCE C6: "rux-ml refuses an ``oracle__`` set and a
``y__``-as-feature set on real files (exit 2, recorded)"; D38 #3: "a ``y__`` column is
a supervised target consumed only as ``target_column``". The real set is program
PR-024's (not landed); these tests drive the repo's real M9 problem configs through
the real CLI on a synthetic set with the C6 layout — a day-partitioned directory,
keys ``stamp_ms, coin, side, kind, p_bp, q_usd, h_ms``, ``feat__`` inputs and three
``y__`` labels. The recorded real-file run is the program's evidence (C6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml.cli import app
from tests.conftest import repo_oracle_values

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


def _argv(tmp: Path, source: Path, *extra: str) -> list[str]:
    sets = {
        "data.source_path": str(source),
        "data.cas_root": str(tmp / "cas"),
        "data.manifests_root": str(tmp / "manifests"),
        "training.device": "cpu",
        "runs.storage_url": f"sqlite:///{tmp}/studies/studies.db",
        "runs.artifacts_root": str(tmp / "studies" / "artifacts"),
        "registry.root": str(tmp / "registry"),
    }
    argv = ["--config", str(REPO / "configs" / "base.toml"), "--problem", "m9_walk_bp"]
    for k, v in sets.items():
        argv += ["--set", f"{k}={v}"]
    return [*argv, *extra]


def _n_trials(tmp: Path) -> int:
    db = tmp / "studies" / "studies.db"
    if not db.exists():
        return 0
    storage = f"sqlite:///{db}"
    return sum(s.n_trials for s in optuna.get_all_study_summaries(storage=storage))


def test_the_clean_c6_set_trains_under_the_real_m9_problem(
    runner: CliRunner, c6_set: Path, tmp_path: Path
) -> None:
    result = runner.invoke(app, _argv(tmp_path, c6_set, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "score (mae):" in result.output
    assert _n_trials(tmp_path) == 1


@pytest.mark.parametrize("label", ["y__time_to_fill_ms", "y__fill_frac", "Y__WALK_BP_ALIAS"])
def test_a_label_among_the_features_exits_2_before_any_trial(
    runner: CliRunner, c6_set: Path, tmp_path: Path, label: str
) -> None:
    """Any label-namespace column (case-insensitive), a listed diagnostic included."""
    for f in c6_set.glob("*.parquet"):  # make the column exist, so only the rule refuses
        df = pl.read_parquet(f)
        if label not in df.columns:
            df.with_columns(pl.col("y__walk_bp").alias(label)).write_parquet(f)
    features = json.dumps(["p_bp", "q_usd", "h_ms", "feat__spread_bp", label])
    argv = _argv(tmp_path, c6_set, "--set", f"features.spec.numeric_columns={features}", "train")
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "label quarantine" in result.output
    assert _n_trials(tmp_path) == 0


def test_tune_start_refuses_a_label_feature_before_any_child(
    runner: CliRunner, c6_set: Path, tmp_path: Path
) -> None:
    features = json.dumps(["p_bp", "q_usd", "h_ms", "y__fill_frac"])
    argv = _argv(tmp_path, c6_set)
    argv = [*argv, "--set", f"features.spec.numeric_columns={features}", "tune", "start"]
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "label quarantine" in result.output
    assert _n_trials(tmp_path) == 0


def test_an_oracle_column_in_the_c6_set_exits_2_through_pr040(
    runner: CliRunner, c6_set: Path, tmp_path: Path
) -> None:
    """The oracle refusal on the C6 layout is PR-040's check, reached unchanged."""
    ns = repo_oracle_values()["namespace"]
    f = sorted(c6_set.glob("*.parquet"))[1]
    df = pl.read_parquet(f)
    df.with_columns(pl.col("y__walk_bp").alias(f"{ns}walk_bp")).write_parquet(f)
    result = runner.invoke(app, _argv(tmp_path, c6_set, "train"))
    assert result.exit_code == 2, result.output
    assert "oracle quarantine" in result.output
    assert _n_trials(tmp_path) == 0


def test_registry_promote_refit_refuses_a_label_feature(
    runner: CliRunner, c6_set: Path, tmp_path: Path
) -> None:
    """The promotion path (C13): a clean trial, re-fit under a config that lists a
    label as a feature, is refused with exit 2 and writes no bundle."""
    trained = runner.invoke(app, _argv(tmp_path, c6_set, "train"), catch_exceptions=False)
    assert trained.exit_code == 0, trained.output
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    features = json.dumps(["p_bp", "q_usd", "h_ms", "y__markout_bp"])
    argv = [
        *_argv(tmp_path, c6_set),
        "--set", f"features.spec.numeric_columns={features}",
        "registry", "promote", "--problem", "m9_walk_bp",
        "--study", summary.study_name, "--trial", "0",
    ]  # fmt: skip
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "label quarantine" in result.output
    assert not (tmp_path / "registry" / "m9_walk_bp").exists()

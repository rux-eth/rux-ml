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

import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml.cli import app
from tests.cli.conftest import m9_argv
from tests.conftest import m9_gates_overrides, repo_oracle_values

REPO = Path(__file__).resolve().parents[2]


def _argv(tmp: Path, source: Path, *extra: str, gates: dict[str, str]) -> list[str]:
    """The real ``m9_walk_bp`` problem, its ``[m9.gates]`` pointed at a signed fixture."""
    return m9_argv(tmp, source, "m9_walk_bp", *m9_gates_overrides(gates), *extra)


def _n_trials(tmp: Path) -> int:
    db = tmp / "studies" / "studies.db"
    if not db.exists():
        return 0
    storage = f"sqlite:///{db}"
    return sum(s.n_trials for s in optuna.get_all_study_summaries(storage=storage))


def test_the_clean_c6_set_trains_under_the_real_m9_problem(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    result = runner.invoke(
        app, _argv(tmp_path, c6_set, "train", gates=signed_m9_gates), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    assert "score (mae):" in result.output
    assert _n_trials(tmp_path) == 1


@pytest.mark.parametrize("label", ["y__time_to_fill_ms", "y__fill_frac", "Y__WALK_BP_ALIAS"])
def test_a_label_among_the_features_exits_2_before_any_trial(
    runner: CliRunner,
    c6_set: Path,
    tmp_path: Path,
    label: str,
    signed_m9_gates: dict[str, str],
) -> None:
    """Any label-namespace column (case-insensitive), a listed diagnostic included."""
    for f in c6_set.glob("*.parquet"):  # make the column exist, so only the rule refuses
        df = pl.read_parquet(f)
        if label not in df.columns:
            df.with_columns(pl.col("y__walk_bp").alias(label)).write_parquet(f)
    features = json.dumps(["p_bp", "q_usd", "h_ms", "feat__spread_bp", label])
    argv = _argv(
        tmp_path, c6_set, "--set", f"features.spec.numeric_columns={features}", "train",
        gates=signed_m9_gates,
    )  # fmt: skip
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "label quarantine" in result.output
    assert _n_trials(tmp_path) == 0


def test_tune_start_refuses_a_label_feature_before_any_child(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    features = json.dumps(["p_bp", "q_usd", "h_ms", "y__fill_frac"])
    argv = _argv(tmp_path, c6_set, gates=signed_m9_gates)
    argv = [*argv, "--set", f"features.spec.numeric_columns={features}", "tune", "start"]
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "label quarantine" in result.output
    assert _n_trials(tmp_path) == 0


def test_an_oracle_column_in_the_c6_set_exits_2_through_pr040(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    """The oracle refusal on the C6 layout is PR-040's check, reached unchanged."""
    ns = repo_oracle_values()["namespace"]
    f = sorted(c6_set.glob("*.parquet"))[1]
    df = pl.read_parquet(f)
    df.with_columns(pl.col("y__walk_bp").alias(f"{ns}walk_bp")).write_parquet(f)
    result = runner.invoke(app, _argv(tmp_path, c6_set, "train", gates=signed_m9_gates))
    assert result.exit_code == 2, result.output
    assert "oracle quarantine" in result.output
    assert _n_trials(tmp_path) == 0


def test_registry_promote_refit_refuses_a_label_feature(
    runner: CliRunner, c6_set: Path, tmp_path: Path, signed_m9_gates: dict[str, str]
) -> None:
    """The promotion path (C13): a clean trial, re-fit under a config that lists a
    label as a feature, is refused with exit 2 and writes no bundle."""
    trained = runner.invoke(
        app, _argv(tmp_path, c6_set, "train", gates=signed_m9_gates), catch_exceptions=False
    )
    assert trained.exit_code == 0, trained.output
    storage = f"sqlite:///{tmp_path}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    features = json.dumps(["p_bp", "q_usd", "h_ms", "y__markout_bp"])
    argv = [
        *_argv(tmp_path, c6_set, gates=signed_m9_gates),
        "--set", f"features.spec.numeric_columns={features}",
        "registry", "promote", "--problem", "m9_walk_bp",
        "--study", summary.study_name, "--trial", "0",
    ]  # fmt: skip
    result = runner.invoke(app, argv)
    assert result.exit_code == 2, result.output
    assert "label quarantine" in result.output
    assert not (tmp_path / "registry" / "m9_walk_bp").exists()

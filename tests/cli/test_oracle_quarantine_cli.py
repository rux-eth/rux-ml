"""End-to-end: every CLI verb that builds or hashes a training set refuses oracle inputs (PR-040).

Program ACCEPTANCE C13: "a test shows the training-set builder and the promotion
path refuse any ``oracle__`` column and any artifact tagged as produced under
oracle inputs". Each test drives the real CLI through the real layered config
loader, and asserts exit code 2 plus the named error's message — polars' own accidental
errors on some of these inputs would exit 1 with a different message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml.cli import app

REFUSED = "oracle quarantine"  # the named error's message prefix (data/quarantine.py)


def _synth() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    return pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()})


@pytest.fixture
def workdir(tmp_path: Path, oracle_toml: str, oracle_values: dict[str, str]) -> Path:
    """Clean source + two oracle sources (namespace column; tagged store) + base.toml."""
    df = _synth()
    df.write_parquet(tmp_path / "clean.parquet")
    df.with_columns(pl.col("y").alias(f"{oracle_values['namespace']}label")).write_parquet(
        tmp_path / "oracle_col.parquet"
    )
    day = tmp_path / "oracle-store" / "day"
    day.mkdir(parents=True)
    df.write_parquet(day / "tagged.parquet")
    (day / oracle_values["tag_file"]).write_text('{"oracle": true}')

    (tmp_path / "base.toml").write_text(
        f"""
[data]
source_path = "{tmp_path}/clean.parquet"
target_column = "y"
cas_root = "{tmp_path}/cas"
manifests_root = "{tmp_path}/manifests"

[features]
spec = {{ numeric_columns = ["x1", "x2"], categorical_columns = [] }}

[training]
kind = "xgboost"
device = "cpu"
metric = "auc"
n_estimators = 8
max_depth = 3
learning_rate = 0.3

[tuning]
sampler = "tpe"
pruner = "wilcoxon"
n_trials = 2
n_startup_trials = 1
entropy = 42

[cv]
kind = "kfold"
n_splits = 3
shuffle = true

[runs]
storage_url = "sqlite:///{tmp_path}/studies/studies.db"
artifacts_root = "{tmp_path}/studies/artifacts"

[registry]
root = "{tmp_path}/registry"

[search_space."training.max_depth"]
type = "int"
low = 2
high = 4
"""
        + oracle_toml
    )
    return tmp_path


def _sources(workdir: Path) -> dict[str, Path]:
    return {
        "column": workdir / "oracle_col.parquet",
        "tagged": workdir / "oracle-store" / "day" / "tagged.parquet",
    }


def _argv(workdir: Path, source: Path | None, *args: str) -> list[str]:
    head = ["--config", str(workdir / "base.toml")]
    if source is not None:
        head += ["--set", f"data.source_path={source}"]
    return [*head, *args]


def _assert_refused(result: object) -> None:
    exit_code = result.exit_code  # type: ignore[attr-defined]
    output = result.output  # type: ignore[attr-defined]
    assert exit_code == 2, output
    assert REFUSED in output, output


def _n_trials(workdir: Path) -> int:
    db = workdir / "studies" / "studies.db"
    if not db.exists():
        return 0
    storage = f"sqlite:///{db}"
    return sum(
        len(optuna.load_study(study_name=s.study_name, storage=storage).trials)
        for s in optuna.get_all_study_summaries(storage=storage)
    )


@pytest.mark.parametrize("kind", ["column", "tagged"])
def test_train_refuses_with_exit_2_and_records_no_trial(
    runner: CliRunner, workdir: Path, kind: str
) -> None:
    result = runner.invoke(app, _argv(workdir, _sources(workdir)[kind], "train"))
    _assert_refused(result)
    assert _n_trials(workdir) == 0


@pytest.mark.parametrize("kind", ["column", "tagged"])
def test_tune_start_refuses_before_dispatching_any_child(
    runner: CliRunner, workdir: Path, kind: str
) -> None:
    result = runner.invoke(
        app, _argv(workdir, _sources(workdir)[kind], "tune", "start", "s", "--n-trials", "2")
    )
    _assert_refused(result)
    assert "dispatching subprocess child" not in result.output
    assert _n_trials(workdir) == 0


@pytest.mark.parametrize("verb", ["hash", "version"])
def test_data_hash_and_version_refuse_and_write_nothing(
    runner: CliRunner, workdir: Path, verb: str
) -> None:
    src = _sources(workdir)["tagged"]
    args = ["data", "hash", str(src)] if verb == "hash" else ["data", "version", "n", str(src)]
    result = runner.invoke(app, _argv(workdir, None, *args))
    _assert_refused(result)
    assert not (workdir / "cas").exists()
    assert not (workdir / "manifests").exists()


def test_clean_train_still_succeeds(runner: CliRunner, workdir: Path) -> None:
    result = runner.invoke(app, _argv(workdir, None, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert _n_trials(workdir) == 1


def _train_clean_and_find_trial(runner: CliRunner, workdir: Path) -> tuple[str, int]:
    result = runner.invoke(app, _argv(workdir, None, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    storage = f"sqlite:///{workdir}/studies/studies.db"
    (summary,) = optuna.get_all_study_summaries(storage=storage)
    study = optuna.load_study(study_name=summary.study_name, storage=storage)
    return summary.study_name, study.trials[-1].number


@pytest.mark.parametrize("kind", ["column", "tagged"])
def test_registry_promote_refit_refuses(runner: CliRunner, workdir: Path, kind: str) -> None:
    study, trial = _train_clean_and_find_trial(runner, workdir)
    result = runner.invoke(
        app,
        _argv(
            workdir,
            _sources(workdir)[kind],
            "registry", "promote", "--problem", "p", "--study", study, "--trial", str(trial),
        ),
    )  # fmt: skip
    _assert_refused(result)
    assert not (workdir / "registry" / "p").exists()


@pytest.mark.parametrize("kind", ["column", "tagged"])
def test_registry_score_refuses(runner: CliRunner, workdir: Path, kind: str) -> None:
    study, trial = _train_clean_and_find_trial(runner, workdir)
    promoted = runner.invoke(
        app,
        _argv(
            workdir, None,
            "registry", "promote", "--problem", "p", "--study", study, "--trial", str(trial),
        ),
        catch_exceptions=False,
    )  # fmt: skip
    assert promoted.exit_code == 0, promoted.output
    result = runner.invoke(
        app,
        _argv(
            workdir, _sources(workdir)[kind],
            "registry", "score", "--problem", "p", "--output", str(workdir / "receipts"),
        ),
    )  # fmt: skip
    _assert_refused(result)
    # scorer.py creates output_dir up front (pre-PR-040); the refusal must leave it empty.
    assert list((workdir / "receipts").iterdir()) == []


def _seed_study(workdir: Path, *, failed_trial: bool) -> str:
    storage = f"sqlite:///{workdir}/studies/studies.db"
    (workdir / "studies").mkdir(exist_ok=True)
    study = optuna.create_study(study_name="s", storage=storage, direction="maximize")
    if failed_trial:
        study.add_trial(
            optuna.trial.create_trial(
                params={"training.max_depth": 3},
                distributions={"training.max_depth": optuna.distributions.IntDistribution(2, 4)},
                state=optuna.trial.TrialState.FAIL,
            )
        )
    return storage


def test_tune_resume_refuses_before_dispatching_any_child(runner: CliRunner, workdir: Path) -> None:
    _seed_study(workdir, failed_trial=False)
    result = runner.invoke(
        app, _argv(workdir, _sources(workdir)["tagged"], "tune", "resume", "s", "--n-trials", "2")
    )
    _assert_refused(result)
    assert "dispatching subprocess child" not in result.output
    assert _n_trials(workdir) == 0


def test_tune_retry_trial_refuses_before_enqueueing(runner: CliRunner, workdir: Path) -> None:
    storage = _seed_study(workdir, failed_trial=True)
    result = runner.invoke(
        app, _argv(workdir, _sources(workdir)["column"], "tune", "retry-trial", "s", "0")
    )
    _assert_refused(result)
    trials = optuna.load_study(study_name="s", storage=storage).trials
    assert [t.state for t in trials] == [optuna.trial.TrialState.FAIL]  # nothing enqueued

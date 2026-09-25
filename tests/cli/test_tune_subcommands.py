"""End-to-end CLI tests for ``rux-ml tune {start,resume,status,retry-trial}``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml.cli import app
from tests.conftest import repo_oracle_toml


@pytest.fixture
def tune_workdir(tmp_path: Path) -> Path:
    """Workdir with synthetic Parquet + base.toml driving a tiny 2-trial / 3-fold sweep."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    df = pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()})
    src = tmp_path / "synth.parquet"
    df.write_parquet(src)

    storage_path = tmp_path / "studies" / "studies.db"
    config = tmp_path / "base.toml"
    config.write_text(
        f"""
[data]
source_path = "{src}"
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
storage_url = "sqlite:///{storage_path}"
artifacts_root = "{tmp_path}/studies/artifacts"

[search_space."training.learning_rate"]
type = "float"
low = 0.1
high = 0.5

[search_space."training.max_depth"]
type = "int"
low = 2
high = 4
"""
        + repo_oracle_toml()
    )
    return tmp_path


def _argv(workdir: Path, *args: str) -> list[str]:
    return ["--config", str(workdir / "base.toml"), *args]


def test_tune_start_in_process_mode_runs_2_trials(
    runner: CliRunner, tune_workdir: Path
) -> None:
    """Explicitly force ``trial_isolation = "in_process"`` and verify the legacy
    PR-007 path still works (no subprocess spawn). Default is "subprocess";
    other tests in this module exercise that path."""
    result = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "tuning.trial_isolation=\"in_process\"",
            "tune", "start", "smoke_in_proc", "--n-trials", "2",
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "isolation: in_process" in result.stderr or "isolation: in_process" in result.stdout
    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_in_proc", storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 2


def test_tune_start_runs_2_trials_and_records_user_attrs(
    runner: CliRunner, tune_workdir: Path
) -> None:
    result = runner.invoke(
        app, _argv(tune_workdir, "tune", "start", "smoke_a", "--n-trials", "2"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "best value (auc):" in result.stdout

    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_a", storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 2

    # PR-015 + PR-006 + PR-011 provenance set lands in user_attrs.
    expected = {
        "data_cfg_hash",
        "features_cfg_hash",
        "training_cfg_hash",
        "tuning_cfg_hash",
        "runs_cfg_hash",
        "registry_cfg_hash",
        "memory_cfg_hash",
        "cv_cfg_hash",
        "root_cfg_hash",
        "git_sha",
        "data_hash",
        "data_bytes_hash",
        "data_logical_hash",
        "metric",
        "peak_rss_mb",  # required since PR-011
    }
    for t in completed:
        assert expected.issubset(t.user_attrs.keys()), expected - t.user_attrs.keys()
        assert t.user_attrs["peak_rss_mb"] > 0

    # Per-trial cv_cfg_hash is identical (no overrides to cv); training_cfg_hash differs
    # (search_space tunes training.learning_rate + training.max_depth).
    hashes = {t.user_attrs["training_cfg_hash"] for t in completed}
    assert len(hashes) == 2


def test_tune_resume_appends_trials(runner: CliRunner, tune_workdir: Path) -> None:
    r1 = runner.invoke(
        app, _argv(tune_workdir, "tune", "start", "smoke_b", "--n-trials", "2"),
        catch_exceptions=False,
    )
    assert r1.exit_code == 0, r1.stderr or r1.stdout
    r2 = runner.invoke(
        app, _argv(tune_workdir, "tune", "resume", "smoke_b", "--n-trials", "1"),
        catch_exceptions=False,
    )
    assert r2.exit_code == 0, r2.stderr or r2.stdout

    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_b", storage=storage)
    assert len(study.trials) == 3


def test_tune_status_reports_best_value(runner: CliRunner, tune_workdir: Path) -> None:
    runner.invoke(
        app, _argv(tune_workdir, "tune", "start", "smoke_c", "--n-trials", "2"),
        catch_exceptions=False,
    )
    result = runner.invoke(app, _argv(tune_workdir, "tune", "status", "smoke_c"))
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "best value (auc):" in result.stdout
    assert "completed=2" in result.stdout


def test_tune_status_missing_study_raises_bad_parameter(
    runner: CliRunner, tune_workdir: Path
) -> None:
    result = runner.invoke(app, _argv(tune_workdir, "tune", "status", "does_not_exist"))
    assert result.exit_code != 0


def test_tune_retry_trial_enqueues_prior_params(
    runner: CliRunner, tune_workdir: Path
) -> None:
    """Retry verb re-enqueues a prior trial's params; new trial completes with same params.

    The retry verb at ``cli/tune.py:retry_trial`` is state-agnostic — it looks up
    ``prior.params`` by trial_id and calls ``study.enqueue_trial(prior.params)``
    + ``study.optimize(n_trials=1)``. Verified end-to-end: 2 initial trials,
    then retry trial 0, then assert trial 2 has params equal to trial 0.
    """
    start_result = runner.invoke(
        app,
        _argv(tune_workdir, "tune", "start", "smoke_retry", "--n-trials", "2"),
        catch_exceptions=False,
    )
    assert start_result.exit_code == 0, start_result.stderr or start_result.stdout

    retry_result = runner.invoke(
        app,
        _argv(tune_workdir, "tune", "retry-trial", "smoke_retry", "0"),
        catch_exceptions=False,
    )
    assert retry_result.exit_code == 0, retry_result.stderr or retry_result.stdout

    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_retry", storage=storage)
    assert len(study.trials) == 3, [t.number for t in study.trials]
    trial_0 = next(t for t in study.trials if t.number == 0)
    trial_2 = next(t for t in study.trials if t.number == 2)
    assert trial_2.params == trial_0.params


def test_tune_retry_trial_missing_trial_raises_bad_parameter(
    runner: CliRunner, tune_workdir: Path
) -> None:
    start_result = runner.invoke(
        app,
        _argv(tune_workdir, "tune", "start", "smoke_retry_missing_trial", "--n-trials", "2"),
        catch_exceptions=False,
    )
    assert start_result.exit_code == 0, start_result.stderr or start_result.stdout

    result = runner.invoke(
        app, _argv(tune_workdir, "tune", "retry-trial", "smoke_retry_missing_trial", "99")
    )
    assert result.exit_code != 0


def test_tune_retry_trial_missing_study_raises_bad_parameter(
    runner: CliRunner, tune_workdir: Path
) -> None:
    result = runner.invoke(
        app, _argv(tune_workdir, "tune", "retry-trial", "nonexistent_study", "0")
    )
    assert result.exit_code != 0


def test_tune_start_honors_tuning_n_trials_from_toml(
    runner: CliRunner, tune_workdir: Path
) -> None:
    """PR-039 Bug 1a (RED): ``tune start`` w/o ``--n-trials`` honors TOML ``[tuning] n_trials``.

    Fixture sets ``[tuning] n_trials = 2``. Pre-fix: Typer binds the parameter
    default ``50`` (``cli/tune.py:133``); banner reads ``n_trials: 50`` and the
    loop runs 50 trials. Post-fix: TOML value wins; banner reads ``n_trials: 2``
    and the loop runs 2 trials.
    """
    result = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "tuning.trial_isolation=\"in_process\"",
            "tune", "start", "smoke_toml_honored",
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "n_trials: 2" in result.stdout, (
        f"banner must report effective n_trials=2 (TOML value); got stdout:\n{result.stdout}"
    )
    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_toml_honored", storage=storage)
    assert len(study.trials) == 2, (
        f"loop must dispatch exactly 2 trials (TOML value); got {len(study.trials)}"
    )


def test_tune_start_cli_flag_overrides_toml_n_trials(
    runner: CliRunner, tune_workdir: Path
) -> None:
    """PR-039 Bug 1a (regression guard): explicit ``--n-trials`` beats TOML ``[tuning] n_trials``.

    Locks the ``CLI > TOML`` half of the override-precedence contract
    (``docs/ARCHITECTURE.md:476-484``) against a Phase 6 fix that overcorrects
    to "TOML always wins". Passes pre-fix AND post-fix.
    """
    result = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "tuning.trial_isolation=\"in_process\"",
            "tune", "start", "smoke_cli_override", "--n-trials", "3",
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "n_trials: 3" in result.stdout, (
        f"banner must report effective n_trials=3 (CLI value); got stdout:\n{result.stdout}"
    )
    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_cli_override", storage=storage)
    assert len(study.trials) == 3, (
        f"loop must dispatch exactly 3 trials (CLI value); got {len(study.trials)}"
    )


def test_tune_resume_honors_tuning_n_trials_from_toml(
    runner: CliRunner, tune_workdir: Path
) -> None:
    """PR-039 Bug 1b (RED): ``tune resume`` w/o ``--n-trials`` honors TOML ``[tuning] n_trials``.

    Same defect class as Bug 1a, at ``cli/tune.py:157``. Seed precursor uses
    explicit ``--n-trials 1`` to isolate the bug under test to the ``resume``
    invocation. Pre-fix: banner reads ``n_trials: 50``; resume adds 50 trials
    (51 total). Post-fix: banner reads ``n_trials: 2``; resume adds 2 (3 total).
    """
    seed = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "tuning.trial_isolation=\"in_process\"",
            "tune", "start", "smoke_resume_toml", "--n-trials", "1",
        ),
        catch_exceptions=False,
    )
    assert seed.exit_code == 0, seed.stderr or seed.stdout
    result = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "tuning.trial_isolation=\"in_process\"",
            "tune", "resume", "smoke_resume_toml",
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "n_trials: 2" in result.stdout, (
        f"banner must report effective n_trials=2 (TOML value); got stdout:\n{result.stdout}"
    )
    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_resume_toml", storage=storage)
    assert len(study.trials) == 3, (
        f"1 seed + 2 resume = 3 trials total (TOML value); got {len(study.trials)}"
    )


def test_tune_resume_cli_flag_overrides_toml_n_trials(
    runner: CliRunner, tune_workdir: Path
) -> None:
    """PR-039 Bug 1b (regression guard): explicit ``--n-trials`` on ``resume`` must beat TOML.

    Mirror of ``test_tune_start_cli_flag_overrides_toml_n_trials`` for the
    ``resume`` verb. Passes pre-fix AND post-fix.
    """
    seed = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "tuning.trial_isolation=\"in_process\"",
            "tune", "start", "smoke_resume_cli_override", "--n-trials", "1",
        ),
        catch_exceptions=False,
    )
    assert seed.exit_code == 0, seed.stderr or seed.stdout
    result = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "tuning.trial_isolation=\"in_process\"",
            "tune", "resume", "smoke_resume_cli_override", "--n-trials", "3",
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "n_trials: 3" in result.stdout, (
        f"banner must report effective n_trials=3 (CLI value); got stdout:\n{result.stdout}"
    )
    storage = f"sqlite:///{tune_workdir}/studies/studies.db"
    study = optuna.load_study(study_name="smoke_resume_cli_override", storage=storage)
    assert len(study.trials) == 4, (
        f"1 seed + 3 resume = 4 trials total (CLI value); got {len(study.trials)}"
    )


@pytest.mark.gpu
def test_tune_start_runs_on_gpu(runner: CliRunner, tune_workdir: Path) -> None:
    result = runner.invoke(
        app,
        _argv(
            tune_workdir,
            "--set", "training.device=cuda",
            "tune", "start", "smoke_gpu", "--n-trials", "2",
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "best value (auc):" in result.stdout

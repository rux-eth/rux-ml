"""End-to-end CLI tests for ``rux-ml runs {list,show,compare}`` (PR-009)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml._internal.env import EnvironmentVersions
from rux_ml._internal.seeds import SeedBag
from rux_ml.cli import app
from rux_ml.config import DataConfig, RunsConfig, RuxMLConfig
from rux_ml.runs import TrialAttrs, data_hashes, one_off_run


@pytest.fixture
def runs_workdir(
    tmp_path: Path, seed_bag: SeedBag, env_versions: EnvironmentVersions
) -> Path:
    """Workdir with a base.toml + 2 completed trials in one study."""
    src = tmp_path / "synth.parquet"
    rng = np.random.default_rng(0)
    pl.DataFrame({"x1": rng.normal(size=50).tolist(), "y": [0, 1] * 25}).write_parquet(src)

    config = tmp_path / "base.toml"
    storage_url = f"sqlite:///{tmp_path}/studies/studies.db"
    config.write_text(
        f"""
[data]
source_path = "{src}"
target_column = "y"

[runs]
storage_url = "{storage_url}"
artifacts_root = "{tmp_path}/studies/artifacts"

[cv]
kind = "kfold"
n_splits = 3
shuffle = true
"""
    )
    # Populate two trials in one named study.
    cfg = RuxMLConfig(
        runs=RunsConfig(storage_url=storage_url, artifacts_root=tmp_path / "studies/artifacts"),
        data=DataConfig(source_path=src, target_column="y"),
    )
    hashes = data_hashes(src)
    with one_off_run(cfg, problem="prob_a", study="wide") as a:
        TrialAttrs.from_cfg(
            cfg,
            hashes,
            metric="auc",
            best_iteration=3,
            peak_rss_mb=0.0,
            bag=seed_bag,
            versions=env_versions,
        ).record(a.trial)
        a.tell(0.91)
    study = optuna.load_study(study_name=a.study.study_name, storage=storage_url)
    trial = study.ask()
    TrialAttrs.from_cfg(
        cfg,
        hashes,
        metric="auc",
        best_iteration=5,
        peak_rss_mb=0.0,
        bag=seed_bag,
        versions=env_versions,
    ).record(trial)
    study.tell(trial, 0.93)
    # Stash the study name for test access via a sentinel file.
    (tmp_path / "study_name.txt").write_text(a.study.study_name)
    return tmp_path


def _argv(workdir: Path, *args: str) -> list[str]:
    return ["--config", str(workdir / "base.toml"), *args]


def test_runs_list_shows_two_trials(runner: CliRunner, runs_workdir: Path) -> None:
    result = runner.invoke(app, _argv(runs_workdir, "runs", "list"))
    assert result.exit_code == 0, result.stderr or result.stdout
    study_name = (runs_workdir / "study_name.txt").read_text().strip()
    assert study_name in result.stdout


def test_runs_list_filters_by_problem_prefix(runner: CliRunner, runs_workdir: Path) -> None:
    result = runner.invoke(app, _argv(runs_workdir, "runs", "list", "--problem", "prob_a"))
    assert result.exit_code == 0, result.stderr or result.stdout
    study_name = (runs_workdir / "study_name.txt").read_text().strip()
    assert study_name in result.stdout

    # Unrelated problem filter → empty result.
    empty = runner.invoke(app, _argv(runs_workdir, "runs", "list", "--problem", "no_such"))
    assert empty.exit_code == 0
    assert "no trials found" in empty.stdout


def test_runs_show_displays_provenance_triple(runner: CliRunner, runs_workdir: Path) -> None:
    study_name = (runs_workdir / "study_name.txt").read_text().strip()
    result = runner.invoke(app, _argv(runs_workdir, "runs", "show", "0", "--study", study_name))
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "trial number: 0" in result.stdout
    assert "value:" in result.stdout
    assert "metric: auc" in result.stdout
    # The 8 cfg_hash fields all show up.
    for layer in (
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
    ):
        assert layer in result.stdout


def test_runs_show_lists_diagnostic_artifacts(
    runner: CliRunner, runs_workdir: Path
) -> None:
    """PR-034: ``runs show`` emits a ``diagnostic artifacts:`` section.

    Optuna rejects uploads onto finished trials, so we create a fresh
    study + trial in this test and upload an artifact inside the
    objective (the production call site).
    """
    import optuna  # noqa: PLC0415
    from optuna.artifacts import FileSystemArtifactStore, upload_artifact  # noqa: PLC0415

    storage_url = f"sqlite:///{runs_workdir}/studies/studies.db"
    study_name = "artifact_show_test"
    store_root = runs_workdir / "studies" / "artifacts" / study_name
    store_root.mkdir(parents=True, exist_ok=True)
    store = FileSystemArtifactStore(str(store_root))
    metrics_path = runs_workdir / "metrics.json"
    metrics_path.write_text('{"fold_0_auc": 0.91}')

    storage = optuna.storages.get_storage(storage_url)
    study = optuna.create_study(study_name=study_name, storage=storage)

    def objective(trial: optuna.Trial) -> float:
        upload_artifact(
            artifact_store=store,
            file_path=str(metrics_path),
            study_or_trial=trial,
        )
        return 0.91

    study.optimize(objective, n_trials=1)

    result = runner.invoke(
        app, _argv(runs_workdir, "runs", "show", "0", "--study", study_name)
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "diagnostic artifacts:" in result.stdout
    assert "metrics.json" in result.stdout
    assert "artifact_id:" in result.stdout


def test_runs_show_errors_on_missing_study(runner: CliRunner, runs_workdir: Path) -> None:
    result = runner.invoke(
        app, _argv(runs_workdir, "runs", "show", "0", "--study", "no_such_study")
    )
    assert result.exit_code != 0


def test_runs_compare_renders_wide_table(runner: CliRunner, runs_workdir: Path) -> None:
    study_name = (runs_workdir / "study_name.txt").read_text().strip()
    result = runner.invoke(
        app, _argv(runs_workdir, "runs", "compare", "0", "1", "--study", study_name)
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    # Two rows surface in the output.
    assert "trial_number" in result.stdout
    assert "value" in result.stdout

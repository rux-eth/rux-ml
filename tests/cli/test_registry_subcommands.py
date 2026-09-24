"""End-to-end CLI tests for ``rux-ml registry {promote,list,rollback}`` (PR-010)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml._internal.env import EnvironmentVersions
from rux_ml._internal.seeds import SeedBag
from rux_ml.cli import app
from rux_ml.config import (
    DataConfig,
    FeaturesConfig,
    KFoldCV,
    RegistryConfig,
    RunsConfig,
    RuxMLConfig,
    XGBoostTraining,
)
from rux_ml.config.features import FeaturesSpec
from rux_ml.runs import TrialAttrs, data_hashes, one_off_run
from tests.conftest import repo_oracle_cfg, repo_oracle_toml


@pytest.fixture
def registry_workdir(
    tmp_path: Path, seed_bag: SeedBag, env_versions: EnvironmentVersions
) -> tuple[Path, str, int]:
    """Workdir with a synthetic Parquet + base.toml + one fully-recorded trial.

    Returns ``(workdir, study_name, trial_number)`` so each test can promote that trial.
    """
    src = tmp_path / "synth.parquet"
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}).write_parquet(src)

    storage_url = f"sqlite:///{tmp_path}/studies/studies.db"
    registry_root = tmp_path / "registry"
    config = tmp_path / "base.toml"
    config.write_text(
        f"""
[data]
source_path = "{src}"
target_column = "y"

[features]
spec = {{ numeric_columns = ["x1", "x2"], categorical_columns = [] }}

[training]
kind = "xgboost"
device = "cpu"
metric = "auc"
n_estimators = 8
max_depth = 3
learning_rate = 0.3

[runs]
storage_url = "{storage_url}"
artifacts_root = "{tmp_path}/studies/artifacts"

[registry]
root = "{registry_root}"

[cv]
kind = "kfold"
n_splits = 3
shuffle = true
"""
        + repo_oracle_toml()
    )

    cfg = RuxMLConfig(
        data=DataConfig(source_path=src, target_column="y"),
        features=FeaturesConfig(spec=FeaturesSpec(numeric_columns=["x1", "x2"])),
        training=XGBoostTraining(
            device="cpu", metric="auc", n_estimators=8, max_depth=3, learning_rate=0.3
        ),
        runs=RunsConfig(storage_url=storage_url, artifacts_root=tmp_path / "studies/artifacts"),
        registry=RegistryConfig(root=registry_root),
        cv=KFoldCV(n_splits=3, shuffle=True),
    )
    hashes = data_hashes(src, oracle=repo_oracle_cfg())
    with one_off_run(cfg, problem="churn_v1", study="wide") as run:
        TrialAttrs.from_cfg(
            cfg,
            hashes,
            metric="auc",
            best_iteration=4,
            peak_rss_mb=0.0,
            bag=seed_bag,
            versions=env_versions,
        ).record(run.trial)
        run.tell(0.91)
    return tmp_path, run.study.study_name, run.trial.number


def _argv(workdir: Path, *args: str) -> list[str]:
    return ["--config", str(workdir / "base.toml"), *args]


def test_registry_promote_writes_bundle_and_prints_version(
    runner: CliRunner, registry_workdir: tuple[Path, str, int]
) -> None:
    workdir, study_name, trial_number = registry_workdir
    result = runner.invoke(
        app,
        _argv(
            workdir,
            "registry",
            "promote",
            "--problem",
            "churn_v1",
            "--study",
            study_name,
            "--trial",
            str(trial_number),
        ),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "promoted: churn_v1@v_" in result.stdout
    assert (workdir / "registry" / "churn_v1" / "champion.json").exists()


def test_registry_list_shows_problems_and_champions(
    runner: CliRunner, registry_workdir: tuple[Path, str, int]
) -> None:
    workdir, study_name, trial_number = registry_workdir
    # First promote, then list.
    runner.invoke(
        app,
        _argv(
            workdir,
            "registry",
            "promote",
            "--problem",
            "churn_v1",
            "--study",
            study_name,
            "--trial",
            str(trial_number),
        ),
        catch_exceptions=False,
    )
    result = runner.invoke(app, _argv(workdir, "registry", "list"))
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "[churn_v1]" in result.stdout
    assert "champion: v_" in result.stdout


def test_registry_list_reports_empty_registry(runner: CliRunner, tmp_path: Path) -> None:
    config = tmp_path / "base.toml"
    config.write_text(f'[registry]\nroot = "{tmp_path}/registry"\n')
    result = runner.invoke(app, ["--config", str(config), "registry", "list"])
    assert result.exit_code == 0
    assert "no registry" in result.stdout or "no problems" in result.stdout


def test_registry_rollback_errors_on_missing_version(
    runner: CliRunner, registry_workdir: tuple[Path, str, int]
) -> None:
    workdir, _, _ = registry_workdir
    result = runner.invoke(
        app,
        _argv(workdir, "registry", "rollback", "--problem", "churn_v1", "--to", "v_no_exist"),
    )
    assert result.exit_code != 0


def test_registry_score_writes_receipt_and_parquet(
    runner: CliRunner, registry_workdir: tuple[Path, str, int]
) -> None:
    """``rux-ml registry score`` end-to-end: promote then score writes receipt + preds parquet."""
    workdir, study_name, trial_number = registry_workdir
    # Promote first so a champion exists.
    promote_result = runner.invoke(
        app,
        _argv(
            workdir,
            "registry",
            "promote",
            "--problem",
            "churn_v1",
            "--study",
            study_name,
            "--trial",
            str(trial_number),
        ),
        catch_exceptions=False,
    )
    assert promote_result.exit_code == 0, promote_result.stderr or promote_result.stdout

    output_dir = workdir / "receipts"
    score_result = runner.invoke(
        app,
        _argv(
            workdir,
            "registry",
            "score",
            "--problem",
            "churn_v1",
            "--output",
            str(output_dir),
        ),
        catch_exceptions=False,
    )
    assert score_result.exit_code == 0, score_result.stderr or score_result.stdout
    assert "scored:  churn_v1@v_" in score_result.stdout
    assert "auc (holdout):" in score_result.stdout
    assert "trial val 0.910000" in score_result.stdout

    receipt_files = list(output_dir.glob("holdout_score_churn_v1_*.json"))
    assert len(receipt_files) == 1
    preds_files = list(output_dir.glob("holdout_preds_churn_v1_*.parquet"))
    assert len(preds_files) == 1


def test_registry_score_missing_champion_raises_bad_parameter(
    runner: CliRunner, tmp_path: Path
) -> None:
    """No champion.json → exit_code != 0 (BadParameter)."""
    config = tmp_path / "base.toml"
    config.write_text(f'[registry]\nroot = "{tmp_path}/registry"\n')
    result = runner.invoke(
        app,
        ["--config", str(config), "registry", "score", "--problem", "nonexistent"],
    )
    assert result.exit_code != 0

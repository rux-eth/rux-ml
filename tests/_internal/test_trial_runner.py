"""Tests for ``rux_ml._internal.trial_runner`` (per PR-008)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest

from rux_ml._internal import trial_runner
from rux_ml.config import MemoryConfig
from tests.conftest import repo_oracle_toml


def test_module_help_works() -> None:
    """``python -m rux_ml._internal.trial_runner --help`` returns 0."""
    result = subprocess.run(
        [sys.executable, "-m", "rux_ml._internal.trial_runner", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "trial_runner" in result.stdout
    assert "--config" in result.stdout
    assert "--study-name" in result.stdout


def test_load_overrides_returns_empty_dict_for_none() -> None:
    assert trial_runner._load_overrides(None) == {}


def test_load_overrides_reads_json_file(tmp_path: Path) -> None:
    p = tmp_path / "overrides.json"
    p.write_text(json.dumps({"training.learning_rate": 0.05}))
    assert trial_runner._load_overrides(p) == {"training.learning_rate": 0.05}


def test_load_overrides_rejects_non_dict_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(TypeError, match="must contain a dict"):
        trial_runner._load_overrides(p)


def test_pin_threads_exports_all_four(monkeypatch: pytest.MonkeyPatch) -> None:
    """``pin_threads`` sets OMP/OPENBLAS/MKL/POLARS env vars from MemoryConfig.

    PR-011 moved the helper from ``trial_runner._pin_thread_env`` to the public
    ``rux_ml._internal.env.pin_threads`` (sub-decision A1).
    """
    import os  # noqa: PLC0415

    from rux_ml._internal.env import pin_threads  # noqa: PLC0415

    # Reset env so we can observe the writes.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        monkeypatch.delenv(key, raising=False)

    memory = MemoryConfig(
        omp_threads=12, openblas_threads=2, mkl_threads=3, polars_threads=8
    )
    pin_threads(memory)
    assert os.environ["OMP_NUM_THREADS"] == "12"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["POLARS_MAX_THREADS"] == "8"


@pytest.fixture
def child_workdir(tmp_path: Path) -> Path:
    """A workdir with a synthetic Parquet + base.toml the child can consume."""
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
n_startup_trials = 1
entropy = 42

[cv]
kind = "kfold"
n_splits = 3
shuffle = true

[runs]
storage_url = "sqlite:///{storage_path}"
"""
        + repo_oracle_toml()
    )
    return tmp_path


def test_trial_runner_runs_one_trial_end_to_end(child_workdir: Path) -> None:
    """Spawn the child via ``python -m`` and assert one COMPLETE trial lands in SQLite."""
    config = child_workdir / "base.toml"
    storage = f"sqlite:///{child_workdir}/studies/studies.db"

    result = subprocess.run(
        [
            sys.executable, "-m", "rux_ml._internal.trial_runner",
            "--config", str(config),
            "--study-name", "child_smoke",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    study = optuna.load_study(study_name="child_smoke", storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 1
    # 8-layer provenance set was recorded (PR-006 + PR-015 + PR-007).
    for key in (
        "data_cfg_hash", "features_cfg_hash", "training_cfg_hash", "tuning_cfg_hash",
        "runs_cfg_hash", "registry_cfg_hash", "memory_cfg_hash", "cv_cfg_hash",
        "root_cfg_hash", "git_sha", "data_hash", "metric",
    ):
        assert key in completed[0].user_attrs


def test_trial_runner_overrides_json_round_trips(child_workdir: Path) -> None:
    """Overrides JSON drives the child's trial_cfg (verified via tuning_cfg_hash change)."""
    config = child_workdir / "base.toml"
    storage = f"sqlite:///{child_workdir}/studies/studies.db"
    overrides = child_workdir / "overrides.json"
    overrides.write_text(json.dumps({"training.learning_rate": 0.123}))

    result = subprocess.run(
        [
            sys.executable, "-m", "rux_ml._internal.trial_runner",
            "--config", str(config),
            "--study-name", "child_overrides",
            "--overrides-json", str(overrides),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    study = optuna.load_study(study_name="child_overrides", storage=storage)
    trial = study.trials[0]
    # training_cfg_hash should differ from the no-overrides baseline because lr changed.
    baseline = subprocess.run(
        [
            sys.executable, "-m", "rux_ml._internal.trial_runner",
            "--config", str(config),
            "--study-name", "child_baseline",
        ],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert baseline.returncode == 0
    baseline_study = optuna.load_study(study_name="child_baseline", storage=storage)
    assert trial.user_attrs["training_cfg_hash"] != baseline_study.trials[0].user_attrs[
        "training_cfg_hash"
    ]

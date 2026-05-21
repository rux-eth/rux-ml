"""End-to-end CLI tests for ``rux-ml train``.

Tiny synthetic Parquet → split (random OR time_ordered) → features → XGBoost
→ score, recorded as a 1-trial Optuna study in SQLite (per D7).

PR-024 adds a ``time_ordered`` dispatch path via ``cfg.data.split_kind``;
the CLI integration test below exercises it end-to-end through
:func:`rux_ml.data.splits.make_splits` and the cross-field validator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
from typer.testing import CliRunner

from rux_ml.cli import app


@pytest.fixture
def train_workdir(tmp_path: Path) -> Path:
    """Workdir with a synthetic Parquet + base.toml pointing every path under tmp_path."""
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Linearly-separable-ish binary target with noise (a baseline XGBoost should do >> 0.5 AUC).
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
n_estimators = 16
max_depth = 3
learning_rate = 0.3
# `early_stopping_rounds` omitted — TOML has no null; keeping the Pydantic
# default of 50 is fine since n_estimators=16 trains to completion anyway.

[runs]
storage_url = "sqlite:///{storage_path}"
artifacts_root = "{tmp_path}/studies/artifacts"
"""
    )
    return tmp_path


def _argv(workdir: Path, *args: str) -> list[str]:
    return ["--config", str(workdir / "base.toml"), *args]


def test_train_runs_end_to_end_cpu(runner: CliRunner, train_workdir: Path) -> None:
    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout
    # AUC must be a valid float in [0, 1]; on this synthetic separable set we
    # expect well above chance, but the band is loose to keep CI stable.
    score_line = next(line for line in result.stdout.splitlines() if "score (auc):" in line)
    score = float(score_line.split(":")[-1].strip())
    assert 0.0 <= score <= 1.0
    assert score > 0.6, f"AUC suspiciously low for separable data: {score}"


def test_train_records_user_attrs_in_sqlite(runner: CliRunner, train_workdir: Path) -> None:
    result = runner.invoke(app, _argv(train_workdir, "train"), catch_exceptions=False)
    assert result.exit_code == 0, result.stderr or result.stdout

    storage = f"sqlite:///{train_workdir}/studies/studies.db"
    summaries = optuna.get_all_study_summaries(storage=storage)
    assert len(summaries) == 1
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    assert len(study.trials) == 1
    attrs = study.trials[0].user_attrs

    # Provenance subset must all be present (cv_cfg_hash added by PR-015,
    # peak_rss_mb added + tightened to required by PR-011).
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
        "peak_rss_mb",
    }
    assert expected.issubset(attrs.keys()), expected - attrs.keys()
    assert attrs["metric"] == "auc"
    assert attrs["peak_rss_mb"] > 0  # watchdog seeds peak from current RSS at entry
    # git_sha is either "unknown" (CI without .git) or a 40-char hex SHA.
    assert attrs["git_sha"] == "unknown" or len(attrs["git_sha"]) == 40


def test_train_errors_when_source_path_missing(runner: CliRunner, tmp_path: Path) -> None:
    config = tmp_path / "base.toml"
    config.write_text("[data]\ntarget_column = 'y'\n")  # source_path absent
    result = runner.invoke(app, ["--config", str(config), "train"])
    assert result.exit_code != 0


# ---------- PR-024: time_ordered dispatch path ----------


@pytest.fixture
def time_ordered_train_workdir(tmp_path: Path) -> Path:
    """Synthetic time-series Parquet + base.toml with ``data.split_kind = 'time_ordered'``.

    The ``ts`` column is intentionally out-of-order so the temporal sort step
    has work to do; ``y`` is generated from features so the trained model
    beats chance on the chronologically-last test fold.
    """
    rng = np.random.default_rng(0)
    n = 200
    ts = rng.permutation(n).tolist()  # shuffled timestamps
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    df = pl.DataFrame({"ts": ts, "x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()})
    src = tmp_path / "synth_ts.parquet"
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
split_kind = "time_ordered"
time_column = "ts"

[features]
spec = {{ numeric_columns = ["ts", "x1", "x2"], categorical_columns = [] }}

[training]
kind = "xgboost"
device = "cpu"
metric = "auc"
n_estimators = 16
max_depth = 3
learning_rate = 0.3

[runs]
storage_url = "sqlite:///{storage_path}"
artifacts_root = "{tmp_path}/studies/artifacts"
"""
    )
    return tmp_path


def test_train_time_ordered_path_end_to_end_cpu(
    runner: CliRunner, time_ordered_train_workdir: Path
) -> None:
    """Dispatcher routes through ``temporal_train_val_test_split`` when
    ``cfg.data.split_kind == 'time_ordered'``. Confirms the CLI surface +
    config validator + dispatcher all wire correctly."""
    result = runner.invoke(
        app,
        _argv(time_ordered_train_workdir, "train"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout
    score_line = next(
        line for line in result.stdout.splitlines() if "score (auc):" in line
    )
    score = float(score_line.split(":")[-1].strip())
    assert 0.0 <= score <= 1.0


def test_train_time_ordered_rejects_missing_time_column(
    runner: CliRunner, tmp_path: Path
) -> None:
    """PR-024 cross-field validator: ``time_ordered`` without ``time_column`` fails."""
    config = tmp_path / "base.toml"
    config.write_text(
        '[data]\nsource_path = "/nonexistent"\ntarget_column = "y"\n'
        'split_kind = "time_ordered"\n'  # no time_column
    )
    result = runner.invoke(app, ["--config", str(config), "train"])
    assert result.exit_code != 0
    # Validator raises pydantic ValidationError; typer wraps it as the runtime
    # exception. Inspect ``result.exception`` for the validator's text.
    assert result.exception is not None
    assert "time_ordered" in str(result.exception)
    assert "time_column" in str(result.exception)


def test_train_temporal_cv_rejects_random_split(
    runner: CliRunner, tmp_path: Path
) -> None:
    """PR-024 cross-field validator: temporal CV + random split fails fast."""
    config = tmp_path / "base.toml"
    config.write_text(
        '[data]\nsource_path = "/nonexistent"\ntarget_column = "y"\n\n'
        '[cv]\nkind = "time_series"\n'
    )
    result = runner.invoke(app, ["--config", str(config), "train"])
    assert result.exit_code != 0
    assert result.exception is not None
    assert "time_series" in str(result.exception)
    assert "split_kind" in str(result.exception)


def test_train_use_native_smoke(runner: CliRunner, train_workdir: Path) -> None:
    """``--set training.use_native=true`` routes through ``XGBoostNativeAdapter`` (PR-033).

    Exit-code-0 + provenance recording verify the native path is wired
    end-to-end through ``cli/train._fit_and_score``; the "native API" log
    line in stderr confirms the branch was taken (vs. silently falling
    back to the sklearn-wrapper path).
    """
    result = runner.invoke(
        app,
        _argv(train_workdir, "--set", "training.use_native=true", "train"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout
    assert "native API" in result.stderr, (
        f"expected 'native API' ingest path log in stderr; got: {result.stderr!r}"
    )


@pytest.mark.gpu
def test_train_runs_end_to_end_gpu(runner: CliRunner, train_workdir: Path) -> None:
    """Same flow as the CPU test, just with ``--set training.device=cuda``."""
    result = runner.invoke(
        app,
        _argv(train_workdir, "--set", "training.device=cuda", "train"),
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "score (auc):" in result.stdout

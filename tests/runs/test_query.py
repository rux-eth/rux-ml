"""Tests for :mod:`rux_ml.runs.query` (PR-009)."""

from __future__ import annotations

from pathlib import Path

import optuna
import polars as pl
import pytest

from rux_ml.config import DataConfig, RunsConfig, RuxMLConfig
from rux_ml.runs import (
    TrialAttrs,
    compare_runs,
    data_hashes,
    ensure_storage_parent,
    list_runs,
    load_run,
    one_off_run,
)


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.parquet"
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]}).write_parquet(p)
    return p


def _cfg_with(tmp_path: Path) -> RuxMLConfig:
    return RuxMLConfig(
        runs=RunsConfig(
            storage_url=f"sqlite:///{tmp_path}/studies/studies.db",
            artifacts_root=tmp_path / "studies/artifacts",
        ),
        data=DataConfig(),
    )


def _populate_two_trials(cfg: RuxMLConfig, parquet_file: Path) -> str:
    """Create two completed trials in one study; return the study name."""
    hashes = data_hashes(parquet_file)
    with one_off_run(cfg, problem="prob_a", study="wide") as a:
        TrialAttrs.from_cfg(cfg, hashes, metric="auc", best_iteration=3).record(a.trial)
        a.tell(0.91)
    # Reuse the same study by passing the exact name through study_name.
    study_name = a.study.study_name
    study = optuna.load_study(study_name=study_name, storage=cfg.runs.storage_url)
    trial = study.ask()
    TrialAttrs.from_cfg(cfg, hashes, metric="auc", best_iteration=5).record(trial)
    study.tell(trial, 0.93)
    return study_name


def test_list_runs_returns_polars_dataframe(tmp_path: Path, parquet_file: Path) -> None:
    cfg = _cfg_with(tmp_path)
    name = _populate_two_trials(cfg, parquet_file)
    df = list_runs(cfg.runs.storage_url)
    assert isinstance(df, pl.DataFrame)
    assert df.height >= 2
    assert "study_name" in df.columns
    assert name in df["study_name"].to_list()


def test_list_runs_filters_by_study_name(tmp_path: Path, parquet_file: Path) -> None:
    cfg = _cfg_with(tmp_path)
    name = _populate_two_trials(cfg, parquet_file)
    df = list_runs(cfg.runs.storage_url, study=name)
    assert df.height >= 2
    assert set(df["study_name"].to_list()) == {name}


def test_list_runs_filters_by_problem_prefix(tmp_path: Path, parquet_file: Path) -> None:
    cfg = _cfg_with(tmp_path)
    _populate_two_trials(cfg, parquet_file)  # problem="prob_a" → study name starts "prob_a_"
    df = list_runs(cfg.runs.storage_url, problem="prob_a")
    assert df.height >= 2
    assert all(s.startswith("prob_a_") for s in df["study_name"].to_list())
    # No false positives for an unrelated problem.
    empty = list_runs(cfg.runs.storage_url, problem="prob_z")
    assert empty.is_empty()


def test_list_runs_empty_when_no_studies(tmp_path: Path) -> None:
    cfg = _cfg_with(tmp_path)
    df = list_runs(cfg.runs.storage_url)
    assert df.is_empty()


def test_load_run_returns_validated_attrs(tmp_path: Path, parquet_file: Path) -> None:
    cfg = _cfg_with(tmp_path)
    name = _populate_two_trials(cfg, parquet_file)
    run = load_run(cfg.runs.storage_url, name, trial_number=0)
    assert run.study_name == name
    assert run.trial_number == 0
    assert run.state == "COMPLETE"
    assert run.value == 0.91
    assert run.attrs is not None
    assert run.attrs.metric == "auc"
    assert run.attrs.best_iteration == 3


def test_load_run_raises_on_missing_study(tmp_path: Path) -> None:
    cfg = _cfg_with(tmp_path)
    with pytest.raises(KeyError, match="not found"):
        load_run(cfg.runs.storage_url, "no_such_study", trial_number=0)


def test_load_run_raises_on_missing_trial(tmp_path: Path, parquet_file: Path) -> None:
    cfg = _cfg_with(tmp_path)
    name = _populate_two_trials(cfg, parquet_file)
    with pytest.raises(KeyError, match="trial #99"):
        load_run(cfg.runs.storage_url, name, trial_number=99)


def test_load_run_returns_none_attrs_on_invalid_user_attrs(tmp_path: Path) -> None:
    """Trials without the required-now provenance set get ``attrs=None``."""
    cfg = _cfg_with(tmp_path)
    ensure_storage_parent(cfg.runs.storage_url)
    study = optuna.create_study(
        study_name="legacy",
        storage=cfg.runs.storage_url,
    )
    trial = study.ask()
    trial.set_user_attr("metric", "auc")  # incomplete schema
    study.tell(trial, 0.5)

    run = load_run(cfg.runs.storage_url, "legacy", trial_number=0)
    assert run.attrs is None


def test_compare_runs_returns_wide_dataframe(tmp_path: Path, parquet_file: Path) -> None:
    cfg = _cfg_with(tmp_path)
    name = _populate_two_trials(cfg, parquet_file)
    df = compare_runs(cfg.runs.storage_url, name, [0, 1])
    assert isinstance(df, pl.DataFrame)
    assert df.height == 2
    assert "trial_number" in df.columns
    assert "value" in df.columns
    # cfg-hash columns surface for the 8 layers.
    cv_col = [c for c in df.columns if c.endswith("cv_cfg_hash")]
    assert len(cv_col) == 1

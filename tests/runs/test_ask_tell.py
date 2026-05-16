"""Tests for :func:`rux_ml.runs.ask_tell.one_off_run` (PR-009)."""

from __future__ import annotations

import time
from pathlib import Path

import optuna
import pytest

from rux_ml.config import DataConfig, RunsConfig, RuxMLConfig
from rux_ml.runs import one_off_run


def _cfg(tmp_path: Path) -> RuxMLConfig:
    return RuxMLConfig(
        runs=RunsConfig(
            storage_url=f"sqlite:///{tmp_path}/studies/studies.db",
            artifacts_root=tmp_path / "studies/artifacts",
        ),
        data=DataConfig(),
    )


def test_one_off_run_records_trial_and_tells_score(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with one_off_run(cfg, problem=None, study=None, direction="maximize") as run:
        run.tell(0.95)

    study = optuna.load_study(study_name=run.study.study_name, storage=cfg.runs.storage_url)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) == 1
    assert completed[0].value == 0.95


def test_one_off_run_marks_fail_if_caller_doesnt_tell(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with one_off_run(cfg, problem=None, study=None) as run:
        pass  # caller forgot to .tell()

    study = optuna.load_study(study_name=run.study.study_name, storage=cfg.runs.storage_url)
    assert any(t.state == optuna.trial.TrialState.FAIL for t in study.trials)


def test_one_off_run_marks_fail_on_exception(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    captured_name: list[str] = []
    with (
        pytest.raises(RuntimeError, match="boom"),
        one_off_run(cfg, problem=None, study=None) as run,
    ):
        captured_name.append(run.study.study_name)
        raise RuntimeError("boom")

    study = optuna.load_study(study_name=captured_name[0], storage=cfg.runs.storage_url)
    assert any(t.state == optuna.trial.TrialState.FAIL for t in study.trials)


def test_one_off_run_creates_distinct_studies_per_call(tmp_path: Path) -> None:
    """Default study_name_template includes a UTC stamp, so successive calls
    create distinct studies (no accidental clobber)."""
    cfg = _cfg(tmp_path)
    with one_off_run(cfg, problem=None, study=None) as a:
        a.tell(0.1)
    time.sleep(1.1)  # ensure stamp ticks at least one second
    with one_off_run(cfg, problem=None, study=None) as b:
        b.tell(0.2)
    assert a.study.study_name != b.study.study_name

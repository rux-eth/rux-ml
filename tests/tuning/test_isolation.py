"""Tests for the parent-side subprocess dispatcher (per PR-008)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import optuna

from rux_ml.config import RuxMLConfig, TuningConfig
from rux_ml.tuning.isolation import (
    _child_argv,
    _mark_trial_failed,
    _write_overrides_json,
    run_subprocess_trial,
)


def test_child_argv_minimal() -> None:
    argv = _child_argv(
        cfg_path=Path("/tmp/base.toml"),
        problem=None,
        study_layer=None,
        study_name="s1",
        overrides_json=None,
    )
    assert argv[0] == sys.executable
    assert "-m" in argv
    assert "rux_ml._internal.trial_runner" in argv
    assert "--config" in argv
    assert "/tmp/base.toml" in argv
    assert "--study-name" in argv
    assert "s1" in argv
    # Optional flags absent when None
    assert "--problem" not in argv
    assert "--study" not in argv
    assert "--overrides-json" not in argv


def test_child_argv_all_options() -> None:
    argv = _child_argv(
        cfg_path=Path("/tmp/base.toml"),
        problem="churn_v1",
        study_layer="wide",
        study_name="s2",
        overrides_json=Path("/tmp/overrides.json"),
    )
    # Each optional flag is followed by its value.
    for flag, val in (
        ("--problem", "churn_v1"),
        ("--study", "wide"),
        ("--overrides-json", "/tmp/overrides.json"),
    ):
        assert flag in argv
        assert argv[argv.index(flag) + 1] == val


def test_write_overrides_json_persists_until_unlink(tmp_path: Path) -> None:
    p = _write_overrides_json({"training.learning_rate": 0.05, "tuning.n_trials": 100})
    try:
        assert p.exists()
        loaded = json.loads(p.read_text())
        assert loaded == {"training.learning_rate": 0.05, "tuning.n_trials": 100}
    finally:
        p.unlink(missing_ok=True)


def test_run_subprocess_trial_invokes_subprocess_run(tmp_path: Path) -> None:
    """``run_subprocess_trial`` wraps ``subprocess.run`` with the constructed argv."""
    cfg = RuxMLConfig(tuning=TuningConfig(trial_timeout_s=None))
    captured: dict[str, object] = {}

    class _FakeCompleted:
        returncode = 0

    def _fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        captured["argv"] = argv
        captured["timeout"] = kwargs.get("timeout")
        return _FakeCompleted()

    with patch("rux_ml.tuning.isolation.subprocess.run", side_effect=_fake_run):
        rc = run_subprocess_trial(
            cfg=cfg,
            cfg_path=tmp_path / "base.toml",
            problem=None,
            study_layer=None,
            study_name="t1",
            overrides={},
        )

    assert rc == 0
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "rux_ml._internal.trial_runner" in argv
    assert "t1" in argv
    assert captured["timeout"] is None  # trial_timeout_s not set


def test_run_subprocess_trial_passes_timeout(tmp_path: Path) -> None:
    cfg = RuxMLConfig(tuning=TuningConfig(trial_timeout_s=42))
    captured_timeout: list[int | None] = []

    class _FakeCompleted:
        returncode = 0

    def _fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        _ = argv
        captured_timeout.append(kwargs.get("timeout"))  # type: ignore[arg-type]
        return _FakeCompleted()

    with patch("rux_ml.tuning.isolation.subprocess.run", side_effect=_fake_run):
        run_subprocess_trial(
            cfg=cfg,
            cfg_path=tmp_path / "base.toml",
            problem=None,
            study_layer=None,
            study_name="t1",
            overrides={},
        )
    assert captured_timeout == [42]


def test_run_subprocess_trial_timeout_returns_124(tmp_path: Path) -> None:
    """On ``subprocess.TimeoutExpired`` the dispatcher returns 124 (GNU `timeout` code)."""
    import subprocess  # noqa: PLC0415

    cfg = RuxMLConfig(tuning=TuningConfig(trial_timeout_s=1))

    def _fake_run(argv: list[str], **kwargs: object) -> object:
        _ = (argv, kwargs)
        raise subprocess.TimeoutExpired(cmd="fake", timeout=1)

    with (
        patch("rux_ml.tuning.isolation.subprocess.run", side_effect=_fake_run),
        patch("rux_ml.tuning.isolation._mark_trial_failed") as fail_mock,
    ):
        rc = run_subprocess_trial(
            cfg=cfg,
            cfg_path=tmp_path / "base.toml",
            problem=None,
            study_layer=None,
            study_name="t1",
            overrides={},
        )
    assert rc == 124
    fail_mock.assert_called_once_with("t1", cfg.runs.storage_url)


def test_run_subprocess_trial_writes_and_cleans_up_overrides_json(tmp_path: Path) -> None:
    """When overrides are non-empty, a tmp JSON is created and deleted after the call."""

    written_paths: list[Path] = []
    real_writer = _write_overrides_json

    def _spy_writer(overrides: dict[str, object]) -> Path:
        p = real_writer(overrides)
        written_paths.append(p)
        return p

    class _FakeCompleted:
        returncode = 0

    def _fake_run(argv: list[str], **kwargs: object) -> _FakeCompleted:
        _ = (argv, kwargs)
        return _FakeCompleted()

    cfg = RuxMLConfig()
    with (
        patch("rux_ml.tuning.isolation._write_overrides_json", side_effect=_spy_writer),
        patch("rux_ml.tuning.isolation.subprocess.run", side_effect=_fake_run),
    ):
        run_subprocess_trial(
            cfg=cfg,
            cfg_path=tmp_path / "base.toml",
            problem=None,
            study_layer=None,
            study_name="t1",
            overrides={"training.learning_rate": 0.05},
        )

    assert len(written_paths) == 1
    # File should be cleaned up after the call.
    assert not written_paths[0].exists()


def test_mark_trial_failed_flips_running_trial_state(tmp_path: Path) -> None:
    """``_mark_trial_failed`` sets RUNNING trials to FAIL in the study storage."""
    storage = f"sqlite:///{tmp_path}/studies.db"
    Path(f"{tmp_path}").mkdir(exist_ok=True, parents=True)
    study = optuna.create_study(study_name="t_mark", storage=storage)
    # creates a RUNNING trial; live `Trial` objects have no .state, but the
    # FrozenTrial reloaded from storage does.
    study.ask()

    pre = optuna.load_study(study_name="t_mark", storage=storage)
    assert any(t.state == optuna.trial.TrialState.RUNNING for t in pre.trials)

    _mark_trial_failed("t_mark", storage)

    post = optuna.load_study(study_name="t_mark", storage=storage)
    assert all(t.state != optuna.trial.TrialState.RUNNING for t in post.trials)
    assert any(t.state == optuna.trial.TrialState.FAIL for t in post.trials)

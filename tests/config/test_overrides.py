"""Override precedence tests (per D17): CLI > env > .env > study > problem > base > defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from rux_ml.config import RuxMLConfig


def test_env_var_overrides_base(base_toml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUXML_TRAINING__LEARNING_RATE", "0.07")
    cfg = RuxMLConfig.from_layers(base_toml)
    assert cfg.training.learning_rate == 0.07


def test_env_var_overrides_study(
    base_toml: Path,
    problem_toml: Path,
    study_toml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUXML_TRAINING__LEARNING_RATE", "0.09")
    cfg = RuxMLConfig.from_layers(
        base_toml,
        problem="dummy",
        study="dummy",
        problems_dir=problem_toml,
        studies_dir=study_toml,
    )
    # env beats study (which set 0.05)
    assert cfg.training.learning_rate == 0.09


def test_cli_override_beats_env(base_toml: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUXML_TRAINING__LEARNING_RATE", "0.07")
    cfg = RuxMLConfig.from_layers(base_toml, overrides={"training.learning_rate": 0.42})
    # CLI dot-path beats env
    assert cfg.training.learning_rate == 0.42


def test_study_overrides_problem(base_toml: Path, problem_toml: Path, study_toml: Path) -> None:
    cfg = RuxMLConfig.from_layers(
        base_toml,
        problem="dummy",
        study="dummy",
        problems_dir=problem_toml,
        studies_dir=study_toml,
    )
    # study set learning_rate=0.05, which beats base's 0.1
    assert cfg.training.learning_rate == 0.05


def test_problem_overrides_base(base_toml: Path, problem_toml: Path) -> None:
    cfg = RuxMLConfig.from_layers(base_toml, problem="dummy", problems_dir=problem_toml)
    # problem set max_depth=7, base had 6
    assert cfg.training.max_depth == 7


def test_full_precedence_chain(
    base_toml: Path,
    problem_toml: Path,
    study_toml: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk every level of the precedence chain on the same field."""
    # Base sets max_depth=6; problem overrides to 7; env overrides to 9; CLI to 11.
    monkeypatch.setenv("RUXML_TRAINING__MAX_DEPTH", "9")
    cfg = RuxMLConfig.from_layers(
        base_toml,
        problem="dummy",
        study="dummy",
        problems_dir=problem_toml,
        studies_dir=study_toml,
        overrides={"training.max_depth": 11},
    )
    assert cfg.training.max_depth == 11

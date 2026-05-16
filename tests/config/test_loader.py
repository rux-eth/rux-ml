"""Tests for the layered TOML loader (per D17)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from rux_ml.config import RuxMLConfig


def test_from_layers_loads_base_only(base_toml: Path) -> None:
    cfg = RuxMLConfig.from_layers(base_toml)
    assert cfg.data.target_column == "y"
    assert cfg.training.learning_rate == 0.1
    assert cfg.training.max_depth == 6


def test_from_layers_overlays_problem(base_toml: Path, problem_toml: Path) -> None:
    cfg = RuxMLConfig.from_layers(base_toml, problem="dummy", problems_dir=problem_toml)
    # problem overrides target_column and max_depth
    assert cfg.data.target_column == "label"
    assert cfg.training.max_depth == 7
    # learning_rate falls through to base
    assert cfg.training.learning_rate == 0.1


def test_from_layers_overlays_study(base_toml: Path, problem_toml: Path, study_toml: Path) -> None:
    cfg = RuxMLConfig.from_layers(
        base_toml,
        problem="dummy",
        study="dummy",
        problems_dir=problem_toml,
        studies_dir=study_toml,
    )
    # study overrides learning_rate
    assert cfg.training.learning_rate == 0.05
    # max_depth falls through to problem
    assert cfg.training.max_depth == 7


def test_extra_keys_in_toml_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        dedent(
            """
            [training]
            learning_rate = 0.1
            this_field_does_not_exist = "boom"
            """
        ).strip()
        + "\n"
    )
    with pytest.raises(ValidationError):
        RuxMLConfig.from_layers(bad)


def test_extra_top_level_section_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text(
        dedent(
            """
            [unknown_layer]
            anything = 1
            """
        ).strip()
        + "\n"
    )
    with pytest.raises(ValidationError):
        RuxMLConfig.from_layers(bad)


def test_runtime_paths_are_reset_after_loading(base_toml: Path) -> None:
    """Class-state TOML paths must reset so subsequent calls don't leak prior layers."""
    RuxMLConfig.from_layers(base_toml)
    # Internal is reset; calling without a TOML uses defaults only
    cfg = RuxMLConfig()
    # `target_column` has no default — None means we got pure defaults, not last call's TOML
    assert cfg.data.target_column is None


def test_default_config_is_constructable() -> None:
    """Pure-defaults instantiation must work for the type system + tests."""
    cfg = RuxMLConfig()
    assert cfg.training.kind == "xgboost"
    assert cfg.training.device == "cuda"
    assert cfg.tuning.sampler == "tpe"
    assert cfg.tuning.pruner == "hyperband"
    assert cfg.memory.watchdog_threshold_gb == 28.0

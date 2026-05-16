"""SearchSpec discriminated-union round-trip (per D16)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rux_ml.config import (
    CatSpec,
    FloatSpec,
    IntSpec,
    RuxMLConfig,
)


def test_search_space_round_trips_through_toml(
    base_toml: Path, problem_toml: Path, study_toml: Path
) -> None:
    cfg = RuxMLConfig.from_layers(
        base_toml,
        problem="dummy",
        study="dummy",
        problems_dir=problem_toml,
        studies_dir=study_toml,
    )
    assert "training.learning_rate" in cfg.search_space
    assert "training.max_depth" in cfg.search_space

    lr_spec = cfg.search_space["training.learning_rate"]
    assert isinstance(lr_spec, FloatSpec)
    assert lr_spec.low == 1e-4
    assert lr_spec.high == 0.3
    assert lr_spec.log is True

    md_spec = cfg.search_space["training.max_depth"]
    assert isinstance(md_spec, IntSpec)
    assert md_spec.low == 3
    assert md_spec.high == 12
    assert md_spec.log is False


def test_categorical_spec_round_trips(tmp_path: Path) -> None:
    toml = tmp_path / "study.toml"
    toml.write_text(
        '[search_space."training.tree_method"]\n'
        'type = "categorical"\n'
        'choices = ["hist", "approx"]\n'
    )
    cfg = RuxMLConfig.from_layers(toml)
    spec = cfg.search_space["training.tree_method"]
    assert isinstance(spec, CatSpec)
    assert spec.choices == ["hist", "approx"]


def test_searchspec_discriminator_rejects_invalid_type(tmp_path: Path) -> None:
    """Discriminator should fail-fast on an unknown `type` value."""
    toml = tmp_path / "bad.toml"
    toml.write_text(
        '[search_space."training.lr"]\ntype = "weird_distribution"\nlow = 0\nhigh = 1\n'
    )
    with pytest.raises(ValidationError):
        RuxMLConfig.from_layers(toml)

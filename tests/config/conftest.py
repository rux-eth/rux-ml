"""Fixtures for config-layer tests."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def base_toml(tmp_path: Path) -> Path:
    """Minimal valid base.toml inside a temp dir; used as the floor of the layer stack."""
    p = tmp_path / "base.toml"
    p.write_text(
        dedent(
            """
            [data]
            target_column = "y"

            [training]
            kind = "xgboost"
            learning_rate = 0.1
            max_depth = 6
            """
        ).strip()
        + "\n"
    )
    return p


@pytest.fixture
def problem_toml(tmp_path: Path) -> Path:
    p = tmp_path / "problems"
    p.mkdir(exist_ok=True)
    (p / "dummy.toml").write_text(
        dedent(
            """
            [data]
            target_column = "label"  # overrides base

            [training]
            kind = "xgboost"
            max_depth = 7  # overrides base; learning_rate stays at base
            """
        ).strip()
        + "\n"
    )
    return p


@pytest.fixture
def study_toml(tmp_path: Path) -> Path:
    p = tmp_path / "studies"
    p.mkdir(exist_ok=True)
    (p / "dummy.toml").write_text(
        dedent(
            """
            [training]
            kind = "xgboost"
            learning_rate = 0.05  # overrides base; max_depth stays at problem (7)

            [search_space."training.learning_rate"]
            type = "float"
            low = 1e-4
            high = 0.3
            log = true

            [search_space."training.max_depth"]
            type = "int"
            low = 3
            high = 12
            """
        ).strip()
        + "\n"
    )
    return p

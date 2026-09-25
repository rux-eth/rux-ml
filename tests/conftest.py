"""Shared pytest fixtures for the rux-ml test suite.

Per-layer fixtures live in each ``tests/<layer>/conftest.py``. This file
holds cross-layer helpers — currently the PR-013 :class:`SeedBag` and
:class:`EnvironmentVersions` test doubles used wherever a test needs to
construct a ``TrialAttrs`` directly (rather than going through the full
trial body that produces them). PR-040 adds the oracle-quarantine fixtures,
which read ``[data.oracle]`` from the repo's ``configs/base.toml`` so no test
restates the namespace or tag-file values.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rux_ml._internal.env import EnvironmentVersions
from rux_ml._internal.seeds import SeedBag, make_seed_bag

if TYPE_CHECKING:
    from rux_ml.config.data import OracleQuarantineConfig

REPO_BASE_TOML = Path(__file__).resolve().parents[1] / "configs" / "base.toml"


@pytest.fixture
def seed_bag() -> SeedBag:
    """Deterministic :class:`SeedBag` for tests (entropy=42, trial_number=0)."""
    return make_seed_bag(master_entropy=42, trial_number=0)


@pytest.fixture
def env_versions() -> EnvironmentVersions:
    """Synthetic :class:`EnvironmentVersions` for tests — no nvidia-smi shell-out.

    Mirrors a CPU-only dev host: ``gpu_model`` / ``driver_version`` are
    ``None``; the required fields carry stable test placeholders so tests
    asserting on TrialAttrs round-trips have known values.
    """
    return EnvironmentVersions(
        xgboost_version="3.2.0",
        cuda_runtime_version="12.9",
        omp_threads=1,
        image_digest="sha256:test",
        gpu_model=None,
        driver_version=None,
    )


def repo_oracle_values() -> dict[str, str]:
    """``[data.oracle]`` from the repo's ``configs/base.toml`` (PR-040) — the single source."""
    return dict(tomllib.loads(REPO_BASE_TOML.read_text())["data"]["oracle"])


def repo_oracle_cfg() -> OracleQuarantineConfig:
    """The repo's :class:`OracleQuarantineConfig`; plain-function twin of ``oracle_cfg``.

    For module-level test helpers that build configs or hash data without
    fixture access (``from ..conftest import repo_oracle_cfg``).
    """
    from rux_ml.config.data import OracleQuarantineConfig  # noqa: PLC0415

    return OracleQuarantineConfig(**repo_oracle_values())


def repo_oracle_toml() -> str:
    """A ``[data.oracle]`` table (repo values) to append to a test-written TOML."""
    values = repo_oracle_values()
    return (
        f'\n[data.oracle]\nnamespace = "{values["namespace"]}"\ntag_file = "{values["tag_file"]}"\n'
    )


@pytest.fixture
def oracle_values() -> dict[str, str]:
    return repo_oracle_values()


@pytest.fixture
def oracle_cfg() -> OracleQuarantineConfig:
    return repo_oracle_cfg()


@pytest.fixture
def oracle_toml() -> str:
    return repo_oracle_toml()

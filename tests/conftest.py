"""Shared pytest fixtures for the rux-ml test suite.

Per-layer fixtures live in each ``tests/<layer>/conftest.py``. This file
holds cross-layer helpers — currently the PR-013 :class:`SeedBag` and
:class:`EnvironmentVersions` test doubles used wherever a test needs to
construct a ``TrialAttrs`` directly (rather than going through the full
trial body that produces them).
"""

from __future__ import annotations

import pytest

from rux_ml._internal.env import EnvironmentVersions
from rux_ml._internal.seeds import SeedBag, make_seed_bag


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

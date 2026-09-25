"""Tests for the residual provenance helpers (PR-007 → PR-009 formalization).

PR-009 absorbed ``build_user_attrs`` into :class:`rux_ml.runs.attrs.TrialAttrs`;
those tests now live in ``tests/runs/test_attrs.py``. This module covers the
remaining utility helpers (``HASH_LAYERS``, ``data_hashes``,
``ensure_storage_parent``, ``study_name``).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rux_ml.config import RuxMLConfig
from rux_ml.runs.provenance import (
    HASH_LAYERS,
    data_hashes,
    ensure_storage_parent,
    study_name,
)
from tests.conftest import repo_oracle_cfg


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.parquet"
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]}).write_parquet(p)
    return p


def test_hash_layers_is_the_8_layer_set() -> None:
    """8-layer provenance triple per PR-006 + PR-015 (`cv` added)."""
    assert set(HASH_LAYERS) == {
        "data",
        "features",
        "training",
        "tuning",
        "runs",
        "registry",
        "memory",
        "cv",
    }


def test_data_hashes_returns_all_three_components(parquet_file: Path) -> None:
    hashes = data_hashes(parquet_file, oracle=repo_oracle_cfg())
    assert set(hashes) == {"data_hash", "data_bytes_hash", "data_logical_hash"}
    assert hashes["data_hash"] == f"{hashes['data_bytes_hash']}|{hashes['data_logical_hash']}"


def test_ensure_storage_parent_creates_dir_for_sqlite_url(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "studies.db"
    ensure_storage_parent(f"sqlite:///{target}")
    assert target.parent.exists()


def test_ensure_storage_parent_no_op_for_non_sqlite() -> None:
    """Non-sqlite URLs are passed through; no dir creation attempted."""
    # Should not raise.
    ensure_storage_parent("postgresql://localhost/foo")


def test_study_name_substitution() -> None:
    cfg = RuxMLConfig()  # default study_name_template = "{problem}_{study}_{stamp}"
    name = study_name(cfg, problem="churn_v1", study="wide")
    assert name.startswith("churn_v1_wide_")
    # stamp is YYYYMMDDTHHMMSSZ → 16 chars
    stamp = name.split("_")[-1]
    assert len(stamp) == 16
    assert stamp.endswith("Z")


def test_study_name_falls_back_to_defaults_when_unset() -> None:
    cfg = RuxMLConfig()
    name = study_name(cfg, problem=None, study=None)
    assert name.startswith("default_oneoff_")

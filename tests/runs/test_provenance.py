"""Tests for the provenance helpers extracted in PR-007 (sub-decision D1)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from rux_ml.config import KFoldCV, RuxMLConfig, StratifiedKFoldCV
from rux_ml.runs.provenance import (
    HASH_LAYERS,
    build_user_attrs,
    data_hashes,
    ensure_storage_parent,
    study_name,
)


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
    hashes = data_hashes(parquet_file)
    assert set(hashes) == {"data_hash", "data_bytes_hash", "data_logical_hash"}
    assert hashes["data_hash"] == f"{hashes['data_bytes_hash']}|{hashes['data_logical_hash']}"


def test_build_user_attrs_includes_full_8_layer_set(parquet_file: Path) -> None:
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    attrs = build_user_attrs(cfg, hashes)
    for layer in HASH_LAYERS:
        assert f"{layer}_cfg_hash" in attrs
    assert "root_cfg_hash" in attrs
    assert "git_sha" in attrs
    assert "data_hash" in attrs


def test_build_user_attrs_changes_when_cv_strategy_changes(parquet_file: Path) -> None:
    hashes = data_hashes(parquet_file)
    a = build_user_attrs(RuxMLConfig(cv=KFoldCV(n_splits=5)), hashes)
    b = build_user_attrs(RuxMLConfig(cv=StratifiedKFoldCV(n_splits=5)), hashes)
    assert a["cv_cfg_hash"] != b["cv_cfg_hash"]
    assert a["root_cfg_hash"] != b["root_cfg_hash"]


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

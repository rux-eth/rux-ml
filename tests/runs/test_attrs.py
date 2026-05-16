"""Tests for :class:`rux_ml.runs.attrs.TrialAttrs` (PR-009 + PR-013)."""

from __future__ import annotations

from pathlib import Path

import optuna
import polars as pl
import pytest
from pydantic import ValidationError

from rux_ml._internal.env import EnvironmentVersions
from rux_ml._internal.seeds import SeedBag
from rux_ml.config import KFoldCV, RuxMLConfig, StratifiedKFoldCV
from rux_ml.runs import HASH_LAYERS, TrialAttrs, data_hashes


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.parquet"
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]}).write_parquet(p)
    return p


def test_from_cfg_populates_required_now_fields(
    parquet_file: Path, seed_bag: SeedBag, env_versions: EnvironmentVersions
) -> None:
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    attrs = TrialAttrs.from_cfg(
        cfg, hashes, metric="auc", peak_rss_mb=123.4, bag=seed_bag, versions=env_versions
    )
    for layer in HASH_LAYERS:
        assert getattr(attrs, f"{layer}_cfg_hash")
    assert attrs.root_cfg_hash
    assert attrs.git_sha
    assert attrs.data_hash == hashes["data_hash"]
    assert attrs.data_bytes_hash == hashes["data_bytes_hash"]
    assert attrs.data_logical_hash == hashes["data_logical_hash"]
    assert attrs.metric == "auc"
    # Required (PR-011): peak_rss_mb was passed explicitly.
    assert attrs.peak_rss_mb == 123.4
    # PR-013-required environment block.
    assert attrs.entropy_hex == seed_bag.entropy_hex
    assert attrs.image_digest == env_versions.image_digest
    assert attrs.xgboost_version == env_versions.xgboost_version
    assert attrs.cuda_runtime_version == env_versions.cuda_runtime_version
    assert attrs.omp_threads == env_versions.omp_threads
    # PR-013 optional GPU-only fields stay None on CPU-only fixtures.
    assert attrs.gpu_model is None
    assert attrs.driver_version is None
    # best_iteration stays Optional (caller didn't pass it).
    assert attrs.best_iteration is None


def test_from_cfg_changes_when_cv_strategy_changes(
    parquet_file: Path, seed_bag: SeedBag, env_versions: EnvironmentVersions
) -> None:
    hashes = data_hashes(parquet_file)
    a = TrialAttrs.from_cfg(
        RuxMLConfig(cv=KFoldCV(n_splits=5)),
        hashes,
        metric="auc",
        peak_rss_mb=0.0,
        bag=seed_bag,
        versions=env_versions,
    )
    b = TrialAttrs.from_cfg(
        RuxMLConfig(cv=StratifiedKFoldCV(n_splits=5)),
        hashes,
        metric="auc",
        peak_rss_mb=0.0,
        bag=seed_bag,
        versions=env_versions,
    )
    assert a.cv_cfg_hash != b.cv_cfg_hash
    assert a.root_cfg_hash != b.root_cfg_hash


def test_record_writes_only_non_none_fields(
    parquet_file: Path, seed_bag: SeedBag, env_versions: EnvironmentVersions
) -> None:
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    attrs = TrialAttrs.from_cfg(
        cfg,
        hashes,
        metric="auc",
        best_iteration=42,
        peak_rss_mb=234.5,
        bag=seed_bag,
        versions=env_versions,
    )

    study = optuna.create_study()
    trial = study.ask()
    attrs.record(trial)
    study.tell(trial, 0.5)

    written = study.trials[0].user_attrs
    # All required-now fields land.
    for layer in HASH_LAYERS:
        assert f"{layer}_cfg_hash" in written
    assert "root_cfg_hash" in written
    assert "metric" in written
    assert written["metric"] == "auc"
    # best_iteration was explicitly set → it lands.
    assert written["best_iteration"] == 42
    # peak_rss_mb is required (PR-011) — always lands.
    assert "peak_rss_mb" in written
    # PR-013 required fields land.
    assert written["entropy_hex"] == seed_bag.entropy_hex
    assert written["image_digest"] == env_versions.image_digest
    assert written["xgboost_version"] == env_versions.xgboost_version
    assert written["cuda_runtime_version"] == env_versions.cuda_runtime_version
    assert written["omp_threads"] == env_versions.omp_threads
    # Optional fields left as None do NOT pollute user_attrs.
    assert "gpu_model" not in written
    assert "driver_version" not in written


def test_from_trial_round_trips(
    parquet_file: Path, seed_bag: SeedBag, env_versions: EnvironmentVersions
) -> None:
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    original = TrialAttrs.from_cfg(
        cfg,
        hashes,
        metric="logloss",
        best_iteration=7,
        peak_rss_mb=345.6,
        bag=seed_bag,
        versions=env_versions,
    )

    study = optuna.create_study()
    trial = study.ask()
    original.record(trial)
    study.tell(trial, 0.3)

    reloaded = TrialAttrs.from_trial(study.trials[0])
    assert reloaded.metric == "logloss"
    assert reloaded.best_iteration == 7
    assert reloaded.cv_cfg_hash == original.cv_cfg_hash
    assert reloaded.root_cfg_hash == original.root_cfg_hash
    assert reloaded.entropy_hex == original.entropy_hex
    assert reloaded.image_digest == original.image_digest
    assert reloaded.xgboost_version == original.xgboost_version
    assert reloaded.cuda_runtime_version == original.cuda_runtime_version
    assert reloaded.omp_threads == original.omp_threads


def test_from_trial_raises_on_missing_required_field() -> None:
    """A trial without the required-now fields fails ``TrialAttrs.from_trial``.

    This is the contract PR-010's registry promotion will rely on.
    """
    study = optuna.create_study()
    trial = study.ask()
    trial.set_user_attr("metric", "auc")  # only one of many required fields
    study.tell(trial, 0.5)

    with pytest.raises(ValidationError):
        TrialAttrs.from_trial(study.trials[0])


def test_from_trial_ignores_unknown_user_attrs(
    parquet_file: Path, seed_bag: SeedBag, env_versions: EnvironmentVersions
) -> None:
    """``extra="ignore"`` — unrelated user_attrs don't break validation."""
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    attrs = TrialAttrs.from_cfg(
        cfg, hashes, metric="auc", peak_rss_mb=123.4, bag=seed_bag, versions=env_versions
    )

    study = optuna.create_study()
    trial = study.ask()
    attrs.record(trial)
    trial.set_user_attr("user_added_debug_flag", "anything")
    study.tell(trial, 0.5)

    # Should not raise; unrelated key is silently dropped on read.
    reloaded = TrialAttrs.from_trial(study.trials[0])
    assert reloaded.metric == "auc"

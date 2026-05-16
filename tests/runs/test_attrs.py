"""Tests for :class:`rux_ml.runs.attrs.TrialAttrs` (PR-009)."""

from __future__ import annotations

from pathlib import Path

import optuna
import polars as pl
import pytest
from pydantic import ValidationError

from rux_ml.config import KFoldCV, RuxMLConfig, StratifiedKFoldCV
from rux_ml.runs import HASH_LAYERS, TrialAttrs, data_hashes


@pytest.fixture
def parquet_file(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.parquet"
    pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]}).write_parquet(p)
    return p


def test_from_cfg_populates_required_now_fields(parquet_file: Path) -> None:
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    attrs = TrialAttrs.from_cfg(cfg, hashes, metric="auc")
    for layer in HASH_LAYERS:
        assert getattr(attrs, f"{layer}_cfg_hash")
    assert attrs.root_cfg_hash
    assert attrs.git_sha
    assert attrs.data_hash == hashes["data_hash"]
    assert attrs.data_bytes_hash == hashes["data_bytes_hash"]
    assert attrs.data_logical_hash == hashes["data_logical_hash"]
    assert attrs.metric == "auc"
    # Optional fields default to None.
    assert attrs.best_iteration is None
    assert attrs.entropy_hex is None
    assert attrs.peak_rss_mb is None


def test_from_cfg_changes_when_cv_strategy_changes(parquet_file: Path) -> None:
    hashes = data_hashes(parquet_file)
    a = TrialAttrs.from_cfg(RuxMLConfig(cv=KFoldCV(n_splits=5)), hashes, metric="auc")
    b = TrialAttrs.from_cfg(RuxMLConfig(cv=StratifiedKFoldCV(n_splits=5)), hashes, metric="auc")
    assert a.cv_cfg_hash != b.cv_cfg_hash
    assert a.root_cfg_hash != b.root_cfg_hash


def test_record_writes_only_non_none_fields(parquet_file: Path) -> None:
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    attrs = TrialAttrs.from_cfg(cfg, hashes, metric="auc", best_iteration=42)

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
    # Optional fields left as None do NOT pollute user_attrs.
    assert "entropy_hex" not in written
    assert "peak_rss_mb" not in written


def test_from_trial_round_trips(parquet_file: Path) -> None:
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    original = TrialAttrs.from_cfg(cfg, hashes, metric="logloss", best_iteration=7)

    study = optuna.create_study()
    trial = study.ask()
    original.record(trial)
    study.tell(trial, 0.3)

    reloaded = TrialAttrs.from_trial(study.trials[0])
    assert reloaded.metric == "logloss"
    assert reloaded.best_iteration == 7
    assert reloaded.cv_cfg_hash == original.cv_cfg_hash
    assert reloaded.root_cfg_hash == original.root_cfg_hash


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


def test_from_trial_ignores_unknown_user_attrs(parquet_file: Path) -> None:
    """``extra="ignore"`` — unrelated user_attrs don't break validation."""
    cfg = RuxMLConfig()
    hashes = data_hashes(parquet_file)
    attrs = TrialAttrs.from_cfg(cfg, hashes, metric="auc")

    study = optuna.create_study()
    trial = study.ask()
    attrs.record(trial)
    trial.set_user_attr("user_added_debug_flag", "anything")
    study.tell(trial, 0.5)

    # Should not raise; unrelated key is silently dropped on read.
    reloaded = TrialAttrs.from_trial(study.trials[0])
    assert reloaded.metric == "auc"

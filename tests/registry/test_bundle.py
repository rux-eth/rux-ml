"""Tests for ``rux_ml.registry.bundle`` save/load round trip (per PR-010)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rux_ml.registry.bundle import (
    BOOSTER_FILENAME,
    MANIFEST_FILENAME,
    PIPELINE_FILENAME,
    load_bundle,
    save_bundle,
)
from rux_ml.registry.manifest import LibraryVersions, ModelManifest, PromotedFrom


def _trivial_pipeline() -> Pipeline:
    return Pipeline(steps=[("scaler", StandardScaler())])


def _trivial_booster() -> xgb.Booster:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    y = (rng.random(20) > 0.5).astype(int)
    dmatrix = xgb.DMatrix(x, label=y)
    return xgb.train({"objective": "binary:logistic", "verbosity": 0}, dmatrix, num_boost_round=2)


def _sample_manifest() -> ModelManifest:
    return ModelManifest(
        version="v_2026_05_16_a8f3c2",
        problem="churn_v1",
        promoted_from=PromotedFrom(study="s1", trial_number=0, metric_value=0.9),
        metric_name="auc",
        metric_value=0.9,
        data_cfg_hash="d" * 64,
        features_cfg_hash="f" * 64,
        training_cfg_hash="t" * 64,
        tuning_cfg_hash="u" * 64,
        runs_cfg_hash="r" * 64,
        registry_cfg_hash="g" * 64,
        memory_cfg_hash="m" * 64,
        cv_cfg_hash="c" * 64,
        root_cfg_hash="0" * 64,
        git_sha="abc1234",
        data_hash="bytes|logical",
        data_bytes_hash="b" * 64,
        data_logical_hash="l" * 64,
        library_versions=LibraryVersions(
            xgboost="3.2.0",
            skops="0.14.0",
            scikit_learn="1.8.0",
            polars="1.40.1",
            numpy="2.4.5",
            rux_ml="0.0.1",
        ),
        feature_list_hash="e" * 64,
        created_at="2026-05-16T15:00:00+00:00",
    )


def test_save_bundle_writes_three_files(tmp_path: Path) -> None:
    save_bundle(_trivial_pipeline(), _trivial_booster(), _sample_manifest(), tmp_path / "bundle")
    for name in (PIPELINE_FILENAME, BOOSTER_FILENAME, MANIFEST_FILENAME):
        assert (tmp_path / "bundle" / name).exists()


def test_load_bundle_round_trip(tmp_path: Path) -> None:
    pipeline = _trivial_pipeline()
    booster = _trivial_booster()
    manifest = _sample_manifest()
    save_bundle(pipeline, booster, manifest, tmp_path / "bundle")

    loaded_pipeline, loaded_booster, loaded_manifest = load_bundle(tmp_path / "bundle")
    assert isinstance(loaded_pipeline, Pipeline)
    assert isinstance(loaded_booster, xgb.Booster)
    assert loaded_manifest == manifest


def test_load_bundle_raises_on_missing_file(tmp_path: Path) -> None:
    save_bundle(_trivial_pipeline(), _trivial_booster(), _sample_manifest(), tmp_path / "bundle")
    (tmp_path / "bundle" / MANIFEST_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
        load_bundle(tmp_path / "bundle")

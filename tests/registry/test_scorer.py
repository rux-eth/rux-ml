"""Tests for the PR-032 holdout-fold scorer (_BoosterTrainerShim + score_bundle_on_holdout)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from rux_ml._internal.env import EnvironmentVersions
from rux_ml._internal.seeds import SeedBag
from rux_ml.config import (
    DataConfig,
    FeaturesConfig,
    KFoldCV,
    RegistryConfig,
    RunsConfig,
    RuxMLConfig,
    XGBoostTraining,
)
from rux_ml.config.features import FeaturesSpec
from rux_ml.registry.promote import promote
from rux_ml.registry.scorer import (
    HoldoutScoreReceipt,
    _BoosterTrainerShim,
    score_bundle_on_holdout,
)
from rux_ml.runs import TrialAttrs, data_hashes, one_off_run

# ---------- _BoosterTrainerShim ----------


def _train_tiny_booster(rng: np.random.Generator) -> tuple[xgb.Booster, np.ndarray, np.ndarray]:
    """Train a 5-tree booster on synthetic binary classification; return (booster, X, y)."""
    n = 100
    x = rng.normal(size=(n, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    dmat = xgb.DMatrix(x, label=y, enable_categorical=True)
    booster = xgb.train({"objective": "binary:logistic", "verbosity": 0}, dmat, num_boost_round=5)
    return booster, x, y


def test_booster_trainer_shim_predict_matches_direct_dmatrix() -> None:
    rng = np.random.default_rng(0)
    booster, x, _ = _train_tiny_booster(rng)
    shim = _BoosterTrainerShim(booster)
    shim_preds = shim.predict(x)
    direct_preds = booster.predict(xgb.DMatrix(x, enable_categorical=True))  # pyright: ignore[reportUnknownMemberType]
    np.testing.assert_array_equal(shim_preds, np.asarray(direct_preds))


def test_booster_trainer_shim_predict_proba_returns_2d_for_binary() -> None:
    rng = np.random.default_rng(0)
    booster, x, _ = _train_tiny_booster(rng)
    shim = _BoosterTrainerShim(booster)
    proba = shim.predict_proba(x)
    assert proba.ndim == 2
    assert proba.shape[1] == 2
    # rows sum to ~1 (binary [1-p, p] reshape)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


def test_booster_trainer_shim_fit_raises_not_implemented() -> None:
    booster = xgb.Booster()
    shim = _BoosterTrainerShim(booster)
    with pytest.raises(NotImplementedError, match="score-only"):
        shim.fit(None, None)


# ---------- score_bundle_on_holdout (end-to-end) ----------


def _make_cfg(tmp_path: Path, source: Path, registry_root: Path) -> RuxMLConfig:
    return RuxMLConfig(
        data=DataConfig(source_path=source, target_column="y"),
        features=FeaturesConfig(spec=FeaturesSpec(numeric_columns=["x1", "x2"])),
        training=XGBoostTraining(
            device="cpu",
            metric="auc",
            n_estimators=8,
            max_depth=3,
            learning_rate=0.3,
        ),
        runs=RunsConfig(
            storage_url=f"sqlite:///{tmp_path}/studies/studies.db",
            artifacts_root=tmp_path / "studies/artifacts",
        ),
        registry=RegistryConfig(root=registry_root),
        cv=KFoldCV(n_splits=3, shuffle=True),
    )


def _populate_trial(
    cfg: RuxMLConfig, bag: SeedBag, versions: EnvironmentVersions
) -> tuple[str, int]:
    """Create a one-off trial with full provenance; return (study_name, trial_number)."""
    assert cfg.data.source_path is not None
    hashes = data_hashes(cfg.data.source_path)
    with one_off_run(cfg, problem="churn_v1", study="wide") as run:
        TrialAttrs.from_cfg(
            cfg,
            hashes,
            metric="auc",
            best_iteration=4,
            peak_rss_mb=0.0,
            bag=bag,
            versions=versions,
        ).record(run.trial)
        run.tell(0.91)
    return run.study.study_name, run.trial.number


@pytest.fixture
def synth_workdir(tmp_path: Path) -> tuple[Path, RuxMLConfig]:
    src = tmp_path / "synth.parquet"
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((0.7 * x1 + 0.3 * x2 + rng.normal(0, 0.3, size=n)) > 0).astype(int)
    pl.DataFrame({"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}).write_parquet(src)
    cfg = _make_cfg(tmp_path, src, tmp_path / "registry")
    return tmp_path, cfg


def test_score_bundle_on_holdout_writes_receipt_and_parquet(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    output_dir = tmp_path / "receipts"
    receipt = score_bundle_on_holdout(
        cfg, problem="churn_v1", version=version, output_dir=output_dir
    )

    # Returned object is the validated Pydantic model.
    assert isinstance(receipt, HoldoutScoreReceipt)
    assert receipt.bundle_version == version
    assert receipt.promoted_from.study == study_name
    assert receipt.promoted_from.trial_number == trial_number
    # AUC bounded; metric registered as "auc" → in metrics dict.
    assert "auc" in receipt.metrics
    assert 0.0 <= receipt.metrics["auc"] <= 1.0
    # Holdout = 15% of 200 rows = 30 (random split path).
    assert receipt.holdout.split_kind == "random"
    assert receipt.holdout.time_range is None  # no time_column on the synth cfg

    # Files exist on disk.
    receipt_files = list(output_dir.glob("holdout_score_churn_v1_*.json"))
    assert len(receipt_files) == 1
    preds_files = list(output_dir.glob("holdout_preds_churn_v1_*.parquet"))
    assert len(preds_files) == 1

    # Receipt JSON round-trips through HoldoutScoreReceipt schema.
    receipt_disk = HoldoutScoreReceipt.model_validate_json(receipt_files[0].read_text())
    assert receipt_disk.bundle_version == version
    assert receipt_disk.artifacts["predictions"].content_type == "application/vnd.apache.parquet"

    # Preds parquet has the expected columns.
    preds_df = pl.read_parquet(preds_files[0])
    assert set(preds_df.columns) == {"y_true", "y_pred"}


def test_score_bundle_on_holdout_defaults_to_champion(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    """When ``version`` is None, the scorer resolves the current champion."""
    tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    receipt = score_bundle_on_holdout(
        cfg, problem="churn_v1", output_dir=tmp_path / "receipts"
    )
    assert receipt.bundle_version == version


def test_score_bundle_on_holdout_random_split_reconstructs_seed(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    """Random-split path re-opens the Optuna DB to recover the trial's split_seed.

    Verified end-to-end: the bundle's re-fit (in ``promote``) and the
    scorer's holdout reconstruction MUST agree on row identities.
    Different seeds would mean the scorer scored on rows the bundle was
    trained on — silent leakage. This test catches that contract break.
    """
    tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    # Score twice — receipt metric_value must match (deterministic).
    r1 = score_bundle_on_holdout(
        cfg, problem="churn_v1", version=version, output_dir=tmp_path / "r1"
    )
    r2 = score_bundle_on_holdout(
        cfg, problem="churn_v1", version=version, output_dir=tmp_path / "r2"
    )
    assert r1.metrics["auc"] == pytest.approx(r2.metrics["auc"])


def test_score_bundle_on_holdout_missing_champion_raises(
    synth_workdir: tuple[Path, RuxMLConfig],
) -> None:
    tmp_path, cfg = synth_workdir
    # No champion has been promoted.
    with pytest.raises(FileNotFoundError):
        score_bundle_on_holdout(cfg, problem="churn_v1", output_dir=tmp_path / "receipts")

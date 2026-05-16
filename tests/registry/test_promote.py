"""End-to-end promotion test (per PR-010 verification criteria)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline

from rux_ml.config import (
    DataConfig,
    FeaturesConfig,
    KFoldCV,
    RegistryConfig,
    RunsConfig,
    RuxMLConfig,
    TrainingConfig,
)
from rux_ml.config.features import FeaturesSpec
from rux_ml.registry import load_model
from rux_ml.registry.champion import read_champion
from rux_ml.registry.paths import champion_path, version_dir
from rux_ml.registry.promote import promote, rollback
from rux_ml.runs import TrialAttrs, data_hashes, one_off_run


def _make_cfg(tmp_path: Path, source: Path, registry_root: Path) -> RuxMLConfig:
    return RuxMLConfig(
        data=DataConfig(source_path=source, target_column="y"),
        features=FeaturesConfig(spec=FeaturesSpec(numeric_columns=["x1", "x2"])),
        training=TrainingConfig(
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


def _populate_trial(cfg: RuxMLConfig) -> tuple[str, int]:
    """Create a one-off trial with full provenance; return (study_name, trial_number)."""
    assert cfg.data.source_path is not None
    hashes = data_hashes(cfg.data.source_path)
    with one_off_run(cfg, problem="churn_v1", study="wide") as run:
        TrialAttrs.from_cfg(cfg, hashes, metric="auc", best_iteration=4).record(run.trial)
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


def test_promote_writes_bundle_and_champion(synth_workdir: tuple[Path, RuxMLConfig]) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg)

    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    # Bundle directory contains all three files.
    vdir = version_dir(cfg.registry.root, "churn_v1", version)
    assert (vdir / "pipeline.skops").exists()
    assert (vdir / "model.ubj").exists()
    assert (vdir / "manifest.json").exists()

    # Champion.json points at the new version.
    champ = read_champion(champion_path(cfg.registry.root, "churn_v1"))
    assert champ["version"] == version
    assert champ["promoted_from"]["study"] == study_name
    assert champ["promoted_from"]["trial_number"] == trial_number


def test_load_model_returns_pipeline_and_booster_and_predict_works(
    synth_workdir: tuple[Path, RuxMLConfig],
) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg)
    promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    pipeline, booster = load_model("churn_v1", registry_root=cfg.registry.root)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(booster, xgb.Booster)

    # End-to-end predict on a sample.
    assert cfg.data.source_path is not None
    df = pl.read_parquet(cfg.data.source_path).head(5)
    x_raw = df.drop("y")
    x_transformed = pipeline.transform(x_raw)  # pyright: ignore[reportUnknownMemberType]
    x_pd = x_transformed.to_pandas() if hasattr(x_transformed, "to_pandas") else x_transformed
    dmatrix = xgb.DMatrix(x_pd)
    preds = booster.predict(dmatrix)
    assert preds.shape == (5,)


def test_load_model_with_explicit_version(synth_workdir: tuple[Path, RuxMLConfig]) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg)
    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    pipeline, booster = load_model("churn_v1", version=version, registry_root=cfg.registry.root)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(booster, xgb.Booster)


def test_promote_refuses_trial_with_missing_provenance(
    synth_workdir: tuple[Path, RuxMLConfig],
) -> None:
    """A trial whose user_attrs are incomplete must NOT promote (PR-009 contract)."""
    _tmp_path, cfg = synth_workdir

    # Create a study with a half-recorded trial (only `metric` set; full TrialAttrs missing).
    storage_url = cfg.runs.storage_url
    from rux_ml.runs.provenance import ensure_storage_parent  # noqa: PLC0415

    ensure_storage_parent(storage_url)
    study = optuna.create_study(study_name="bad_study", storage=storage_url)
    trial = study.ask()
    trial.set_user_attr("metric", "auc")  # incomplete schema
    study.tell(trial, 0.5)

    with pytest.raises(Exception):  # noqa: B017 — pydantic.ValidationError surfaces here
        promote(cfg, problem="churn_v1", study_name="bad_study", trial_number=0)


def test_rollback_atomically_updates_champion(synth_workdir: tuple[Path, RuxMLConfig]) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg)
    v1 = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    # Manually fabricate v2 = v1 with a different version-id, copying the bundle on disk.
    import shutil  # noqa: PLC0415

    v1_dir = version_dir(cfg.registry.root, "churn_v1", v1)
    v2_id = v1.replace(v1[-6:], "ffffff")
    v2_dir = version_dir(cfg.registry.root, "churn_v1", v2_id)
    shutil.copytree(v1_dir, v2_dir)
    # Update champion to v2 manually so we can roll back to v1 below.
    from rux_ml.registry.champion import write_champion  # noqa: PLC0415

    write_champion(
        champion_path(cfg.registry.root, "churn_v1"),
        version=v2_id,
        promoted_from_study=study_name,
        promoted_from_trial_number=trial_number,
        metric_value=0.91,
    )

    rollback(cfg, problem="churn_v1", version=v1)
    champ = read_champion(champion_path(cfg.registry.root, "churn_v1"))
    assert champ["version"] == v1


def test_rollback_rejects_nonexistent_version(synth_workdir: tuple[Path, RuxMLConfig]) -> None:
    _tmp_path, cfg = synth_workdir
    with pytest.raises(FileNotFoundError, match="no manifest"):
        rollback(cfg, problem="churn_v1", version="v_does_not_exist")


# ---------- Inference-deps separation test (sub-decision B1) ----------


_INFERENCE_DEPS_PROBE = """
import sys
# Block the training-stack sub-packages BEFORE importing rux_ml.registry.
# If load_model transitively pulls any of these in, import will fail.
for blocked in (
    'rux_ml.tuning',
    'rux_ml.tuning.objective',
    'rux_ml.tuning.samplers',
    'rux_ml.tuning.pruners',
    'rux_ml.tuning.study',
    'rux_ml.tuning.isolation',
    'rux_ml.data.cv',
    'rux_ml.training',
    'rux_ml.training.factory',
    'rux_ml.training.metrics',
    'rux_ml.training.ingest',
    'rux_ml.training.protocol',
    'optuna',
    'optuna_integration',
    'skfolio',
):
    sys.modules[blocked] = None  # poison: subsequent `import <blocked>` will fail
import rux_ml.registry
# load_model + manifest schema accessible:
assert rux_ml.registry.load_model
assert rux_ml.registry.ModelManifest
print('OK')
"""


def test_registry_inference_deps_separation() -> None:
    """``import rux_ml.registry`` does NOT transitively pull in the training stack."""
    result = subprocess.run(
        [sys.executable, "-c", _INFERENCE_DEPS_PROBE],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"registry imports leaked to training stack:\nstderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )
    assert "OK" in result.stdout

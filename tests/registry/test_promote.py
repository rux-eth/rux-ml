"""End-to-end promotion test (per PR-010 verification criteria)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import pytest
import xgboost as xgb
from sklearn.pipeline import Pipeline

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
from rux_ml.data.bridge import HarnessManifestError
from rux_ml.registry import load_model
from rux_ml.registry.champion import read_champion
from rux_ml.registry.manifest import HarnessManifestRef
from rux_ml.registry.manifest import read as read_manifest
from rux_ml.registry.paths import champion_path, version_dir
from rux_ml.registry.promote import PromoteDataHashError, promote, rollback
from rux_ml.runs import TrialAttrs, data_hashes, one_off_run
from tests.conftest import repo_oracle_cfg, write_harness_manifest


def _make_cfg(tmp_path: Path, source: Path, registry_root: Path) -> RuxMLConfig:
    return RuxMLConfig(
        data=DataConfig(source_path=source, target_column="y", oracle=repo_oracle_cfg()),
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
    hashes = data_hashes(cfg.data.source_path, oracle=cfg.data.oracle)
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


def test_promote_writes_bundle_and_champion(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)

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


@pytest.mark.parametrize("how", ["rewritten", "other_source"])
def test_promote_refuses_a_refit_whose_data_hash_differs_from_the_trials(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
    how: str,
) -> None:
    """PR-050 (PR-040 successor (i)): the re-fit's ``data_hash`` must equal the one the
    trial recorded. ``data.source_path`` is hash-elided, so a trial can be promoted by
    re-fitting on other data — the file rewritten in place, or the config pointed at
    another file. Refused with a named error before any bundle is written."""
    tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    assert cfg.data.source_path is not None
    recorded = data_hashes(cfg.data.source_path, oracle=cfg.data.oracle)["data_hash"]
    changed = pl.read_parquet(cfg.data.source_path).with_columns(pl.col("x1") * 2.0)
    if how == "rewritten":
        changed.write_parquet(cfg.data.source_path)
    else:
        other = tmp_path / "other.parquet"
        changed.write_parquet(other)
        cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"source_path": other})})
    refit = data_hashes(cfg.data.source_path, oracle=cfg.data.oracle)["data_hash"]
    assert refit != recorded

    with pytest.raises(PromoteDataHashError) as exc:
        promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    assert isinstance(exc.value, ValueError)  # the CLI's exit-2 family
    assert recorded in str(exc.value)
    assert refit in str(exc.value)
    assert f"{study_name}#{trial_number}" in str(exc.value)
    assert not (cfg.registry.root / "churn_v1").exists()


def test_promote_manifest_data_hash_equals_the_trials(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    """The equal case: the bundle's ``data_hash`` is the trial's recorded one."""
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    run = optuna.load_study(study_name=study_name, storage=cfg.runs.storage_url).trials[
        trial_number
    ]
    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)
    manifest = read_manifest(version_dir(cfg.registry.root, "churn_v1", version) / "manifest.json")
    assert manifest.data_hash == run.user_attrs["data_hash"]


@pytest.fixture
def harness_workdir(tmp_path: Path) -> tuple[Path, RuxMLConfig]:
    """A harness-shaped set (program PR-024): ``<set>/fill/<day>.parquet`` with the
    set's ``manifest.json`` and the ``fill.manifest.json`` view; the source is ``fill/``."""
    root = tmp_path / "training" / "3f1c0a9e2b7d-8a4e6c2f0d1b"
    (root / "fill").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for day in ("2026-05-01", "2026-05-02"):
        x1, x2 = rng.normal(size=100), rng.normal(size=100)
        y = ((0.7 * x1 + 0.3 * x2) > 0).astype(int)
        frame = {"x1": x1.tolist(), "x2": x2.tolist(), "y": y.tolist()}
        pl.DataFrame(frame).write_parquet(root / "fill" / f"{day}.parquet")
    write_harness_manifest(root, ["fill"])
    return root, _make_cfg(tmp_path, root / "fill", tmp_path / "registry")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("carries_data_hash", [False, True])
def test_promote_records_the_harness_manifest_id(
    harness_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
    carries_data_hash: bool,
) -> None:
    """PR-051 (program D45 #6, C11): a model trained on a harness set records the id of
    the set's provenance manifest — the sha256 of ``manifest.json``, the id the harness's
    own view cites — with the set, the subtree, the view's sha256 and the manifest's copy
    of rux-ml's ``data_hash`` (equal to the model's when present)."""
    root, cfg = harness_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    assert cfg.data.source_path is not None
    data_hash = data_hashes(cfg.data.source_path, oracle=cfg.data.oracle)["data_hash"]
    if carries_data_hash:
        write_harness_manifest(root, ["fill"], ruxml_data_hash={"fill": data_hash})

    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)

    manifest = read_manifest(version_dir(cfg.registry.root, "churn_v1", version) / "manifest.json")
    assert manifest.harness_manifest == HarnessManifestRef(
        manifest_id=_sha256(root / "manifest.json"),
        set_name=root.name,
        subtree="fill",
        view_sha256=_sha256(root / "fill.manifest.json"),
        ruxml_data_hash=data_hash if carries_data_hash else None,
    )
    assert manifest.data_hash == data_hash


@pytest.mark.parametrize("defect", ["stale_view", "other_subtree", "other_data_hash", "no_set"])
def test_promote_refuses_a_harness_manifest_that_does_not_describe_the_set(
    harness_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
    defect: str,
) -> None:
    """The id is recorded only if it is the id of this set: the view must cite the set
    manifest as it is (not a rewrite), name this subtree, and the manifest's copy of
    rux-ml's ``data_hash`` (if any) must equal the re-fit's. Refused before any bundle."""
    root, cfg = harness_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    view_path, set_path = root / "fill.manifest.json", root / "manifest.json"
    if defect == "stale_view":  # the set manifest rewritten, the view not
        body = json.loads(set_path.read_text())
        set_path.write_text(json.dumps({**body, "days_in_window_missing": ["2026-05-03"]}))
    elif defect == "other_subtree":
        view = json.loads(view_path.read_text())
        view_path.write_text(json.dumps({**view, "subtree": "walk"}))
    elif defect == "other_data_hash":
        write_harness_manifest(root, ["fill"], ruxml_data_hash={"fill": "0" * 16 + "|" + "0" * 16})
    else:  # the view names a set manifest that is not there
        set_path.unlink()

    with pytest.raises(HarnessManifestError, match="harness manifest"):
        promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)
    assert not (cfg.registry.root / "churn_v1").exists()


def test_promote_records_no_harness_manifest_for_a_plain_source(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    """A source with no harness view (a non-M9 problem) records ``harness_manifest: null``."""
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)
    manifest = read_manifest(version_dir(cfg.registry.root, "churn_v1", version) / "manifest.json")
    assert manifest.harness_manifest is None


def test_load_model_returns_pipeline_and_booster_and_predict_works(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
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


def test_load_model_with_explicit_version(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
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


def test_rollback_atomically_updates_champion(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    _tmp_path, cfg = synth_workdir
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
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


def test_promote_records_the_in_memory_refit(
    synth_workdir: tuple[Path, RuxMLConfig],
    seed_bag: SeedBag,
    env_versions: EnvironmentVersions,
) -> None:
    """PR-059 on the in-memory path: the re-fit's rows are its own split's, the served range
    is the loaded booster's best iteration + 1 (or every tree), the device is the fit's; a
    trial with no fold_meta.json ``oos`` record promotes with ``oos`` / its sha256 None."""
    from rux_ml.data import load_parquet, make_splits, materialize  # noqa: PLC0415
    from rux_ml.registry.bundle import load_bundle  # noqa: PLC0415

    _tmp_path, cfg = synth_workdir
    assert cfg.data.source_path is not None
    study_name, trial_number = _populate_trial(cfg, seed_bag, env_versions)
    version = promote(cfg, problem="churn_v1", study_name=study_name, trial_number=trial_number)
    _, booster, manifest = load_bundle(version_dir(cfg.registry.root, "churn_v1", version))

    splits = make_splits(
        cfg, materialize(load_parquet(cfg.data.source_path, oracle=cfg.data.oracle)),
        seed=seed_bag.split_seed,
    )  # fmt: skip
    refit = manifest.refit
    assert refit is not None
    assert (refit.train_rows, refit.val_rows) == (splits["train"].height, splits["val"].height)
    best = booster.attr("best_iteration")
    want = (0, 0) if best is None else (0, int(best) + 1)
    assert refit.iteration_range == want
    assert refit.best_iteration == (None if best is None else int(best))
    assert refit.device == "cpu"
    assert manifest.oos is None and manifest.oos_export_sha256 is None

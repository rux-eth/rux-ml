"""End-to-end XGBoost golden regression test (per PR-014).

Two surfaces:

1. ``test_golden_xgb_baseline_in_process`` — fits the workbench's pipeline
   (load_parquet → train_val_test_split → make_features → make_trainer →
   fit → predict_proba) on a committed fixed-seed synthetic dataset and
   compares predictions + AUC against committed fixtures with tolerance
   bands. Catches regressions in XGBoost / sklearn / numpy / polars.
2. ``test_golden_load_model_matches_in_process`` — promotes the trained
   pipeline into a temporary registry, loads it back via
   :func:`rux_ml.registry.load_model`, and compares the registry-loaded
   predictions against the in-process predictions within the same
   tolerance. Catches regressions in the skops / ubj serialization round
   trip (PR-010 surface) that an in-process-only test would miss.

**This test will fail when CUDA / XGBoost / sklearn / numpy / polars versions
change** — that's a feature, not a bug. The failure message in
:func:`assert_predictions_close` lists the investigation procedure. The
short version:

1. Diff ``manifest.json`` library versions vs the current environment.
2. Run ``uv run pytest tests/integration/test_determinism_cpu.py`` to
   confirm the PR-013 CPU bit-exact contract still holds — if THAT fails,
   the determinism contract regressed and we should not regenerate.
3. Only after both check out, run ``make regenerate-golden`` and commit
   the refreshed fixtures with a manual diff of ``manifest.json``.

Per ``docs/CONSTRAINTS.md`` Tolerance-Based Golden Tests Only: no exact
hashes, no ``assert_array_equal`` against fixtures.
"""

from __future__ import annotations

import importlib.metadata as _metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from rux_ml._internal.env import get_versions
from rux_ml._internal.seeds import make_seed_bag
from rux_ml.config import (
    DataConfig,
    FeaturesConfig,
    KFoldCV,
    RegistryConfig,
    RunsConfig,
    RuxMLConfig,
    TuningConfig,
    XGBoostTraining,
)
from rux_ml.config.features import FeaturesSpec
from rux_ml.data import load_parquet, materialize, train_val_test_split
from rux_ml.features import cardinalities_from, make_features
from rux_ml.registry import load_model
from rux_ml.registry.promote import promote
from rux_ml.runs import TrialAttrs, data_hashes, one_off_run
from rux_ml.training import make_trainer

from .conftest import (
    DEFAULT_ATOL,
    DEFAULT_METRIC_BAND,
    DEFAULT_RTOL,
    GOLDEN_DIR,
    assert_metric_within,
    assert_predictions_close,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = pytest.mark.golden

# Locked master entropy for the golden test. Changing this value invalidates
# every committed fixture in golden_v1/ — only change deliberately with a
# coordinated `make regenerate-golden` + manifest diff review.
GOLDEN_MASTER_ENTROPY: int = 42

# Dataset construction parameters — see _build_synthetic_dataset. These are
# part of the golden contract; changing any forces a regen.
_DATASET_ROWS = 200
_DATASET_NUMERIC = 3
_CATEGORICAL_LEVELS = 3


# ---------------------------------------------------------------------------
# Synthetic dataset construction (deterministic from GOLDEN_MASTER_ENTROPY)
# ---------------------------------------------------------------------------


def _build_synthetic_dataset() -> pl.DataFrame:
    """Construct the golden synthetic dataset deterministically.

    Output schema: ``x1, x2, x3`` (float64) + ``cat`` (categorical, 3 levels) +
    ``y`` (int, binary target). 200 rows. The construction is keyed off
    ``GOLDEN_MASTER_ENTROPY`` so two regen calls produce identical bytes.
    """
    rng = np.random.default_rng(GOLDEN_MASTER_ENTROPY)
    x_arr, y_arr = make_classification(
        n_samples=_DATASET_ROWS,
        n_features=_DATASET_NUMERIC,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.05,
        random_state=GOLDEN_MASTER_ENTROPY,
    )
    x_numeric = cast("NDArray[np.float64]", x_arr)
    y_int = cast("NDArray[np.int_]", y_arr)
    cat_levels = ["a", "b", "c"][:_CATEGORICAL_LEVELS]
    cat_col = rng.choice(cat_levels, size=_DATASET_ROWS)
    return pl.DataFrame(
        {
            "x1": x_numeric[:, 0].tolist(),
            "x2": x_numeric[:, 1].tolist(),
            "x3": x_numeric[:, 2].tolist(),
            "cat": cat_col.tolist(),
            "y": y_int.tolist(),
        }
    )


def _golden_cfg(synthetic_path: Path, tmp_path: Path) -> RuxMLConfig:
    """Build the locked golden ``RuxMLConfig``.

    Every field here is part of the golden contract — changing any value
    invalidates the committed fixtures. The `cv` slot is required by the
    config schema but unused (this test does single train/val/test, not CV).
    """
    return RuxMLConfig(
        data=DataConfig(source_path=synthetic_path, target_column="y"),
        features=FeaturesConfig(
            # Threshold > 3 so the 3-level `cat` column passes through to
            # XGBoost's native categorical handling (no NestedCVWrapper target
            # encoding). Locks the encoding decision rule used for golden_v1.
            categorical_low_card_threshold=10,
            spec=FeaturesSpec(numeric_columns=["x1", "x2", "x3"], categorical_columns=["cat"]),
        ),
        training=XGBoostTraining(
            kind="xgboost",
            device="cpu",
            tree_method="hist",
            enable_categorical=True,
            metric="auc",
            n_estimators=20,
            max_depth=3,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=None,
            model_kwargs={"n_jobs": 1},  # belt-and-suspenders with OMP_NUM_THREADS=1
        ),
        runs=RunsConfig(
            storage_url=f"sqlite:///{tmp_path}/studies/studies.db",
            artifacts_root=tmp_path / "studies/artifacts",
        ),
        registry=RegistryConfig(root=tmp_path / "registry"),
        cv=KFoldCV(n_splits=3, shuffle=True),
        tuning=TuningConfig(entropy=GOLDEN_MASTER_ENTROPY),
    )


# ---------------------------------------------------------------------------
# In-process pipeline fit + predict
# ---------------------------------------------------------------------------


def _fit_predict_in_process(cfg: RuxMLConfig) -> tuple[NDArray[np.float64], float]:
    """Run the workbench pipeline once and return ``(predict_proba[:, 1], auc)``.

    Mirrors ``cli/train.py:_fit_and_score`` but returns the prediction array
    so the golden assertion can compare element-wise.
    """
    assert cfg.data.source_path is not None
    assert cfg.data.target_column is not None
    bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=0)

    df = materialize(load_parquet(cfg.data.source_path))
    splits = train_val_test_split(df, ratios=cfg.data.split_ratios, seed=bag.split_seed)
    x_train = splits["train"].drop(cfg.data.target_column)
    y_train = splits["train"][cfg.data.target_column]
    x_val = splits["val"].drop(cfg.data.target_column)
    y_val = splits["val"][cfg.data.target_column]

    cards = cardinalities_from(x_train, cfg.features.spec.categorical_columns)
    pipeline = make_features(cfg.features, cardinalities=cards if cards else None)
    pipeline.fit(x_train, y_train.to_numpy())  # pyright: ignore[reportUnknownMemberType]
    x_train_t = cast("pl.DataFrame", pipeline.transform(x_train))  # pyright: ignore[reportUnknownMemberType]
    x_val_t = cast("pl.DataFrame", pipeline.transform(x_val))  # pyright: ignore[reportUnknownMemberType]

    trainer = make_trainer(cfg.training, seed=bag.xgb_seed)
    trainer.fit(
        x_train_t.to_pandas(),
        y_train.to_numpy(),
        eval_set=[(x_val_t.to_pandas(), y_val.to_numpy())],
        verbose=False,
    )

    # Use predict_proba[:, 1] for binary classification — single class
    # probability column is the standard golden surface for AUC.
    preds = cast(
        "NDArray[np.float64]",
        trainer.predict_proba(x_val_t.to_pandas()),  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    )[:, 1].astype(np.float64)
    auc = float(roc_auc_score(y_val.to_numpy(), preds))
    return preds, auc


# ---------------------------------------------------------------------------
# Manifest construction (regen path)
# ---------------------------------------------------------------------------


def _build_manifest(cfg: RuxMLConfig, data_path: Path) -> dict[str, object]:
    """Snapshot the env + cfg state that produced the current fixtures."""
    versions = get_versions(cfg.memory)
    hashes = data_hashes(data_path)
    # Use the cfg_hash helpers via TrialAttrs to keep the manifest aligned
    # with the per-trial provenance schema (root_cfg_hash specifically).
    from rux_ml.config import cfg_hash  # noqa: PLC0415

    return {
        "fixture_version": "golden_v1",
        "master_entropy": GOLDEN_MASTER_ENTROPY,
        "entropy_hex": make_seed_bag(
            master_entropy=GOLDEN_MASTER_ENTROPY, trial_number=0
        ).entropy_hex,
        "data_hash": hashes["data_hash"],
        "data_bytes_hash": hashes["data_bytes_hash"],
        "data_logical_hash": hashes["data_logical_hash"],
        "root_cfg_hash": cfg_hash(cfg),
        "library_versions": {
            "xgboost": _metadata.version("xgboost"),
            "scikit_learn": _metadata.version("scikit-learn"),
            "numpy": _metadata.version("numpy"),
            "polars": _metadata.version("polars"),
            "skops": _metadata.version("skops"),
        },
        "cuda_runtime_version": versions.cuda_runtime_version,
        "tolerance": {
            "preds_atol": DEFAULT_ATOL,
            "preds_rtol": DEFAULT_RTOL,
            "metric_abs_tol": DEFAULT_METRIC_BAND,
        },
        "dataset": {
            "rows": _DATASET_ROWS,
            "numeric_columns": _DATASET_NUMERIC,
            "categorical_levels": _CATEGORICAL_LEVELS,
        },
        "created_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Regen path
# ---------------------------------------------------------------------------


def _write_fixtures(
    cfg: RuxMLConfig,
    synthetic_path: Path,
    preds: NDArray[np.float64],
    auc: float,
) -> None:
    """Rewrite ``synthetic.parquet`` + ``preds.npy`` + ``metric.json`` + ``manifest.json``."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    # Rewrite the input data so a fresh checkout's regen step is idempotent.
    _build_synthetic_dataset().write_parquet(synthetic_path)
    np.save(GOLDEN_DIR / "preds.npy", preds)
    (GOLDEN_DIR / "metric.json").write_text(json.dumps({"auc": auc}, indent=2) + "\n")
    manifest = _build_manifest(cfg, synthetic_path)
    (GOLDEN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_golden_xgb_baseline_in_process(
    tmp_path: Path,
    regenerate_golden: bool,
) -> None:
    """Predict-proba + AUC for the locked synthetic dataset matches the committed fixtures.

    See module docstring for the failure investigation procedure. Under
    ``--regenerate-golden``, this test instead rewrites the committed
    fixtures (preds.npy + metric.json + manifest.json + synthetic.parquet).
    """
    synthetic_path = GOLDEN_DIR / "synthetic.parquet"

    if regenerate_golden:
        # Bootstrap path: regenerate the parquet first so the cfg can load it.
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        _build_synthetic_dataset().write_parquet(synthetic_path)

    cfg = _golden_cfg(synthetic_path, tmp_path)
    preds, auc = _fit_predict_in_process(cfg)

    if regenerate_golden:
        _write_fixtures(cfg, synthetic_path, preds, auc)
        # Verify the write happened (the spec's regen verification criterion).
        assert (GOLDEN_DIR / "preds.npy").exists()
        assert (GOLDEN_DIR / "metric.json").exists()
        assert (GOLDEN_DIR / "manifest.json").exists()
        return

    golden_preds = np.load(GOLDEN_DIR / "preds.npy")
    metric_json = json.loads((GOLDEN_DIR / "metric.json").read_text())
    baseline_auc = float(metric_json["auc"])

    assert_predictions_close(preds, golden_preds)
    assert_metric_within(auc, baseline_auc)


def test_golden_load_model_matches_in_process(
    tmp_path: Path,
    regenerate_golden: bool,
) -> None:
    """Registry round-trip via promote → load_model preserves predictions.

    Promotes the trained pipeline into a temp registry, loads it back, and
    compares the registry-loaded predictions against the in-process
    predictions within the same tolerance envelope. Catches skops / ubj
    serialization regressions that the in-process golden alone would miss.

    Skipped under ``--regenerate-golden``: this is a structural test of the
    PR-010 round-trip, not a fixture-comparison test, and doesn't produce
    artifacts to commit.
    """
    if regenerate_golden:
        pytest.skip("registry round-trip test does not produce fixtures")

    synthetic_path = GOLDEN_DIR / "synthetic.parquet"
    cfg = _golden_cfg(synthetic_path, tmp_path)
    in_process_preds, _ = _fit_predict_in_process(cfg)

    # Promote a trained model to a temp registry — uses the same one_off_run
    # surface as cli/train.py so the round-trip exercises the production path.
    bag = make_seed_bag(master_entropy=cfg.tuning.entropy, trial_number=0)
    versions = get_versions(cfg.memory)
    assert cfg.data.source_path is not None
    hashes = data_hashes(cfg.data.source_path)

    with one_off_run(cfg, problem="golden_v1", study="wide") as run:
        TrialAttrs.from_cfg(
            cfg,
            hashes,
            metric=cfg.training.metric,
            peak_rss_mb=0.0,
            bag=bag,
            versions=versions,
        ).record(run.trial)
        run.tell(0.85)  # arbitrary score for the trial; ignored by promote

    version = promote(
        cfg,
        problem="golden_v1",
        study_name=run.study.study_name,
        trial_number=run.trial.number,
    )
    pipeline, booster = load_model("golden_v1", version=version, registry_root=cfg.registry.root)

    # Reload the val fold the in-process test used, transform via the
    # registry-loaded pipeline, predict via the registry-loaded booster.
    assert cfg.data.target_column is not None
    df = materialize(load_parquet(cfg.data.source_path))
    splits = train_val_test_split(df, ratios=cfg.data.split_ratios, seed=bag.split_seed)
    x_val = splits["val"].drop(cfg.data.target_column)
    x_val_t = cast("pl.DataFrame", pipeline.transform(x_val))  # pyright: ignore[reportUnknownMemberType]

    import xgboost as xgb  # noqa: PLC0415 — only needed for the DMatrix construction

    dmatrix = xgb.DMatrix(x_val_t.to_pandas(), enable_categorical=True)
    roundtrip_preds = cast("NDArray[np.float64]", booster.predict(dmatrix)).astype(np.float64)

    assert_predictions_close(roundtrip_preds, in_process_preds)
